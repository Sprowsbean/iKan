"""Measure the OCR pipeline on fixed test sets, so every change is judged on the same images.

From backend/ (models load on first use, as in the app):
    venv\\Scripts\\python eval/run_eval.py --label baseline
    venv\\Scripts\\python eval/run_eval.py --label after --sets real,synthetic
    venv\\Scripts\\python eval/run_eval.py --compare baseline after      # identical outputs?

Sets (images live outside the repo, in ../../test_images; see eval/sets/README.md):
  jaw_lines   500 single text lines from real photos taken in Japan (JaWildText) - one right answer per line,
              so this is where character error rate (CER) is exact; also feeds the confidence check
  jaw_photos  100 whole JaWildText signboard / poster / product photos (order-free scores)
  real        the developer's own phone photos (curved/words.json) + a manga page and a kanji chart
  synthetic   the generated images of test_images/gt_all.json - a regression guard only, never a target

Scores:
  CER          character errors / reference characters (jaw_lines only)
  exact        lines read exactly right (jaw_lines only)
  kanji words  reference kanji words found in the OCR text (maximal kanji runs, or the listed words)
  JP chars     reference kana/kanji found in the OCR text, as a bag (order-free)
  fake kanji   kanji runs in the OCR text that appear nowhere in the reference (only where the full
               reference text is known)
  s median/max seconds per image for analyze_image(), on 2 CPU cores like the free Space
Every image goes through as_sent() first: the same scaling and margin the app applies before upload.
Each image's output is saved to eval/results/<label>/items.jsonl (gitignored) for --compare.
"""
import argparse
import ctypes
import difflib
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
TEST = BACKEND.parents[1] / "test_images"
RESULTS = HERE / "results"
SETS = ("jaw_lines", "jaw_photos", "real", "synthetic")

KANJI_RUN = re.compile(r"[一-鿿々]+")
JP = re.compile(r"[ぁ-ゖァ-ヶ一-鿿々]")
CONF_BINS = (0.0, 0.3, 0.5, 0.7, 0.9, 1.01)


def use_two_cores() -> None:
    """The free Hugging Face Space has 2 vCPUs; pin this process to 2 cores so timings are comparable."""
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {0, 1})
        elif os.name == "nt":
            k = ctypes.windll.kernel32
            k.SetProcessAffinityMask(k.GetCurrentProcess(), 0b11)
    except Exception as e:  # timings are then for all cores; say so rather than fail
        print(f"warning: could not pin to 2 cores ({e})")


def as_sent(img):
    """The image as the app uploads it (src/components/CropView.tsx): at most 2400 px on the long side, and a
    margin of half the short side (repeated edge pixels) when the short side is 300 px or less."""
    import cv2
    h, w = img.shape[:2]
    if max(h, w) > 2400:
        s = 2400 / max(h, w)
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    if min(h, w) <= 300:
        m = round(min(h, w) / 2)
        img = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_REPLICATE)
    return img


def norm(s: str) -> str:
    return "".join(unicodedata.normalize("NFKC", s).split())


def edits(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---------- test sets: (set, file path, {"lines": [...]} or {"words": [...]} or {"line": "..."}) ----------

def load_sets(names: list[str]) -> list[tuple[str, Path, dict]]:
    items = []
    jaw = TEST / "jawildtext"
    if "jaw_lines" in names:
        gt = json.loads((jaw / "lines.json").read_text(encoding="utf-8"))
        items += [("jaw_lines", jaw / "lines" / f, {"line": t}) for f, t in gt.items()]
    if "jaw_photos" in names:
        gt = json.loads((jaw / "photos.json").read_text(encoding="utf-8"))
        items += [("jaw_photos", jaw / "photos" / f, {"lines": ls}) for f, ls in gt.items()]
    if "real" in names:
        words = json.loads((TEST / "curved" / "words.json").read_text(encoding="utf-8"))
        items += [("real", TEST / "curved" / f, {"words": ws}) for f, ws in words.items()]
        extra = json.loads((TEST / "extra_gt.json").read_text(encoding="utf-8"))
        items += [("real", TEST / f, {"lines": ls}) for f, ls in extra.items() if not f.startswith("_")]
    if "synthetic" in names:
        gt = json.loads((TEST / "gt_all.json").read_text(encoding="utf-8"))
        items += [("synthetic", TEST / f, {"lines": [l for l in t.split("\n") if l.strip()]}) for f, t in gt.items()]
    return items


# ---------- scoring one image ----------

def score(ref: dict, text: str) -> dict:
    got = norm(text)
    s: dict = {}
    if "line" in ref:
        want = norm(ref["line"])
        s["edits"], s["ref_chars"], s["exact"] = edits(got, want), len(want), got == want
        lines = [ref["line"]]
    else:
        lines = ref.get("lines")
    if "words" in ref:
        words = [norm(w) for w in ref["words"]]
    else:
        words = list(dict.fromkeys(r for l in lines for r in KANJI_RUN.findall(norm(l))))
    s["words"], s["words_found"] = len(words), sum(w in got for w in words)
    if lines is not None:  # full reference text known: bag of characters, fake kanji
        want_all = norm("".join(lines))
        ref_jp = Counter(c for c in want_all if JP.match(c))
        got_jp = Counter(c for c in got if JP.match(c))
        s["jp_chars"], s["jp_found"] = sum(ref_jp.values()), sum((ref_jp & got_jp).values())
        s["fake"] = sum(r not in want_all for r in KANJI_RUN.findall(got))
    return s


# ---------- hooks into the pipeline ----------

def raw_reader():
    """The module whose _oriented_pass the pipeline calls (main.py now; engines/meiki.py after the split)."""
    try:
        import engines.meiki as mod  # noqa: F401
    except ImportError:
        import main as mod
    return mod


def capture_segments():
    """Wrap the raw OCR pass so each image's characters and confidences can be checked against the reference."""
    mod = raw_reader()
    original, seen = mod._oriented_pass, []

    def wrapped(img):
        segs = original(img)
        seen.append(segs)
        return segs

    mod._oriented_pass = wrapped
    return seen


def calibration(pairs: list[tuple[list, str]]) -> list[tuple[str, int, float]]:
    """Does the recognizer's confidence track correctness? Each raw character is matched against the
    reference line; returns (confidence range, characters, share correct)."""
    bins = [[0, 0] for _ in CONF_BINS[:-1]]
    for segs, ref in pairs:
        chars = [c for seg in segs for c in seg]
        got = "".join(unicodedata.normalize("NFKC", c["char"]) for c in chars)
        ok = [False] * len(got)
        for m in difflib.SequenceMatcher(None, got, norm(ref), autojunk=False).get_matching_blocks():
            ok[m.a:m.a + m.size] = [True] * m.size
        pos = 0
        for c in chars:
            n = len(unicodedata.normalize("NFKC", c["char"]))
            if n == 0 or not c["char"].strip():
                pos += n
                continue
            b = next(i for i in range(len(bins)) if c["conf"] < CONF_BINS[i + 1])
            bins[b][0] += 1
            bins[b][1] += all(ok[pos:pos + n])
            pos += n
    return [(f"{lo:.1f}-{min(hi, 1):.1f}", n, k / n if n else 0.0)
            for (lo, hi), (n, k) in zip(zip(CONF_BINS, CONF_BINS[1:]), bins)]


# ---------- run, summarize, compare ----------

def run(label: str, names: list[str], limit: int | None) -> None:
    use_two_cores()
    sys.path.insert(0, str(BACKEND))
    os.chdir(BACKEND)
    import cv2
    import main
    seen = capture_segments()
    items = load_sets(names)
    out = RESULTS / label
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    records, cal = [], []
    main.analyze_image(as_sent(cv2.imread(str(items[0][1]))))  # warm-up: model loading is not part of a scan
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for set_name, path, ref in items:
            if limit and counts[set_name] >= limit:
                continue
            counts[set_name] += 1
            img = as_sent(cv2.imread(str(path)))
            seen.clear()
            t = time.perf_counter()
            try:
                res, err = main.analyze_image(img), None
            except Exception as e:  # a crash is a result too (it was HTTP 500 in the app)
                res, err = {"text": "", "cards": []}, f"{type(e).__name__}: {e}"
            dt = time.perf_counter() - t
            rec = {"set": set_name, "file": path.name, "seconds": round(dt, 3), "error": err,
                   "text": res["text"], "cards": [c["kanji"] for c in res["cards"]], **score(ref, res["text"])}
            if "line" in ref and seen:
                cal.append((seen[0], ref["line"]))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
            print(f"{set_name:10} {len(records):4}  {dt:5.2f}s  {path.name[:40]}", flush=True)
    summary = summarize(records, cal)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)


def summarize(records: list[dict], cal: list) -> str:
    rows = ["| set | images | CER | exact | kanji words | JP chars | fake kanji | s median | s max | crashes |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name in SETS:
        rs = [r for r in records if r["set"] == name]
        if not rs:
            continue
        tot = lambda k: sum(r.get(k, 0) for r in rs)
        has = lambda k: any(k in r for r in rs)
        pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "-"
        secs = [r["seconds"] for r in rs]
        rows.append(" | ".join([
            f"| {name}", str(len(rs)),
            pct(tot("edits"), tot("ref_chars")) if has("edits") else "-",
            pct(tot("exact"), len(rs)) if has("exact") else "-",
            f"{tot('words_found')}/{tot('words')} ({pct(tot('words_found'), tot('words'))})",
            f"{tot('jp_found')}/{tot('jp_chars')} ({pct(tot('jp_found'), tot('jp_chars'))})" if has("jp_chars") else "-",
            str(tot("fake")) if has("fake") else "-",
            f"{statistics.median(secs):.2f}", f"{max(secs):.2f}",
            str(sum(1 for r in rs if r["error"])) + " |"]))
    text = "\n".join(rows)
    if cal:
        text += "\n\nRecognizer confidence vs correctness (raw characters, jaw_lines):\n\n" \
                "| confidence | characters | correct |\n|---|---|---|\n" + \
                "\n".join(f"| {r} | {n} | {100 * k:.1f}% |" for r, n, k in calibration(cal))
    return text + "\n"


def compare(a: str, b: str) -> None:
    load = lambda l: {(r["set"], r["file"]): r for r in map(json.loads, open(RESULTS / l / "items.jsonl", encoding="utf-8"))}
    ra, rb = load(a), load(b)
    keys = sorted(set(ra) | set(rb))
    diff = [k for k in keys if k not in ra or k not in rb or
            (ra[k]["text"], ra[k]["cards"], ra[k]["error"]) != (rb[k]["text"], rb[k]["cards"], rb[k]["error"])]
    print(f"{len(keys) - len(diff)} of {len(keys)} images give identical text, cards and errors")
    for k in diff[:20]:
        print(" differs:", k, "|", ra.get(k, {}).get("text", "(missing)")[:60], "|", rb.get(k, {}).get("text", "(missing)")[:60])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--limit", type=int, help="at most this many images per set (quick checks)")
    ap.add_argument("--compare", nargs=2, metavar=("LABEL_A", "LABEL_B"))
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        run(a.label, [s for s in a.sets.split(",") if s], a.limit)
