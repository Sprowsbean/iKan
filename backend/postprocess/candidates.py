"""Dictionary-checked correction (Phase 3 of docs/OCR_ROADMAP.md): 先土 -> 先生.

Rules, from the research notes ("never invent text"):
- only a character the recognizer itself was unsure of is touched (conf < LOW_CONF);
- the replacement must be one of the recognizer's own runner-up guesses at that spot, or a known look-alike
  (Lars Yencken's kanji similarity sets, CC BY 3.0, data/kanji_lookalikes.tsv);
- it is accepted only if it turns a kanji run that is not a dictionary word into fewer, longer words
  (JMdict), e.g. 先|土 (two single kanji) -> 先生 (one word). Text that already reads as words is left alone.
"""
import os
import sqlite3
from functools import lru_cache
from pathlib import Path

from jp import KANJI
from postprocess.charset import japanese

# backend/eval/BASELINE.md: on real photos (JaWildText lines) characters below 0.5 confidence are right
# less than 14% of the time; above it, 39-76%. So only those are candidates for a change.
LOW_CONF = 0.5
MAX_WORD = 8          # longest dictionary word tried when splitting a run
LOOKALIKE_PRIOR = 0.01  # a look-alike the recognizer did not propose ranks below any of its own guesses

_DATA = Path(__file__).resolve().parents[1] / "data" / "kanji_lookalikes.tsv"


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    """Every written form in JMdict (about 195,000), loaded once from jamdict's own database."""
    import jamdict_data
    db = os.path.join(os.path.dirname(jamdict_data.__file__), "jamdict.db")
    with sqlite3.connect(db) as c:
        return frozenset(r[0] for r in c.execute("SELECT text FROM Kanji"))


@lru_cache(maxsize=1)
def _lookalikes() -> dict[str, str]:
    out = {}
    for line in _DATA.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            k, _, v = line.partition("\t")
            out[k] = v
    return out


def pieces(run: str) -> int:
    """Fewest dictionary words (a lone kanji counts as one piece) that make up the run."""
    words = _words()
    best = [0] + [len(run) + 1] * len(run)
    for i in range(1, len(run) + 1):
        for j in range(max(0, i - MAX_WORD), i):
            if (i - j == 1 or run[j:i] in words) and best[j] + 1 < best[i]:
                best[i] = best[j] + 1
    return best[-1]


def _candidates(c: dict) -> list[tuple[str, float]]:
    """Kanji the recognizer proposed at this spot (with its probability), then known look-alikes."""
    out = {a["char"]: a["conf"] for a in c.get("alts", [])[1:] if KANJI.fullmatch(a["char"]) and japanese(a["char"])}
    for ch in _lookalikes().get(c["char"], ""):
        out.setdefault(ch, LOOKALIKE_PRIOR)
    out.pop(c["char"], None)
    return sorted(out.items(), key=lambda x: -x[1])


def correct(seg: list[dict]) -> list[dict]:
    """Return the line with unsure kanji replaced where the dictionary clearly prefers a candidate.
    A changed character keeps the original in "was" (shown to the learner as "printed as" / "or")."""
    seg = [dict(c) for c in seg]
    i = 0
    while i < len(seg):
        if not KANJI.match(seg[i]["char"]):
            i += 1
            continue
        j = i
        while j < len(seg) and KANJI.match(seg[j]["char"]):
            j += 1
        _fix_run(seg, i, j)
        i = j
    return seg


def _fix_run(seg: list[dict], i: int, j: int) -> None:
    if j - i < 2:
        return
    for _ in range(2):  # at most two changes per run
        run = "".join(c["char"] for c in seg[i:j])
        now = pieces(run)
        if now == 1:
            return
        best = None
        for k in range(i, j):
            if seg[k]["conf"] >= LOW_CONF:
                continue
            for ch, p in _candidates(seg[k]):
                new = run[:k - i] + ch + run[k - i + 1:]
                n = pieces(new)
                if n < now and (best is None or (n, -p) < (best[0], -best[1])):
                    best = (n, p, k, ch)
        if best is None:
            return
        _, _, k, ch = best
        # confidence stays the recognizer's own, so the later filters treat the spot exactly as before
        seg[k] = {**seg[k], "was": seg[k]["char"], "char": ch}
