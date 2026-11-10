"""Photo -> lines of text: the OCR engine, then the clean-up steps. The cards are made from these lines."""
import importlib
import os

from jp import HIRA
from postprocess import candidates, furigana
from postprocess.filters import is_junk, keep_char
from postprocess.lookalike import fix_lookalikes
from preprocess import flatten, orient, scale

# Pipeline switches, so each step can be measured on its own (eval/run_eval.py --opt key=value).
# The defaults are what the app runs. Environment variables IKAN_<KEY> override them (e.g. IKAN_ENGINE).
OPTIONS: dict[str, str] = {
    "engine": "meiki",    # meiki | paddle_v6 | ndl_lite (engines/)
    "filters": "1",       # fake-kanji filters (postprocess/filters.py); tuned on meikiocr's confidences
    "upscale": "0",       # re-read small text at 2x (preprocess/scale.py)
    "tiles": "0",         # also read big photos in 2x2 overlapping tiles (preprocess/scale.py)
    "orient": "0",        # 0 | always | check | retry: turn sideways / upside-down photos upright (preprocess/orient.py)
    "flatten": "0",       # 0 | always | retry: UVDoc flattening of curved surfaces (preprocess/flatten.py)
    "furigana": "0",      # small kana beside/above a kanji line = a reading, not text (postprocess/furigana.py)
    "correct": "0",       # unsure kanji -> recognizer alternate / look-alike, if JMdict agrees (candidates.py)
    "variants": "0",      # old forms (戰) get their modern form (戦) on the card (postprocess/variants.py)
    "unsure": "0",        # cards list the characters the reader was unsure of, with alternatives
}
OPTIONS.update({k: os.environ[f"IKAN_{k.upper()}"] for k in OPTIONS if f"IKAN_{k.upper()}" in os.environ})


def on(key: str) -> bool:
    return OPTIONS.get(key, "0") not in ("0", "", "off", "false")


def segments(img):
    """The selected engine's raw reading (lines of characters with confidence and box), with the
    optional text-size steps."""
    read = importlib.import_module(f"engines.{OPTIONS['engine']}").read
    turn = OPTIONS.get("orient", "0")
    if turn in ("1", "always"):
        img = orient.upright(img)
    mode = OPTIONS.get("flatten", "0")
    if mode == "always":
        img = flatten.flatten(img)
    segs = read(img)
    if turn == "check" or (turn == "retry" and scale.mean_conf(segs) < scale.WEAK_BELOW):
        # ask the orientation classifier (always for "check", after a weak first reading for "retry");
        # keep the turned reading only if the recognizer is also more confident in it
        turned = orient.upright(img)
        if turned is not img:
            segs2 = read(turned)
            if scale.mean_conf(segs2) > scale.mean_conf(segs):
                img, segs = turned, segs2
    if mode == "retry" and scale.mean_conf(segs) < scale.WEAK_BELOW:
        flat = flatten.flatten(img)
        segs2 = read(flat)
        if scale.mean_conf(segs2) > scale.mean_conf(segs):
            img, segs = flat, segs2
    if on("upscale"):
        segs = scale.with_upscale(read, img, segs)
    if on("tiles"):
        segs = scale.with_tiles(read, img, segs)
    return segs


# A line of text with its characters: (text, [{"char", "conf", "alts"?, "was"?}, ...]), one entry per character,
# so the cards can say which characters the reader was unsure of.
Line = tuple[str, list[dict]]
LAST_HINTS: list[tuple[str, str]] = []  # furigana removed from the last image read: (reading, base line)


def ocr_detail(img) -> list[Line]:
    segs = segments(img)
    LAST_HINTS.clear()
    if on("furigana"):
        segs, hints = furigana.split(segs)
        LAST_HINTS.extend(hints)
    pieces: list[Line] = []
    for seg in segs:
        if on("filters") and is_junk(seg):
            continue
        if on("correct"):
            seg = candidates.correct(seg)
        # drop an unsure kanji and break the line there, so its neighbours don't glue into a fake word
        run: list[dict] = []
        for c in seg + [None]:
            if c is None or c["char"].isspace() or (on("filters") and not keep_char(c, seg)):
                if run:
                    text = fix_lookalikes("".join(x["char"] for x in run))
                    pieces.append((text, [{**x, "char": ch} for x, ch in zip(run, text)]))
                run = []
            else:
                run.append(c)
    # the lower threshold yields overlapping boxes ("焼き魚定食" inside "焼き魚定食八百円"): keep the longer
    first = {}
    for text, chars in pieces:
        first.setdefault(text, chars)
    return [(t, cs) for t, cs in first.items() if not any(t != u and t in u for u in first)]


def ocr_lines(img) -> list[str]:
    return [t for t, _ in ocr_detail(img)]


SENTENCE_END = tuple("。！？!?」』）)")


def join_wrapped(lines: list[Line]) -> list[Line]:
    """Rejoin a sentence wrapped mid-word (最近の子供たちは夜更 / かしです。 -> ...夜更かしです。).
    A line that starts with hiragana continues the previous one: a new item or sentence almost never does."""
    out: list[Line] = []
    for text, chars in lines:
        if out and HIRA.match(text) and not out[-1][0].endswith(SENTENCE_END):
            out[-1] = (out[-1][0] + text, out[-1][1] + chars)
        else:
            out.append((text, chars))
    return out


def read_detail(img) -> list[Line]:
    """BGR image -> the text lines the cards are made from, each with its characters' confidence."""
    return join_wrapped(ocr_detail(img))


def read_lines(img) -> list[str]:
    """BGR image -> the text lines the cards are made from."""
    return [t for t, _ in read_detail(img)]
