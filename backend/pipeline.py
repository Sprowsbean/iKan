"""Photo -> lines of text: the OCR engine, then the clean-up steps. The cards are made from these lines."""
import importlib
import os

from jp import HIRA
from postprocess.filters import is_junk, keep_char
from postprocess.lookalike import fix_lookalikes
from preprocess import orient, scale

# Pipeline switches, so each step can be measured on its own (eval/run_eval.py --opt key=value).
# The defaults are what the app runs. Environment variables IKAN_<KEY> override them (e.g. IKAN_ENGINE).
OPTIONS: dict[str, str] = {
    "engine": "meiki",    # meiki | paddle_v6 | ndl_lite (engines/)
    "filters": "1",       # fake-kanji filters (postprocess/filters.py); tuned on meikiocr's confidences
    "upscale": "0",       # re-read small text at 2x (preprocess/scale.py)
    "tiles": "0",         # also read big photos in 2x2 overlapping tiles (preprocess/scale.py)
    "orient": "0",        # 0 | always | check | retry: turn sideways / upside-down photos upright (preprocess/orient.py)
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
    segs = read(img)
    if turn == "check" or (turn == "retry" and scale.mean_conf(segs) < scale.WEAK_BELOW):
        # ask the orientation classifier (always for "check", after a weak first reading for "retry");
        # keep the turned reading only if the recognizer is also more confident in it
        turned = orient.upright(img)
        if turned is not img:
            segs2 = read(turned)
            if scale.mean_conf(segs2) > scale.mean_conf(segs):
                img, segs = turned, segs2
    if on("upscale"):
        segs = scale.with_upscale(read, img, segs)
    if on("tiles"):
        segs = scale.with_tiles(read, img, segs)
    return segs


def ocr_lines(img) -> list[str]:
    texts = []
    for seg in segments(img):
        if on("filters") and is_junk(seg):
            continue
        # drop an unsure kanji and break the line there, so its neighbours don't glue into a fake word
        text = "".join(c["char"] if not on("filters") or keep_char(c, seg) else " " for c in seg)
        texts += [fix_lookalikes(t) for t in text.split() if t]
    # the lower threshold yields overlapping boxes ("焼き魚定食" inside "焼き魚定食八百円"): keep the longer
    return [t for t in dict.fromkeys(texts) if not any(t != u and t in u for u in texts)]


SENTENCE_END = tuple("。！？!?」』）)")


def join_wrapped(lines: list[str]) -> list[str]:
    """Rejoin a sentence wrapped mid-word (最近の子供たちは夜更 / かしです。 -> ...夜更かしです。).
    A line that starts with hiragana continues the previous one: a new item or sentence almost never does."""
    out: list[str] = []
    for line in lines:
        if out and HIRA.match(line) and not out[-1].endswith(SENTENCE_END):
            out[-1] += line
        else:
            out.append(line)
    return out


def read_lines(img) -> list[str]:
    """BGR image -> the text lines the cards are made from."""
    return join_wrapped(ocr_lines(img))
