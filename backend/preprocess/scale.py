"""Make text the size the models expect (Phase 1 of docs/OCR_ROADMAP.md).

- upscale: if the text that was found is small, read the image again at 2x (plain bicubic, no AI
  super-resolution, which can invent strokes) and keep the more confident reading.
- tiles: a big photo is shrunk to the detector's input size, so its small text can vanish before it is read
  (meikiocr's detector takes 960x544). Read it in 2x2 overlapping tiles as well as whole.

The size limits come from the models' input sizes, not from our test images:
PP-OCR-style recognizers read lines at 48 px high, so text under half that is "small";
meikiocr's detector input is 960 px wide, so photos over twice that are shrunk more than 2x.
"""
import cv2

from jp import JP

SMALL_TEXT_PX = 24       # half the 48 px line height recognizers are trained on
TILE_ABOVE_PX = 1920     # 2x the meikiocr detector's 960 px input
TILE_OVERLAP = 0.2       # each tile reaches 20% into its neighbours, so a line cut by one edge is whole in another


def _jp_chars(segs):
    return [c for s in segs for c in s if JP.match(c.get("char", ""))]


def text_height(segs) -> float | None:
    """Median height of the Japanese characters read, in image pixels (None if too few to judge)."""
    hs = sorted(c["bbox"][3] - c["bbox"][1] for c in _jp_chars(segs) if c.get("bbox"))
    return hs[len(hs) // 2] if len(hs) >= 3 else None


# A reading whose mean confidence on Japanese characters is below this looks weak, and the gated retries
# (orientation, flattening) try again. Our choice, not a published rule: in eval/BASELINE.md a character
# read below 0.8 is wrong about a third of the time or more.
WEAK_BELOW = 0.8


def mean_conf(segs) -> float:
    cs = [c["conf"] for c in _jp_chars(segs)]
    return sum(cs) / len(cs) if cs else 0.0


def with_upscale(read, img, segs):
    """Re-read at 2x when the text found is small; keep whichever reading is more confident."""
    h = text_height(segs)
    if h is None or h >= SMALL_TEXT_PX:
        return segs
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    segs2 = read(big)
    return segs2 if mean_conf(segs2) > mean_conf(segs) else segs


def with_tiles(read, img, segs):
    """For a big photo, add what 2x2 overlapping tiles read. Duplicates (the same text read in the whole
    image and in a tile, or in two tiles) are removed later, where the pipeline keeps the longer of two
    overlapping texts."""
    H, W = img.shape[:2]
    if max(H, W) <= TILE_ABOVE_PX:
        return segs
    th, tw = int(H * (0.5 + TILE_OVERLAP / 2)), int(W * (0.5 + TILE_OVERLAP / 2))
    out = list(segs)
    for y in (0, H - th):
        for x in (0, W - tw):
            out += read(img[y:y + th, x:x + tw])
    return out
