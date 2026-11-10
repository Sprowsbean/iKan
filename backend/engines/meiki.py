"""meikiocr (rtr46/meikiocr, ONNX, CPU): text-line detector + recognizer, built for video-game text.
Around it: padded re-reads of each box, vertical columns cut into characters and read as a horizontal line
(the vertical recognizer was weak), and retries at 90/270/180 degrees for photos taken sideways."""
import cv2
import numpy as np
from meikiocr import MeikiOCR

from engines.base import Segment
from jp import JP, KANJI, KATA, LOOKALIKE, SKIP

ocr = MeikiOCR()          # downloads ONNX models from HuggingFace on first run, then cached

DET_THRESHOLD = 0.15  # the default 0.5 misses big standalone kanji (a table or poster of single characters, 一)
BOX_PAD = 0.15       # detector boxes often clip tall kanji (対 read as 文リ); also try a padded box
# mean recognizer confidence on Japanese chars: upright text measured 0.78-0.97, sideways/upside-down <= 0.51
UPRIGHT_CONF = 0.7   # at or above: the image is upright, skip trying rotations
ROTATED_MIN_CONF = 0.6  # a rotation must reach this to count as real text (not junk from an English image)


def _conf(line: dict) -> float:
    cs = [c["conf"] for c in line["chars"]]
    return sum(cs) / len(cs) if cs else 0.0


def _ocr_boxes(img, boxes):
    # meikiocr has no "recognize these boxes" call, so feed run_ocr our own detections
    ocr.run_detection = lambda image, conf_threshold=0.5: [dict(b) for b in boxes]
    try:
        return ocr.run_ocr(img)
    finally:
        del ocr.run_detection


def _pick(a: dict, b: dict) -> dict:
    """Tight vs padded reading of one box: the more confident one, but a lone 力 beats a lone カ."""
    ta, tb = a["text"].strip(), b["text"].strip()
    if len(ta) == len(tb) == 1 and ta != tb and ta.translate(LOOKALIKE) == tb.translate(LOOKALIKE):
        return a if KANJI.match(ta) else b
    pairs = list(zip(a["chars"], b["chars"]))
    if len(a["chars"]) == len(b["chars"]) >= 3 and ta != tb \
            and sum(ca["char"] == cb["char"] for ca, cb in pairs) * 2 >= len(pairs):
        # Real text read twice mostly agrees: take each differing character from the more confident
        # reading. A heavy bold 更 read as 史 (0.18) tight and correctly (0.50) padded, while 夜 is only
        # right tight: 夜史かし + 変更かし -> 夜更かし. Junk (pencil marks, patterns) reads differently
        # each time; mixing it would only make the junk look confident, so it keeps the rule below.
        chars = [ca if ca["conf"] >= cb["conf"] else cb for ca, cb in pairs]
        return {**a, "chars": chars, "text": "".join(c["char"] for c in chars)}
    return a if _conf(a) >= _conf(b) else b


def _split_gaps(line: dict) -> list[list[dict]]:
    """The detector joins items in one row (日 ... 天 -> 日天); split where the gap exceeds a character."""
    chars = [c for c in line["chars"] if c["char"].strip()]
    if not chars:
        return []
    if line["is_vertical"]:  # vertical char boxes are not reliable enough to split on
        return [chars]
    segs = [[chars[0]]]
    for a, b in zip(chars, chars[1:]):
        size = max(a["bbox"][3] - a["bbox"][1], b["bbox"][3] - b["bbox"][1])
        # recognizer char boxes are stretched, so a real gap measures well under a character;
        # punctuation (、) is narrow and leaves a gap that is not a separation
        if a["char"] not in SKIP and b["char"] not in SKIP and b["bbox"][0] - a["bbox"][2] > size * 0.4:
            segs.append([])
        segs[-1].append(b)
    return segs


def _is_slice(b: dict, boxes: list[dict]) -> bool:
    """A low-threshold box covering only part of a character (top of 議 read as 羊) inside a taller box."""
    x1, y1, x2, y2 = b["bbox"]
    area = max(1, (x2 - x1) * (y2 - y1))
    for o in boxes:
        X1, Y1, X2, Y2 = o["bbox"]
        inter = max(0, min(x2, X2) - max(x1, X1)) * max(0, min(y2, Y2) - max(y1, Y1))
        if o is not b and inter / area > 0.8 and (y2 - y1) < 0.75 * (Y2 - Y1):
            return True
    return False


def _grid_cells(col, em: int) -> list[tuple[int, int]]:
    """Cut a vertical column into character cells. In tategaki every character (small kana too)
    takes one em-square slot, so find the slot pitch from the darkness profile, cut at its lightest rows."""
    g = cv2.GaussianBlur(cv2.cvtColor(col, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    dark = np.clip(np.percentile(g, 90) - g - 25, 0, None)  # -25 ignores paper texture / camera noise
    prof = dark.sum(axis=1)
    if prof.max() <= 0:
        return []
    on = np.flatnonzero(prof > 0.08 * prof.max())
    top, end = int(on[0]), int(on[-1]) + 1
    x = prof[top:end] - prof[top:end].mean()
    lo, hi = int(0.8 * em), int(1.7 * em)
    if lo > 0 and len(x) > hi + 2:  # pitch = strongest repeat distance (autocorrelation)
        ac = [float(np.dot(x[:-k], x[k:])) / (len(x) - k) for k in range(lo, hi)]
        pitch = lo + int(np.argmax(ac))
    else:
        pitch = int(1.1 * em)
    cells, y = [], max(0, top - int(0.1 * pitch))
    while y < end - 0.3 * pitch:
        a, b = int(y + 0.75 * pitch), min(len(prof), int(y + 1.25 * pitch) + 1)
        cut = a + int(np.argmin(prof[a:b])) if b > a else len(prof)
        cells.append((y, cut))
        y = cut
    if y < end:
        cells.append((y, min(len(prof), end + int(0.1 * pitch))))
    ink = [prof[a:b].sum() for a, b in cells]
    return [c for c, i in zip(cells, ink) if i > 0.05 * np.median(ink)]


def _read_vertical(img, box) -> dict:
    """meikiocr's vertical recognizer drops and garbles characters; its horizontal one is strong.
    So cut the column into characters, lay them side by side, and read that as a horizontal line."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
    p = (x2 - x1) // 6
    col = img[max(0, y1 - p):min(h, y2 + p), max(0, x1 - p):min(w, x2 + p)]
    # a box at the very edge of the image can crop to nothing; that used to crash the request (HTTP 500)
    cells = _grid_cells(col, x2 - x1) if col.size and x2 > x1 else []
    if not cells:
        return {"text": "", "chars": [], "is_vertical": True}
    cw = col.shape[1]
    side = max(cw, max(b - a for a, b in cells))
    squares = []
    for a, b in cells:  # each character centred in a white square so small kana keep their size
        sq = np.full((side, side, 3), 255, np.uint8)
        oy, ox = (side - (b - a)) // 2, (side - cw) // 2
        sq[oy:oy + b - a, ox:ox + cw] = col[a:b]
        squares.append(sq)
    line = cv2.hconcat(squares)
    line = cv2.resize(line, (max(1, line.shape[1] * 64 // side), 64))
    chars = [dict(c) for c in ocr.run_recognition([line])[0]["chars"]]
    for prev, c in zip(chars, chars[1:]):
        # a vertical ー laid flat reads as a bar, I or 「 (ニャ「 -> ニャー, ジュIス -> ジュース)
        if c["char"] in "|｜丨「Il1" and KATA.match(prev["char"]):
            c["char"] = "ー"
    return {"text": "".join(c["char"] for c in chars), "chars": chars, "is_vertical": True}


def _jp_conf(lines: list[dict]) -> float:
    """Mean recognizer confidence on the Japanese chars of a pass. Sideways text reads as low-confidence junk."""
    cs = [c["conf"] for l in lines for c in l["chars"] if JP.match(c["char"])]
    return sum(cs) / len(cs) if len(cs) >= 3 else 0.0  # 1-2 chars: too little to judge (junk like 万工)


def _ocr_pass(img) -> tuple[list[list[dict]], float]:
    """One OCR pass: lines as lists of {"char", "conf", ...}, plus the mean confidence on Japanese."""
    h, w = img.shape[:2]
    boxes = ocr.run_detection(img, DET_THRESHOLD)
    boxes = [b for b in boxes if not _is_slice(b, boxes)]
    # tall boxes are vertical columns (tategaki), read right to left
    is_col = lambda b: b["bbox"][3] - b["bbox"][1] > 1.5 * (b["bbox"][2] - b["bbox"][0])
    cols = sorted((b for b in boxes if is_col(b)), key=lambda b: -b["bbox"][2])
    vlines = [r for r in (_read_vertical(img, b) for b in cols) if r["chars"]]
    boxes = [b for b in boxes if not is_col(b)]
    padded = []
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        p = int(round((y2 - y1) * BOX_PAD))  # pad both ways so a square kanji box stays horizontal
        padded.append({**b, "bbox": [max(0, x1 - p), max(0, y1 - p), min(w, x2 + p), min(h, y2 + p)]})
    # read every box tight and padded, keep whichever the recognizer is more confident about
    best = [_pick(a, b) for a, b in zip(_ocr_boxes(img, boxes), _ocr_boxes(img, padded))] if boxes else []
    segs = [s for l in best for s in _split_gaps(l)] + [l["chars"] for l in vlines]
    return segs, _jp_conf(best + vlines)


def _oriented_pass(img) -> list[list[dict]]:
    """OCR, also trying the image turned 90/270/180 degrees: a phone held sideways (rotation lock on)
    sends the text sideways. Keep the orientation the recognizer is most confident about."""
    segs, conf = _ocr_pass(img)
    if conf >= UPRIGHT_CONF:  # upright already (every normal photo): no extra passes
        return segs
    best, best_conf = segs, max(conf, ROTATED_MIN_CONF)
    for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
        s, c = _ocr_pass(cv2.rotate(img, rot))
        if c > best_conf:
            best, best_conf = s, c
            if c >= UPRIGHT_CONF:
                break
    return best


def read(img) -> list[Segment]:
    """The engine's entry point (see engines/base.py)."""
    return _oriented_pass(img)
