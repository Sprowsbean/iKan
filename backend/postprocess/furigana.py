"""Furigana by geometry (Phase 3 of docs/OCR_ROADMAP.md).

Published rule (research notes, post_correction.md section 6; 76-91% F1 in the papers found): ruby glyphs are
about half the size of the text they annotate and sit in a thin strip just above a horizontal line or just
to the right of a vertical column. A line that is kana only, much smaller than a neighbouring line that has
kanji, and right next to it, is a reading, not text: it is removed from the text and returned as a hint.

Needs character boxes in image coordinates (engines/paddle_v6.py has them; meikiocr's vertical columns are
read as a re-drawn strip, so their boxes are not image positions and those columns are never matched).
"""
from jp import HIRA, KANJI, KATA

SIZE_RATIO = 0.6   # "about half the size": ruby is typically 1/2 of the base text; allow for box noise
NEAR = 0.6         # the gap to the base line, in base-character sizes


def _box(seg):
    bs = [c["bbox"] for c in seg if c.get("bbox")]
    if not bs:
        return None
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def _size(seg, vertical: bool) -> float:
    """Median character size across the line: height for horizontal text, width for vertical."""
    s = sorted((c["bbox"][2] - c["bbox"][0]) if vertical else (c["bbox"][3] - c["bbox"][1])
               for c in seg if c.get("bbox"))
    return s[len(s) // 2] if s else 0.0


def _is_kana(seg) -> bool:
    text = "".join(c["char"] for c in seg)
    return bool(text) and all(HIRA.match(ch) or KATA.match(ch) for ch in text)


def split(segs: list[list[dict]]) -> tuple[list[list[dict]], list[tuple[str, str]]]:
    """-> (text lines without furigana, [(furigana, the base line's text), ...])."""
    info = []
    for s in segs:
        b = _box(s)
        vertical = bool(b) and (b[3] - b[1]) > 1.5 * (b[2] - b[0])
        info.append((b, vertical))
    ruby, hints = set(), []
    for i, (s, (b, v)) in enumerate(zip(segs, info)):
        if not b or not _is_kana(s) or len(s) > 12:
            continue
        size = _size(s, v)
        for j, (t, (B, V)) in enumerate(zip(segs, info)):
            if j == i or not B or V != v or not any(KANJI.match(c["char"]) for c in t):
                continue
            base = _size(t, V)
            if not base or size >= SIZE_RATIO * base:
                continue
            if v:   # just right of the column, overlapping it top to bottom
                beside = 0 <= b[0] - B[2] + 0.3 * base <= NEAR * base and min(b[3], B[3]) > max(b[1], B[1])
            else:   # just above the line, overlapping it left to right
                beside = 0 <= B[1] - b[3] + 0.3 * base <= NEAR * base and min(b[2], B[2]) > max(b[0], B[0])
            if beside:
                ruby.add(i)
                hints.append(("".join(c["char"] for c in s), "".join(c["char"] for c in t)))
                break
    return [s for i, s in enumerate(segs) if i not in ruby], hints
