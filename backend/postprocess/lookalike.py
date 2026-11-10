"""Kanji/katakana look-alikes the recognizer confuses (力/カ, 口/ロ, 一/ー ...), decided by the neighbours."""
from jp import KATA, LOOKALIKE, SKIP


def fix_lookalikes(line: str) -> str:
    """ゴウ、ガッ、力 -> ゴウ、ガッ、カ: a lookalike kanji whose neighbours are katakana is katakana."""
    chars = list(line)
    for i, ch in enumerate(chars):
        if ch.translate(LOOKALIKE) == ch:
            continue
        left = next((c for c in reversed(chars[:i]) if c not in SKIP), "")
        right = next((c for c in line[i + 1:] if c not in SKIP), "")
        neighbours = [c for c in (left, right) if c]
        # every neighbour must be katakana, so 八 in ラーメン八百円 stays a kanji
        if neighbours and all(KATA.match(c) or c.translate(LOOKALIKE) != c for c in neighbours) \
                and any(KATA.match(c) for c in neighbours):
            chars[i] = ch.translate(LOOKALIKE)
    return "".join(chars)
