"""Fake-kanji filters: lines and characters read from textures, patterns, icons or English rather than text."""
from collections import Counter

from jp import JP, KANJI, SKIP

# fake kanji (read from textures, patterns, English): measured on 108 images, these rules cut them from
# 140 to 16 while keeping all 602 real ones (a real 無 can score 0.15 inside an otherwise confident line)
KANJI_MIN_CONF = 0.4     # a kanji below this is dropped...
KANJI_CTX_CONF = 0.8     # ...unless the rest of its line averages at least this
JUNK_LINE_CONF = 0.5     # a line averaging below this is junk
LONE_STROKE_CONF = 0.7   # a lone 一 needs this: pencil underlines/arrows read as 一 at 0.44-0.57, a real one 0.88
LATIN_JUNK_CONF = 0.9    # a single kanji among Latin/symbols (Q*A力) needs this to count as text


def is_junk(seg: list[dict]) -> bool:
    """Lines read from a texture or pattern rather than text."""
    text = "".join(c["char"] for c in seg)
    n = len(text)
    if text == "一" and seg[0]["conf"] < LONE_STROKE_CONF:  # an underline / arrow / pencil stroke
        return True
    jp = [c for c in seg if JP.match(c["char"])]
    others = [c for c in seg if not JP.match(c["char"]) and c["char"] not in SKIP]
    # one unsure kanji amid Latin letters/symbols: an icon or logo read as text (Q*A力, Q會); real mixed
    # text such as JLPT先生 has more Japanese, and a clearly printed Tokyo駅 is read confidently
    if len(jp) == 1 and others and KANJI.match(jp[0]["char"]) and jp[0]["conf"] < LATIN_JUNK_CONF:
        return True
    if n >= 5 and len(set(text)) / n <= 0.35:  # repeating pattern: 通の通の通の... (seigaiha waves)
        return True
    if n >= 6 and Counter(text).most_common(1)[0][1] / n >= 0.4:  # one char dominates: の浜の混の混の...
        return True
    return n >= 2 and sum(c["conf"] for c in seg) / n < JUNK_LINE_CONF  # unsure about the whole line


def keep_char(c: dict, seg: list[dict]) -> bool:
    """Keep a character unless it is an unsure kanji in an otherwise unsure line."""
    if not KANJI.match(c["char"]) or c["conf"] >= KANJI_MIN_CONF:
        return True
    rest = [o["conf"] for o in seg if o is not c]
    return bool(rest) and sum(rest) / len(rest) >= KANJI_CTX_CONF
