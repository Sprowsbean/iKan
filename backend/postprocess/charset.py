"""Keep the reading Japanese: PP-OCRv6's recognizer is multilingual and sometimes outputs Chinese forms
(这 for 込, 车 for 車, 场 for 場).

A kanji counts as Japanese if it is in JIS X 0208 (the standard Japanese character set, 6,355 kanji) or is
written in a JMdict word. Both lists come from jamdict's own database (KANJIDIC2 and JMdict), which the cards
already use. Old forms (戰, 稻) and 々 pass; Chinese simplified forms (这, 车, 关, 场, 说) don't.

A non-Japanese kanji is replaced by the recognizer's own best Japanese runner-up at that spot (the original
is kept in "zh"). If it proposed none, the character is dropped and the line breaks there.
"""
import os
import sqlite3
from functools import lru_cache

from jp import KANJI


@lru_cache(maxsize=1)
def _japanese() -> frozenset[str]:
    import jamdict_data
    db = os.path.join(os.path.dirname(jamdict_data.__file__), "jamdict.db")
    with sqlite3.connect(db) as c:
        jis = {lit for (lit,) in c.execute("SELECT ch.literal FROM codepoint cp JOIN character ch ON ch.ID = cp.cid "
                                           "WHERE cp.cp_type = 'jis208'")}
        used = {ch for (t,) in c.execute("SELECT text FROM Kanji") for ch in t}
    return frozenset(ch for ch in jis | used if KANJI.match(ch))


def japanese(ch: str) -> bool:
    """True for kana, punctuation and Japanese kanji; False only for a kanji outside the Japanese sets."""
    return not KANJI.fullmatch(ch) or ch in _japanese()


def clean(seg: list[dict]) -> list[dict]:
    """The line with Chinese-only kanji swapped for a Japanese runner-up, or blanked (" ") if there is none."""
    out = []
    for c in seg:
        if japanese(c["char"]):
            out.append(c)
            continue
        alt = next((a for a in c.get("alts", []) if a["char"] != c["char"] and KANJI.fullmatch(a["char"])
                    and japanese(a["char"])), None)
        if alt:
            # confidence stays the recognizer's own (as in candidates.correct), so the later filters treat the
            # spot as before; "zh" (not "was") so the card doesn't offer the Chinese form as an alternative
            out.append({**c, "char": alt["char"], "zh": c["char"]})
        else:
            out.append({**c, "char": " "})
    return out
