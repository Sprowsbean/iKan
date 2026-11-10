"""Old / traditional character forms -> the modern Japanese form (Phase 3 of docs/OCR_ROADMAP.md): 戰 -> 戦.

From KANJIDIC2 (already used for the cards; EDRDG licence): each kanji lists its JIS X 0208 variants. A kanji
with no school grade whose JIS variant has one is an old or alternative form of that variant (戰 -> 戦,
兒 -> 児, 學 -> 学). The learner sees the modern form's word and meaning, and what was printed.
Never the Chinese simplified/traditional data: Japanese modern forms differ from Chinese simplified ones.
"""
import os
import sqlite3
from functools import lru_cache


def _kuten(code: str) -> str | None:
    """'1-32-79' (plane-row-cell) -> the character, via EUC-JP."""
    try:
        _, row, cell = (int(x) for x in code.split("-"))
        return bytes([row + 0xA0, cell + 0xA0]).decode("euc_jp")
    except (ValueError, UnicodeDecodeError):
        return None


@lru_cache(maxsize=1)
def _table() -> dict[str, str]:
    import jamdict_data
    db = os.path.join(os.path.dirname(jamdict_data.__file__), "jamdict.db")
    with sqlite3.connect(db) as c:
        grade = {lit: int(g) for lit, g in c.execute("SELECT literal, grade FROM character") if g}
        out = {}
        for lit, code in c.execute("SELECT ch.literal, v.value FROM variant v JOIN character ch ON ch.ID = v.cid "
                                   "WHERE v.var_type = 'jis208'"):
            v = _kuten(code)
            # KANJIDIC2 grades: 1-6 school, 8 other jōyō, 9 name kanji, 10 name kanji that are variants of
            # jōyō kanji (戰 國 櫻). So: an ungraded or grade-10 kanji whose variant is jōyō.
            if v and v != lit and grade.get(lit) in (None, 10) and (grade.get(v) or 99) <= 8:
                out.setdefault(lit, v)
    return out


def modern(word: str) -> str:
    """The word with old character forms replaced by modern ones (unchanged if it has none)."""
    t = _table()
    return "".join(t.get(ch, ch) for ch in word)
