"""Text lines -> kanji cards: SudachiPy splits words (with readings and dictionary forms), pykakasi gives romaji,
jamdict gives meanings from JMdict and per-kanji readings and levels from KANJIDIC2."""
import pykakasi
from jamdict import Jamdict
from sudachipy import dictionary, tokenizer as sudachi_tok

from jp import KANJI, kata_to_hira

kks = pykakasi.kakasi()
jam = Jamdict()
tok = dictionary.Dictionary().tokenizer()
MODE = sudachi_tok.Tokenizer.SplitMode.C
MODE_SHORT = sudachi_tok.Tokenizer.SplitMode.A  # splits compounds like 東京駅 -> 東京 + 駅


def romaji(kana: str) -> str:
    return "".join(x["hepburn"] for x in kks.convert(kana))


# KANJIDIC2 uses the old 4-level JLPT scale; map roughly to N-levels
OLD_TO_N = {"4": "N5-N4", "3": "N4-N3", "2": "N3-N2", "1": "N2-N1"}


def kanji_info(word: str):
    """Meaning, on/kun readings, level from KANJIDIC2 (fallback when JMdict has no entry)."""
    meanings, on, kun, levels = [], [], [], []
    for ch in word:
        for c in (x for x in jam.lookup(ch).chars if x.literal == ch):
            # skip numeric glosses like 京 = "10**16"
            meanings += [m for m in c.meanings(english_only=True) if "**" not in m][:2]
            for g in c.rm_groups:
                for r in g.readings:
                    if r.r_type == "ja_on":
                        on.append(r.value)
                    elif r.r_type == "ja_kun":
                        kun.append(r.value)
            if c.jlpt:
                levels.append(c.jlpt)
    # lowest old-number = hardest; pick hardest char in the word
    level = OLD_TO_N.get(min(levels), "?") if levels else "?"
    return meanings, on[:3], kun[:3], level


def char_cards(word: str, hira: str = "") -> list[dict]:
    """One entry per kanji in the word (KANJIDIC2): its own meaning and readings, for the card's breakdown.
    `used` marks the reading this word uses (東京 = とう+きょう, so 東 is トウ here, not ひがし): the readings
    are matched in order against the word's reading; when one doesn't fit (sound changes like がっこう),
    the rest are left unmarked."""
    info = {}
    for ch in dict.fromkeys(c for c in word if KANJI.search(c)):
        for c in (x for x in jam.lookup(ch).chars if x.literal == ch):
            info[ch] = {
                "char": ch,
                "meaning": ", ".join([m for m in c.meanings(english_only=True) if "**" not in m][:3]),
                # KANJIDIC marks okurigana with a dot (た.べる) and prefixes/suffixes with "-"
                "on": [{"kana": r.value, "romaji": romaji(r.value), "used": False}
                       for g in c.rm_groups for r in g.readings if r.r_type == "ja_on"][:4],
                "kun": [{"kana": r.value, "romaji": romaji(r.value.replace(".", "").replace("-", "")), "used": False}
                        for g in c.rm_groups for r in g.readings if r.r_type == "ja_kun"][:4],
                "level": OLD_TO_N.get(str(c.jlpt), "") if c.jlpt else "",
                "strokes": c.stroke_count,
            }
            break
    rest = hira
    for ch in word:
        if ch not in info:  # kana in the word reads as itself
            k = kata_to_hira(ch)
            if not rest.startswith(k):
                break
            rest = rest[len(k):]
            continue
        stems = [(kata_to_hira(r["kana"].split(".")[0].replace("-", "")), r) for r in info[ch]["on"] + info[ch]["kun"]]
        fit = max((x for x in stems if x[0] and rest.startswith(x[0])), key=lambda x: len(x[0]), default=None)
        if not fit:
            break
        if not any(r["used"] for r in info[ch]["on"] + info[ch]["kun"]):  # a kanji repeated in the word
            fit[1]["used"] = True
        rest = rest[len(fit[0]):]
    return list(info.values())


def glosses(word: str, hira: str = "", n_senses: int = 2) -> str:
    entries = jam.lookup(word).entries
    if not entries:
        return ""
    # 降る is both ふる (to fall) and くだる (to descend): prefer the entry matching the reading
    entry = next((e for e in entries if hira in (k.text for k in e.kana_forms)), entries[0])
    return "; ".join(g.text for s in entry.senses[:n_senses] for g in s.gloss[:2])


def reading(text: str) -> str:
    kana = "".join(m.reading_form() for m in tok.tokenize(text, MODE))
    # Sudachi echoes unknown words back unchanged (e.g. 做); pykakasi knows more single kanji
    if KANJI.search(kana):
        kana = "".join(x["kana"] for x in kks.convert(text))
    return kana


def make_card(surface: str, base: str, line: str) -> dict:
    # show the dictionary form (食べた -> 食べる) so the card is a word you can look up
    word = base if KANJI.search(base) else surface
    kana = reading(word)
    hira = kata_to_hira(kana)
    seg = {"hira": hira, "kana": kana, "hepburn": romaji(kana)}
    meanings, on, kun, level = kanji_info(word)
    meaning = glosses(word, hira) or glosses(surface)
    parts = [m.surface() for m in tok.tokenize(word, MODE_SHORT)]
    if not meaning and len(parts) > 1:
        # compound missing from JMdict (e.g. 東京駅): explain it from its sub-words
        meaning = " + ".join(f"{p} ({glosses(p, n_senses=1) or '?'})" for p in parts)
    elif not meaning:
        meaning = ", ".join(dict.fromkeys(meanings)) or "(no entry)"
    # context = where the word was actually found, not the meaning again
    if line.strip() != surface:
        context = f"Found in: 「{line.strip()}」"
    elif len(word) > 1:
        context = f"Found on its own. Characters: {' + '.join(word)}"
    else:
        context = "Found on its own (e.g. a sign or label)"
    if surface != word:
        context += f" (written as {surface})"
    return {
        "kanji": word,
        "romaji": seg["hepburn"],
        "readings": f"On'yomi: {', '.join(on) or '-'} | Kun'yomi: {', '.join(kun) or '-'} | "
                    f"Hiragana: {seg['hira']} | Katakana: {seg['kana']}",
        "meaning": meaning,
        "context": context,
        "level": level,
        "hiragana": seg["hira"],
        "chars": char_cards(word, hira),
    }


def cards_for(lines: list[str]) -> list[dict]:
    """One card per kanji word, in reading order."""
    cards, seen = [], set()
    for line in lines:  # per line, so a word never spans a line break and we know its line
        for m in tok.tokenize(line, MODE):
            w = m.surface()
            if KANJI.search(w):
                card = make_card(w, m.dictionary_form(), line)
                if card["kanji"] not in seen:  # 食べた and 食べる share one card
                    seen.add(card["kanji"])
                    cards.append(card)
    return cards
