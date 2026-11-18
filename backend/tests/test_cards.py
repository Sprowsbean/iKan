"""cards.py: text lines -> kanji flashcards (readings, meanings, levels)."""
from cards import cards_for, char_cards, glosses, kanji_info, make_card, reading, romaji


def test_romaji_and_reading():
    assert romaji("とうきょう") == "toukyou"
    assert reading("東京") == "トウキョウ"


def test_glosses_comes_from_jmdict():
    assert glosses("先生").startswith("teacher")
    assert glosses("存在しない単語") == ""  # not a word: no meaning rather than a guess


def test_glosses_prefers_the_entry_with_the_matching_reading():
    # 降る is both ふる (to fall, of rain) and くだる (to descend)
    assert glosses("降る", "くだる").startswith("to descend")


def test_kanji_info_gives_meaning_readings_and_level():
    meanings, on, kun, level = kanji_info("日")
    assert "sun" in meanings and "ニチ" in on and "ひ" in kun
    assert level == "N5-N4"


def test_char_cards_marks_the_reading_the_word_uses():
    used = {c["char"]: [r["kana"] for r in c["on"] + c["kun"] if r["used"]] for c in char_cards("東京", "とうきょう")}
    assert used == {"東": ["トウ"], "京": ["キョウ"]}  # 東 is トウ here, not ひがし


def test_make_card_uses_the_dictionary_form():
    card = make_card("食べた", "食べる", "私は食べた")
    assert card["kanji"] == "食べる"
    assert card["hiragana"] == "たべる" and card["romaji"] == "taberu"
    assert card["meaning"].startswith("to eat")
    assert "written as 食べた" in card["context"]


def test_cards_for_makes_one_card_per_word():
    kanji = [c["kanji"] for c in cards_for(["食べた。食べる", "東京駅"])]
    assert kanji == ["食べる", "東京駅"]  # 食べた and 食べる share one card


def test_cards_for_ignores_lines_without_kanji():
    assert cards_for(["ひらがなだけ", "Hello"]) == []
