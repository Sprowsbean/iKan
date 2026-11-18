"""jp.py: the Japanese character classes every other module relies on."""
from jp import HIRA, JP, KANJI, KATA, kata_to_hira


def test_kata_to_hira_converts_katakana_only():
    assert kata_to_hira("トウキョウ") == "とうきょう"
    assert kata_to_hira("東京タワー") == "東京たわー"  # kanji and the long-vowel mark ー stay


def test_character_classes():
    assert KANJI.match("漢") and KANJI.match("々")  # 々 repeats the previous kanji (人々)
    assert not KANJI.match("か") and not KANJI.match("A")
    assert HIRA.match("ひ") and not HIRA.match("ヒ")
    assert KATA.match("ヒ") and KATA.match("ー")
    assert JP.match("ひ") and JP.match("ヒ") and JP.match("火") and not JP.match("1")
