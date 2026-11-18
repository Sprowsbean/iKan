"""pipeline.py and main.py: joining lines and the extra notes on the cards (no OCR model needed)."""
from conftest import char
from main import add_modern_forms, mark_unsure
from pipeline import OPTIONS, join_wrapped, on


def test_defaults_are_the_chosen_setup():
    assert OPTIONS["engine"] == "paddle_v6"
    assert on("correct") and on("charset") and on("variants") and on("unsure")
    assert OPTIONS["orient"] == "check"
    assert not on("tiles") and not on("flatten")


def test_join_wrapped_rejoins_a_sentence_broken_mid_word():
    lines = [("最近の子供たちは夜更", []), ("かしです。", []), ("次の話", [])]
    assert [t for t, _ in join_wrapped(lines)] == ["最近の子供たちは夜更かしです。", "次の話"]


def test_join_wrapped_keeps_a_finished_sentence_apart():
    lines = [("終わりです。", []), ("はい", [])]
    assert [t for t, _ in join_wrapped(lines)] == ["終わりです。", "はい"]


def test_mark_unsure_lists_japanese_alternatives_only():
    cards = [{"kanji": "古"}]
    detail = [("古", [char("古", 0.44, alts=["吉", "场"])])]  # 场 is a Chinese form
    mark_unsure(cards, detail)
    assert cards[0]["unsure"] == [{"char": "古", "conf": 0.44, "alternatives": ["吉"]}]


def test_mark_unsure_leaves_confident_cards_alone():
    cards = [{"kanji": "東京"}]
    mark_unsure(cards, [("東京", [char("東"), char("京")])])
    assert "unsure" not in cards[0]


def test_add_modern_forms():
    cards = [{"kanji": "戰爭", "meaning": "(no entry)", "hiragana": "せんそう"}]
    add_modern_forms(cards)
    assert cards[0]["modern"] == "戦争"
    assert cards[0]["meaning"].startswith("war")
