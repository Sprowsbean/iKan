"""postprocess/: the clean-up steps between the OCR engine and the cards."""
from conftest import char, line
from postprocess import charset
from postprocess.candidates import correct, pieces
from postprocess.filters import is_junk, keep_char
from postprocess.lookalike import fix_lookalikes
from postprocess.variants import modern


# ---------- look-alikes: 力 (kanji "power") vs カ (katakana "ka") ----------

def test_lookalike_becomes_katakana_between_katakana():
    assert fix_lookalikes("ゴウ、ガッ、力") == "ゴウ、ガッ、カ"


def test_lookalike_stays_kanji_next_to_kanji():
    assert fix_lookalikes("ラーメン八百円") == "ラーメン八百円"  # 八 is followed by 百: a price, not ハ
    assert fix_lookalikes("人力") == "人力"


def test_lone_lookalike_is_left_alone():
    assert fix_lookalikes("力") == "力"  # no neighbours: nothing says it is katakana


# ---------- fake-kanji filters ----------

def test_unsure_lone_stroke_is_junk_but_a_clear_one_is_kept():
    assert is_junk([char("一", 0.5)])       # a pencil underline read as 一
    assert not is_junk([char("一", 0.95)])  # a printed 一 (ichi)


def test_repeating_pattern_is_junk():
    assert is_junk(line("通の通の通の通の"))  # a wave pattern read as text


def test_unsure_line_is_junk_and_a_clear_line_is_not():
    assert is_junk(line("東京駅", conf=0.3))
    assert not is_junk(line("東京駅", conf=0.95))


def test_keep_char_drops_an_unsure_kanji_only_in_an_unsure_line():
    good = [char("東"), char("京", 0.2), char("駅")]
    assert keep_char(good[1], good)  # the rest of the line is confident: keep it
    bad = [char("東", 0.5), char("京", 0.2), char("駅", 0.5)]
    assert not keep_char(bad[1], bad)


# ---------- Japanese-only kanji (PP-OCRv6 is multilingual) ----------

def test_japanese_accepts_japanese_and_old_forms():
    for ch in "込車場戦戰稲稻々あアA1":
        assert charset.japanese(ch), ch


def test_japanese_rejects_chinese_simplified_forms():
    for ch in "这车关场说门":
        assert not charset.japanese(ch), ch


def test_clean_swaps_a_chinese_form_for_the_readers_japanese_runner_up():
    out = charset.clean([char("这", 0.9, alts=["込"])])
    assert out[0]["char"] == "込"
    assert out[0]["zh"] == "这"      # the original is kept, but not as "was" (no card alternative)
    assert out[0]["conf"] == 0.9     # the recognizer's own confidence stays


def test_clean_drops_a_chinese_form_with_no_japanese_runner_up():
    out = charset.clean([char("东"), char("车", 0.8, alts=["东"]), char("あ")])
    assert [c["char"] for c in out] == [" ", " ", "あ"]  # a space breaks the line there


def test_clean_leaves_japanese_text_unchanged():
    seg = line("東京駅")
    assert charset.clean(seg) == seg


# ---------- dictionary-checked correction (先土 -> 先生) ----------

def test_pieces_counts_dictionary_words():
    assert pieces("先生") == 1  # one JMdict word
    assert pieces("先土") == 2  # two lone kanji


def test_correct_fixes_an_unsure_kanji_when_the_dictionary_agrees():
    seg = [char("先"), char("土", 0.3, alts=["生", "士"])]
    out = correct(seg)
    assert "".join(c["char"] for c in out) == "先生"
    assert out[1]["was"] == "土"  # the card can say "printed as 土?"


def test_correct_never_touches_a_confident_kanji():
    seg = [char("先"), char("土", 0.95, alts=["生"])]
    assert "".join(c["char"] for c in correct(seg)) == "先土"


def test_correct_never_offers_a_chinese_form():
    seg = [char("出"), char("口", 0.3, alts=["门"])]  # 门 is the Chinese simplified form of 門
    assert "".join(c["char"] for c in correct(seg)) == "出口"


def test_correct_does_not_change_the_input():
    seg = [char("先"), char("土", 0.3, alts=["生"])]
    correct(seg)
    assert seg[1]["char"] == "土"


# ---------- old character forms ----------

def test_modern_replaces_old_forms():
    assert modern("戰爭") == "戦争"
    assert modern("國") == "国"


def test_modern_leaves_modern_words_alone():
    assert modern("東京") == "東京"
