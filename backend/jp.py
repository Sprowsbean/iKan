"""Japanese character classes shared by the OCR engines, the clean-up steps and the cards."""
import re

KANJI = re.compile(r"[一-鿿々]")
KATA = re.compile(r"[ァ-ヶー]")
HIRA = re.compile(r"[ぁ-ゖ]")
JP = re.compile(r"[ぁ-ゖァ-ヶー一-鿿々]")
# kanji the OCR confuses with katakana: 力/カ, 口/ロ, 工/エ, 二/ニ, 夕/タ, 卜/ト, 八/ハ, 一/ー
LOOKALIKE = str.maketrans("力口工二夕卜八一", "カロエニタトハー")
SKIP = set("、・，, 　()（）「」")


def kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)
