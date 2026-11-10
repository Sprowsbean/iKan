"""Photo -> lines of text: the OCR engine, then the clean-up steps. The cards are made from these lines."""
from engines import meiki
from jp import HIRA
from postprocess.filters import is_junk, keep_char
from postprocess.lookalike import fix_lookalikes


def ocr_lines(img) -> list[str]:
    texts = []
    for seg in meiki.read(img):
        if is_junk(seg):
            continue
        # drop an unsure kanji and break the line there, so its neighbours don't glue into a fake word
        text = "".join(c["char"] if keep_char(c, seg) else " " for c in seg)
        texts += [fix_lookalikes(t) for t in text.split() if t]
    # the lower threshold yields overlapping boxes ("焼き魚定食" inside "焼き魚定食八百円"): keep the longer
    return [t for t in dict.fromkeys(texts) if not any(t != u and t in u for u in texts)]


SENTENCE_END = tuple("。！？!?」』）)")


def join_wrapped(lines: list[str]) -> list[str]:
    """Rejoin a sentence wrapped mid-word (最近の子供たちは夜更 / かしです。 -> ...夜更かしです。).
    A line that starts with hiragana continues the previous one: a new item or sentence almost never does."""
    out: list[str] = []
    for line in lines:
        if out and HIRA.match(line) and not out[-1].endswith(SENTENCE_END):
            out[-1] += line
        else:
            out.append(line)
    return out


def read_lines(img) -> list[str]:
    """BGR image -> the text lines the cards are made from."""
    return join_wrapped(ocr_lines(img))
