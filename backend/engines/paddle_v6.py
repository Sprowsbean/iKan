"""PaddleOCR PP-OCRv6 (Apache-2.0, June 2026) through RapidOCR: ONNX text detector + multilingual recognizer
(Japanese included), on CPU. Unlike meikiocr it was trained on real-world photos, scans and documents.
Used with RapidOCR's published defaults; models download on first use (about 25 MB for the small tier)."""
import os

import numpy as np

from engines.base import Segment

DET_MODEL = os.environ.get("PPV6_DET", "small")   # tiny / small / medium
REC_MODEL = os.environ.get("PPV6_REC", "small")   # small / medium (tiny has no Japanese)
THREADS = int(os.environ.get("OCR_THREADS", "-1"))  # -1 = onnxruntime's default (all cores)

_engine = None


def _get():
    global _engine
    if _engine is None:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
        _engine = RapidOCR(params={
            "Global.log_level": "warning",
            "Global.return_word_box": True, "Global.return_single_char_box": True,  # per-character conf + box
            "Det.lang_type": LangRec.JAPAN, "Det.ocr_version": OCRVersion.PPOCRV6,
            "Det.model_type": ModelType(DET_MODEL),
            "Rec.lang_type": LangRec.JAPAN, "Rec.ocr_version": OCRVersion.PPOCRV6,
            "Rec.model_type": ModelType(REC_MODEL),
            "EngineConfig.onnxruntime.intra_op_num_threads": THREADS,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1 if THREADS > 0 else -1,
        })
        _keep_alternates(_engine._load_rec_model())
    return _engine


TOP_K = 5
_alts: list[tuple[str, list[list[dict]]]] = []  # (line text, per character: top-k {"char", "conf"}), per call


def _keep_alternates(rec) -> None:
    """The recognizer scores every character at every position (CTC); RapidOCR keeps only the best. Keep the
    runners-up too, so later steps can choose among characters the model itself considered (Phase 3)."""
    decode = rec.postprocess_op
    chars, blank = decode.character, set(decode.get_ignored_tokens())

    def wrapped(preds, return_word_box=False, **kw):
        out = decode(preds, return_word_box, **kw)
        idx = preds.argmax(axis=2)
        for row, (text, _) in zip(range(len(idx)), out[0]):
            t = idx[row]
            sel = np.ones(len(t), bool)
            sel[1:] = t[1:] != t[:-1]
            for b in blank:
                sel &= t != b
            per_char = []
            for f in np.flatnonzero(sel):
                top = np.argsort(-preds[row, f])[:TOP_K + 1]
                per_char.append([{"char": chars[i], "conf": float(preds[row, f, i])} for i in top if i not in blank][:TOP_K])
            _alts.append((text, per_char))
        return out

    rec.postprocess_op = wrapped


def _rect(quad) -> list[float]:
    xs, ys = [p[0] for p in quad], [p[1] for p in quad]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def read(img) -> list[Segment]:
    """BGR image -> lines of {"char", "conf", "bbox"}. Horizontal lines top to bottom, then vertical columns
    right to left (Japanese reading order; RapidOCR itself sorts everything top-to-bottom, left-to-right)."""
    eng = _get()
    _alts.clear()
    # RapidOCR keeps use_* flags from the previous call (engines/ndl_lite.py calls it with use_rec=False),
    # so pass its defaults every time
    out = eng(img, use_det=True, use_cls=True, use_rec=True)
    if not out.txts:
        return []
    by_text: dict[str, list] = {}
    for text, per_char in _alts:
        by_text.setdefault(text, []).append(per_char)
    lines = []
    for txt, quad, words in zip(out.txts, out.boxes, out.word_results or [()] * len(out.txts)):
        x1, y1, x2, y2 = _rect(quad)
        cand = by_text.get(txt) or [[]]
        per_char = cand.pop(0) if len(cand) > 1 else cand[0]
        if len(per_char) != len(words):  # can't line the alternates up with the characters: leave them out
            per_char = [[]] * len(words)
        seg = [{"char": ch, "conf": float(conf), "bbox": _rect(q), "alts": alts}
               for (ch, conf, q), alts in zip(words, per_char) if ch.strip()]
        if seg:
            lines.append(((y2 - y1) > 1.5 * (x2 - x1), x1, x2, seg))
    horiz = [seg for vert, _, _, seg in lines if not vert]
    vert = [seg for _, _, x2, seg in sorted((l for l in lines if l[0]), key=lambda l: -l[2])]
    return horiz + vert
