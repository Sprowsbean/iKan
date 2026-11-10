"""NDLOCR-Lite's text-line recognizer (National Diet Library, 2026, CC BY 4.0; PARSeq, ONNX, built for CPU)
on the lines PP-OCRv6's detector finds (engines/paddle_v6.py). NDL trained it on Japanese books and magazines
(Meiji to today, old character forms) and, since v1.2, handwriting.

Only the recognizer is used: NDL's own layout detector carries extra DINOv3 licence terms, and its layout
pass is built for full book pages. Three model sizes for 30 / 50 / 100 characters per line, chosen the way
NDL's ocr.py does it (start small; a line that fills the model goes to the next size).
Experimental: the models (about 117 MB) are not bundled; set NDL_MODEL_DIR to a folder with the three
parseq-ndl-*.onnx files and config/NDLmoji.yaml from https://github.com/ndl-lab/ndlocr-lite (src/).
"""
import os
from pathlib import Path

import cv2
import numpy as np

from engines import paddle_v6
from engines.base import Segment

MODEL_DIR = Path(os.environ.get("NDL_MODEL_DIR", Path(__file__).resolve().parents[3] / "models" / "ndlocr-lite"))
TIERS = [(30, 25), (50, 45), (100, None)]  # (model's max characters, length that sends a line up a tier)
TOP_K = 5

_models = None


def _load():
    global _models
    if _models is None:
        import onnxruntime as ort
        import yaml
        chars = yaml.safe_load((MODEL_DIR / "NDLmoji.yaml").read_text(encoding="utf-8"))["model"]["charset_train"]
        opt = ort.SessionOptions()
        n = int(os.environ.get("OCR_THREADS", "-1"))
        if n > 0:
            opt.intra_op_num_threads, opt.inter_op_num_threads = n, 1
        sessions = {}
        for f in MODEL_DIR.glob("parseq-ndl-24x*-*-tiny-*.onnx"):
            size = int(f.name.split("-")[3])
            sessions[size] = ort.InferenceSession(str(f), opt, providers=["CPUExecutionProvider"])
        _models = (list(chars), sessions)
    return _models


def _read_line(crop) -> list[dict]:
    """One line crop -> characters with confidence (softmax at each output step) and top-k alternates."""
    chars, sessions = _load()
    h, w = crop.shape[:2]
    if h > w * 0.8:  # NDL reads vertical lines turned a quarter left (parseq.py)
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    result: list[dict] = []
    for size, up in TIERS:
        s = sessions[size]
        inp = s.get_inputs()[0]
        ih, iw = inp.shape[2:]
        x = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
        logits = s.run(None, {inp.name: np.ascontiguousarray(x.transpose(2, 0, 1)[None])})[0][0]
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        result = []
        for step in p:
            best = int(step.argmax())
            if best == 0:  # end of text
                break
            top = [int(i) for i in np.argsort(-step)[:TOP_K + 1] if i != 0][:TOP_K]
            result.append({"char": chars[best - 1], "conf": float(step[best]),
                           "alts": [{"char": chars[i - 1], "conf": float(step[i])} for i in top]})
        if up is None or len(result) < up:
            break
    return result


def read(img) -> list[Segment]:
    from rapidocr.utils.process_img import get_rotate_crop_image
    det = paddle_v6._get()(img, use_cls=False, use_rec=False)
    if det.boxes is None or not len(det.boxes):
        return []
    lines = []
    for quad in det.boxes:
        crop = get_rotate_crop_image(img, np.array(quad, dtype=np.float32))
        x1, y1 = float(min(p[0] for p in quad)), float(min(p[1] for p in quad))
        x2, y2 = float(max(p[0] for p in quad)), float(max(p[1] for p in quad))
        seg = [c for c in _read_line(crop) if c["char"].strip()]
        if seg:
            lines.append(((y2 - y1) > 1.5 * (x2 - x1), x2, seg))
    horiz = [s for v, _, s in lines if not v]
    vert = [s for _, _, s in sorted((l for l in lines if l[0]), key=lambda l: -l[1])]
    return horiz + vert
