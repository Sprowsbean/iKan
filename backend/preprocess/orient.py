"""Turn a sideways or upside-down photo upright before reading, with PaddleOCR's own whole-image orientation
classifier (PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx on Hugging Face, Apache-2.0, about 7 MB, a few ms on CPU).

meikiocr's engine handles this by re-reading the photo turned 90/270/180 degrees when the first reading is
weak (engines/meiki.py, up to 4 passes). PP-OCRv6 has no such step, and RapidOCR's classifier only flips
single lines 180 degrees, so a phone held sideways (rotation lock on) reads as junk. Preprocessing follows the
model's inference.yml: short side 256, centre crop 224, ImageNet mean/std, labels 0 / 90 / 180 / 270.
Loaded on first use only.
"""
import cv2
import numpy as np

REPO = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx"
# label -> how to turn the image back upright. The label is how far the content is turned clockwise
# (checked: a photo turned with cv2.ROTATE_90_CLOCKWISE gets label 1).
UNDO = {1: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_CLOCKWISE}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

_session = None


def angle(img) -> tuple[int, float]:
    """BGR image -> (label 0-3, probability)."""
    global _session
    if _session is None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        _session = ort.InferenceSession(hf_hub_download(REPO, "inference.onnx"), providers=["CPUExecutionProvider"])
    h, w = img.shape[:2]
    s = 256 / min(h, w)
    x = cv2.resize(img, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (x.shape[0] - 224) // 2, (x.shape[1] - 224) // 2
    x = (x[y0:y0 + 224, x0:x0 + 224].astype(np.float32) / 255.0 - MEAN) / STD
    p = _session.run(None, {"x": np.ascontiguousarray(x.transpose(2, 0, 1)[None])})[0][0]
    p = np.exp(p - p.max()) / np.exp(p - p.max()).sum() if p.min() < 0 or p.sum() > 1.01 else p
    k = int(p.argmax())
    return k, float(p[k])


def upright(img):
    k, _ = angle(img)
    return cv2.rotate(img, UNDO[k]) if k in UNDO else img
