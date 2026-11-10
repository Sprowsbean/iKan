"""Flatten a curved page or label before reading (Phase 4 of docs/OCR_ROADMAP.md), with UVDoc
(PaddlePaddle/UVDoc_onnx on Hugging Face, Apache-2.0, about 32 MB, about 0.5 s on a laptop CPU).

It predicts how the surface bends and resamples the photo flat; the output has the input's size. Measured
on branch experiment/curved: real curved photos 67 -> 85 of 109 words, but flat test images 216 -> 203, so
it is only worth it when the first reading looks weak (preprocess/scale.py WEAK_BELOW; no published rule says
when to flatten). Loaded on first use only, so scans that don't need it don't pay for it.
"""
import cv2
import numpy as np

_session = None


def flatten(img):
    global _session
    if _session is None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("PaddlePaddle/UVDoc_onnx", "inference.onnx")
        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    y = _session.run(None, {"image": x})[0][0].transpose(1, 2, 0)
    return cv2.cvtColor((y * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
