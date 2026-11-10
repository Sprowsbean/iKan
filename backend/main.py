"""Free, local Kanji scanner: photo -> OCR (engines/) -> clean-up (postprocess/) -> kanji cards (cards.py).
No API keys, no tokens. Run: uvicorn main:app --reload --port 8000
"""
import os
from pathlib import Path

# keep the OCR model cache next to the code (not in the user's home on C:); Docker sets HF_HOME itself.
# Must be set before the engines import meikiocr, which downloads the models.
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / ".cache" / "huggingface"))

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from cards import cards_for
from pipeline import read_lines

app = FastAPI(title="Kanji OCR")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not a valid image")
    result = analyze_image(img)
    if not result["cards"]:
        raise HTTPException(404, "No Kanji found in the image")
    return result


def analyze_image(img) -> dict:
    """BGR image -> {"text": OCR text, "cards": [KanjiCard, ...]} (cards may be empty)."""
    lines = read_lines(img)
    return {"text": "\n".join(lines), "cards": cards_for(lines)}
