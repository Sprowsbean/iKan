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

from cards import cards_for, glosses
from jp import KANJI
from pipeline import LAST_HINTS, on, read_detail
from postprocess import variants
from postprocess.candidates import LOW_CONF

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
    detail = read_detail(img)
    lines = [t for t, _ in detail]
    cards = cards_for(lines)
    if on("variants"):
        add_modern_forms(cards)
    if on("unsure"):
        mark_unsure(cards, detail)
    out = {"text": "\n".join(lines), "cards": cards}
    if LAST_HINTS:
        out["furigana"] = [{"reading": r, "line": b} for r, b in LAST_HINTS]
    return out


def add_modern_forms(cards: list[dict]) -> None:
    """A word printed in old forms (戰爭) also gets its modern form (戦争) and, if needed, that word's meaning."""
    for card in cards:
        new = variants.modern(card["kanji"])
        if new != card["kanji"]:
            card["modern"] = new
            if card["meaning"] in ("(no entry)", "") or card["meaning"].startswith(card["kanji"][0] + " ("):
                card["meaning"] = glosses(new, card["hiragana"]) or card["meaning"]


def mark_unsure(cards: list[dict], detail) -> None:
    """Say which kanji in a card's word the reader was unsure of, with the alternatives it considered:
    on real photos a character under LOW_CONF is right less than 14% of the time (eval/BASELINE.md)."""
    for card in cards:
        stem = "".join(ch for ch in card["kanji"] if KANJI.match(ch))  # 食べる is printed 食べた: find 食
        for text, chars in detail:
            word = card["kanji"] if card["kanji"] in text else stem
            i = text.find(word) if word else -1
            if i < 0:
                continue
            unsure = []
            for c in chars[i:i + len(word)]:
                if KANJI.match(c["char"]) and (c["conf"] < LOW_CONF or "was" in c):
                    alts = [a["char"] for a in c.get("alts", []) if a["char"] != c["char"] and KANJI.match(a["char"])]
                    if "was" in c:
                        alts = [c["was"]] + [a for a in alts if a != c["was"]]
                    unsure.append({"char": c["char"], "conf": round(c["conf"], 2), "alternatives": alts[:3]})
            if unsure:
                card["unsure"] = unsure
            break
