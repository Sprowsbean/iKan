"""Shared test setup: run from backend/ with `venv\\Scripts\\python -m pytest` (see pytest.ini).

The unit tests need no OCR model. Tests marked `slow` load the real OCR models (downloaded once, then cached)
and read real images, like the app does; skip them with `-m "not slow"`.
"""
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(BACKEND))  # the app's modules (main, pipeline, cards...) import each other this way


def char(ch: str, conf: float = 0.99, alts: list[str] | None = None) -> dict:
    """One character as an OCR engine returns it; `alts` are its runner-up guesses (best first)."""
    c = {"char": ch, "conf": conf}
    if alts is not None:
        c["alts"] = [{"char": ch, "conf": conf}] + [{"char": a, "conf": 0.01} for a in alts]
    return c


def line(text: str, conf: float = 0.99) -> list[dict]:
    """A whole line of characters, all with the same confidence."""
    return [char(ch, conf) for ch in text]


@pytest.fixture
def data_dir() -> Path:
    return DATA
