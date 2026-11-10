"""What every OCR engine returns, so engines can be swapped or combined and measured one at a time."""
from typing import Protocol, TypedDict

import numpy as np


class Char(TypedDict, total=False):
    char: str
    conf: float         # recognizer confidence, 0..1
    bbox: list[float]   # x1, y1, x2, y2 in the image the engine read
    alts: list[dict]    # the recognizer's top-k candidates here, best first: [{"char", "conf"}, ...] (if known)


# one line of text as read, left to right (a vertical column is laid out the same way)
Segment = list[Char]


class Engine(Protocol):
    def read(self, img: np.ndarray) -> list[Segment]:
        """BGR image -> lines of characters with confidences, before any clean-up."""
        ...
