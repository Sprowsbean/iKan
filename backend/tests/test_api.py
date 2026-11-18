"""End to end: POST /analyze with real images, through the real OCR models, as the website does."""
import pytest
from fastapi.testclient import TestClient

from main import app

pytestmark = pytest.mark.slow
client = TestClient(app)


def post(path) -> "object":
    with open(path, "rb") as f:
        return client.post("/analyze", files={"file": (path.name, f, "image/png")})


def test_sign_gives_text_and_cards(data_dir):
    r = post(data_dir / "sign_gothic_clean.png")  # a generated sign: 東京駅 / 出口はこちら
    assert r.status_code == 200
    body = r.json()
    assert body["text"].splitlines() == ["東京駅", "出口はこちら"]
    cards = {c["kanji"]: c for c in body["cards"]}
    assert set(cards) == {"東京駅", "出口"}
    assert cards["出口"]["hiragana"] == "でぐち" and cards["出口"]["meaning"].startswith("exit")
    for card in body["cards"]:  # every card has what the flashcard screen shows
        assert {"kanji", "romaji", "readings", "meaning", "context", "level", "hiragana", "chars"} <= set(card)


def test_blank_image_is_404(data_dir):
    r = post(data_dir / "blank.png")
    assert r.status_code == 404
    assert r.json()["detail"] == "No Kanji found in the image"


def test_not_an_image_is_400():
    r = client.post("/analyze", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
