"""Make the two fixed JaWildText test sets used by run_eval.py, from one downloaded shard of each dataset.

JaWildText (llm-jp/jawildtext, Apache-2.0) is a 2026 benchmark of real photos taken in Japan, so it is a test
set we didn't make ourselves. Download (about 365 MB, into a folder outside the repo):

    python -c "from huggingface_hub import hf_hub_download as d; r='../../test_images/jawildtext/raw';
      d('llm-jp/jawildtext', 'board_vqa/train-00005-of-00006.parquet', repo_type='dataset', local_dir=r);
      d('nagohachi/jawildtext_cropped', 'data/shard-00000.tar', repo_type='dataset', local_dir=r)"

Then, from backend/:

    venv\\Scripts\\python eval/prepare_jawildtext.py [path/to/test_images/jawildtext]

Writes, next to raw/:
  photos/*.jpg + photos.json   whole signboard/poster/product photos, shrunk like the app sends them;
                               {file: [text of each annotated region]}
  lines/*.jpg  + lines.json    single text-line crops (from the cropped copy); {file: text}
The subsets are drawn with a fixed seed, so every run evaluates the same images.
"""
import io
import json
import random
import re
import sys
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

SEED = 0
N_PHOTOS = 100
N_LINES = 500
MAX_SIDE = 2400  # the app scales crops down to this before upload (src/components/CropView.tsx)
JP = re.compile(r"[ぁ-ゖァ-ヶ一-鿿々]")

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[3] / "test_images" / "jawildtext")
raw = root / "raw"


def photos() -> None:
    table = pq.read_table(raw / "board_vqa" / "train-00005-of-00006.parquet", columns=["image_id", "polygons", "image"])
    rows = table.to_pylist()
    picked = sorted(random.Random(SEED).sample(range(len(rows)), min(N_PHOTOS, len(rows))))
    out = root / "photos"
    out.mkdir(exist_ok=True)
    gt = {}
    for i in picked:
        r = rows[i]
        lines = [p["text"] for p in r["polygons"] if p["text"] and JP.search(p["text"])]
        if not lines:
            continue
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        scale = min(1.0, MAX_SIDE / max(img.size))
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        name = f"{r['image_id']}.jpg"
        img.save(out / name, quality=92)  # the app uploads JPEG at 0.92
        gt[name] = lines
    (root / "photos.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"photos: {len(gt)} of {len(rows)} in the shard")


def lines() -> None:
    samples: dict[str, dict[str, bytes]] = {}
    with tarfile.open(raw / "data" / "shard-00000.tar") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            stem, ext = m.name.rsplit(".", 1)
            samples.setdefault(stem, {})[ext] = tar.extractfile(m).read()
    usable = sorted(k for k, v in samples.items()
                    if "jpg" in v and "txt" in v and JP.search(v["txt"].decode("utf-8")))
    picked = random.Random(SEED).sample(usable, min(N_LINES, len(usable)))
    out = root / "lines"
    out.mkdir(exist_ok=True)
    gt = {}
    for k in sorted(picked):
        name = Path(k).name + ".jpg"
        (out / name).write_bytes(samples[k]["jpg"])
        gt[name] = samples[k]["txt"].decode("utf-8").strip()
    (root / "lines.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"lines: {len(gt)} of {len(usable)} Japanese lines in the shard")


if __name__ == "__main__":
    photos()
    lines()
