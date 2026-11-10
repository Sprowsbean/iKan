"""Experiment: score PaddleOCR PP-OCRv6 (through RapidOCR, ONNX on CPU) on the same test sets as run_eval.py.

Not part of the app. Run with a separate venv so the backend's own packages stay as deployed:
    python -m venv ../../venv-exp && ../../venv-exp/Scripts/pip install rapidocr==3.10.0 onnxruntime==1.31.0
    ../../venv-exp/Scripts/python eval/try_rapidocr.py --label ppv6_small [--det small --rec small --sets ... --limit N]
Results go to eval/results/<label>/ in the same format, so the summary table compares directly with the baseline.
The models are used with RapidOCR's published defaults; nothing is tuned on our images.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval as ev  # noqa: E402


def make_reader(det: str, rec: str):
    from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR
    ort = {"EngineConfig.onnxruntime.intra_op_num_threads": 2, "EngineConfig.onnxruntime.inter_op_num_threads": 1}
    eng = RapidOCR(params={
        "Global.log_level": "warning",
        "Det.engine_type": EngineType.ONNXRUNTIME, "Det.lang_type": LangRec.JAPAN,  # PP-OCRv6 models are multilingual; routing wants a language
        "Det.ocr_version": OCRVersion.PPOCRV6, "Det.model_type": ModelType(det),
        "Rec.engine_type": EngineType.ONNXRUNTIME, "Rec.lang_type": LangRec.JAPAN,
        "Rec.ocr_version": OCRVersion.PPOCRV6, "Rec.model_type": ModelType(rec),
        **ort,
    })

    def read(img) -> str:
        out = eng(img)
        return "\n".join(out.txts) if out.txts else ""
    return read


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--det", default="small")
    ap.add_argument("--rec", default="small")
    ap.add_argument("--sets", default=",".join(ev.SETS))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    import cv2
    ev.use_two_cores()
    read = make_reader(a.det, a.rec)
    items = ev.load_sets(a.sets.split(","))
    read(ev.as_sent(cv2.imread(str(items[0][1]))))  # warm-up
    out = ev.RESULTS / a.label
    out.mkdir(parents=True, exist_ok=True)
    counts, records = Counter(), []
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for set_name, path, ref in items:
            if a.limit and counts[set_name] >= a.limit:
                continue
            counts[set_name] += 1
            img = ev.as_sent(cv2.imread(str(path)))
            t = time.perf_counter()
            try:
                text, err = read(img), None
            except Exception as e:
                text, err = "", f"{type(e).__name__}: {e}"
            dt = time.perf_counter() - t
            rec = {"set": set_name, "file": path.name, "seconds": round(dt, 3), "error": err, "text": text,
                   "cards": [], **ev.score(ref, text)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
            print(f"{set_name:10} {len(records):4}  {dt:5.2f}s  {path.name[:40]}", flush=True)
    summary = ev.summarize(records, [])
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
