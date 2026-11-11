# OCR experiments: results of Phases 1–4

Every idea from [docs/OCR_ROADMAP.md](../../docs/OCR_ROADMAP.md) was built as a switch in `pipeline.OPTIONS` and
measured with `eval/run_eval.py` on the four test sets of [BASELINE.md](BASELINE.md), on 2026-11-10/11. Nothing
was tuned on our images: models run with their published defaults, and the few thresholds come from the models'
input sizes or from the calibration table in BASELINE.md. **Everything is off by default**: the app still
runs exactly as deployed (checked: identical output on the sampled images).

## The numbers

Each cell gives **kanji words found** and the **fake kanji** count (kanji runs in the output that are not in the
reference). The real photos decide; the synthetic set is only a guard against regressions.

| setup | real photos (12) | JaWildText photos (78) | JaWildText lines (500) | synthetic guard (156) |
|---|---|---|---|---|
| **meikiocr (baseline, deployed)** | 59.6% · 6 | 58.3% · 829 | 60.8% · 168 | 98.5% · 13 |
| meiki + upscale | 61.8% · 6 | 58.3% · 820 | 61.1% · 167 | 98.5% · 13 |
| meiki + tiles | 59.6% · 10 | 69.8% · 2071 | – | 98.5% · 64 |
| meiki + correct | 58.8% · 6 | 58.4% · 825 | 60.8% · 166 | 98.5% · 13 |
| PP-OCRv6 raw (no clean-up at all) | 78.7% · 2 | 86.9% · 571 | 66.0% · 72 | 96.7% · 22 |
| PP-OCRv6, without our filters | 78.7% · 1 | 86.7% · 521 | 65.6% · 74 | 96.7% · 21 |
| **PP-OCRv6** (with our filters) | 78.7% · 1 | 86.6% · 478 | 65.1% · 71 | 96.7% · 19 |
| v6 + upscale | 78.7% · 1 | 86.6% · 475 | 65.1% · 71 | 96.9% · 19 |
| v6 + tiles | 78.7% · 1 | 89.5% · 828 | – | 97.0% · 33 |
| v6 + furigana | 78.7% · 1 | 86.6% · 480 | 65.1% · 71 | 96.7% · 19 |
| **v6 + correct** | 79.4% · 1 | 87.0% · 463 | 65.5% · 66 | 96.9% · 18 |
| v6 + correct + orient always | 80.9% · 4 | 86.2% · 456 | 59.6% · 70 | 98.5% · 9 |
| v6 + correct + orient retry | 79.4% · 1 | 87.0% · 463 | 66.4% · 68 | 96.9% · 18 |
| **v6 + correct + orient check** | 79.4% · 1 | 86.7% · 457 | 65.7% · 72 | 97.5% · 10 |
| **+ Japanese-only kanji (charset) = the new default** | **80.9% · 1** | **87.6% · 422** | **66.6% · 67** | **97.5% · 9** |
| v6 + flatten retry | 78.7% · 1 | 86.6% · 478 | – | 96.9% · 19 |
| v6 + flatten always | 83.8% · 3 | 85.1% · 456 | – | 95.7% · 10 |
| NDLOCR-Lite | 72.8% · 1 | 83.7% · 512 | 61.4% · 97 | 79.7% · 45 |
| NDLOCR-Lite + correct | 72.8% · 1 | 83.8% · 508 | 61.6% · 96 | 79.7% · 45 |

"–" = not run: tiles and flattening only change whole photos, not single-line crops. The JaWildText photos'
fake count is an upper bound: annotators skipped some real text.

## What each idea did

| idea (roadmap step) | result | verdict |
|---|---|---|
| **PP-OCRv6 small** as the reader (2.1) | Real photos +19 points, JaWildText photos +28 points, fake cards on photos −42%. Line crops: more words (+4 points), but fewer kana/kanji characters (77.0% → 67.2%): it returns nothing on 157 of 500 tiny crops (meikiocr: 87). Synthetic −1.8 points, all on sideways / upside-down images and one very large sign. | **Adopt.** The biggest win by far. |
| Correction from alternates + look-alikes, JMdict check (3.1–3.3) | With PP-OCRv6: 19 images better, 0 worse (先土-type fixes, and simplified Chinese forms the model emits, like 这 → 込, 车 → 洋). With meikiocr: 7 better, 4 worse. | **Adopt with PP-OCRv6.** |
| Orientation classifier (new; PP-LCNet doc_ori, 7 MB) | Needed because PP-OCRv6 has no rotation retry. "check" (both models must agree): synthetic 96.9 → 97.5%, photos −0.3 points (3 photos with text in two directions). "always": synthetic back to 98.5% but line crops −6 points. "retry" never fires: v6 is confident even on sideways junk. | **Adopt "check"** (a few ms; an extra read only when the classifier says the photo is turned). |
| Our fake-kanji filters with PP-OCRv6 | Remove 43 fake cards on photos for 6 lost words. | Keep on. |
| Full-resolution camera capture (1.1) | Not measurable offline. Implemented with the video frame as fallback. | Keep; **test on a real phone** (iPhone Safari has no ImageCapture, so it keeps the old path). |
| Blur warning (1.2) | Sharpness does not predict accuracy on whole photos (Spearman ≈ +0.08); on line crops it mostly tracks resolution. | Dropped; replaced by a "move closer" hint for small text. |
| Upscale small text 2x (1.3) | meikiocr: real +3 words. PP-OCRv6: no effect (it already reads near full resolution). | Drop with PP-OCRv6. |
| 2x2 tiles (1.4) | v6: photos +2.9 points, but fake cards +73% and about +50% time; meikiocr: fake cards 2.5x. | Don't turn on by default. Maybe as a "read more" button later. |
| Furigana by size and position (3.5) | No change in words found; removes the readings from the text (photos JP chars −0.3 points, as expected) and returns them as hints. | Optional. Nothing here measures its real benefit; needs a furigana test set. |
| Old forms → modern (3.4) | Works (戰 → 戦, 獸 → 獣, 兒 → 児 on the kanji grid photo). No test set measures it. | **Adopt** (card-only; can't hurt the reading). |
| Unsure marks with alternatives (3.6) | Works (e.g. 古 0.44, "maybe 吉" on a page where the name is 吉川). Some alternatives are simplified Chinese (场, 稻). | **Adopt**, after filtering alternatives to Japanese kanji. |
| UVDoc flattening (4) | "always": real photos +5 points (curved bottles and bags) but JaWildText photos −1.5 and synthetic −1.0. "retry": the confidence gate almost never fires with PP-OCRv6. | **Offer as a "curved surface" button**, not automatic. |
| NDLOCR-Lite recognizer (2.2) | Below PP-OCRv6 on every set; synthetic 79.7%. Reads old forms well, but no test set shows that as a net win. | Drop as a reader. |
| Masked-language-model scorer (3.7) | Not built: optional in the roadmap, and the dictionary check already gives 19 up / 0 down. | Revisit only if needed. |

## Speed

Most of the scan time was never the OCR. Profiling one dense photo showed card building taking about 35 s.
jamdict's `lookup()` fetches every character's KANJIDIC2 record on each call. Commit 9121bc4 caches those
lookups. The output is unchanged: 746 of 746 images give identical text, cards and errors.

Median and worst seconds per image on 2 cores. Each row was run on its own cores, at the same time as the others:

| setup | JaWildText photos | real photos | synthetic |
|---|---|---|---|
| meikiocr, before the card fix (baseline) | 33.1 / 179 | 22.3 / 68 | 2.9 / 16 |
| meikiocr, card fix | **3.4 / 9.9** | 3.8 / 12.6 | 0.4 / 8.8 |
| PP-OCRv6 + correct + orient check, card fix | 4.5 / 23.9 | **2.0 / 3.2** | 1.8 / 6.9 |

On one dense 2400 px photo, the reading step alone took 1.0 s with meikiocr and 4.5 s with PP-OCRv6. On a real
bottle photo it took 5.1 s and 1.8 s.

PP-OCRv6 reads near full resolution, which costs time on big photos full of text but finds far more of it.
meikiocr shrinks the photo to 960×544. On our own photos PP-OCRv6 is faster, because meikiocr's rotation and
padding retries cost more there.

The card fix can ship on its own, with no change in what the app reads.

## Recommendation

Switch the reader to **PP-OCRv6 small + correct + orient=check**, and turn on **variants** and **unsure** for the cards:

- **Real photos:** 59.6% → 79.4% of kanji words.
- **JaWildText photos:** 58.3% → 86.7% of kanji words.
- **Fake cards on photos:** 829 → 457.
- **Speed:** with the card fix, 2.0 s median on real photos and 4.5 s on dense ones, against 22 s and 33 s deployed today.
- **Synthetic guard:** 98.5% → 97.5%. This breaks the roadmap's "don't lower the synthetic sets" rule by 1 point, so it is your call. Against meikiocr, 3 synthetic images gain words and 5 lose them:
  - two upside-down images (−7 words);
  - one large-font sign at two sizes, where v6 misses 出口はこちら (−2);
  - a poster of single kanji, where it misses 中 and 力 (−2).

Chosen on 2026-11-11 (the 1-point synthetic drop accepted): these are now the defaults in `pipeline.py`, plus
`charset=1`. The old reader: `IKAN_ENGINE=meiki IKAN_CORRECT=0 IKAN_ORIENT=0 IKAN_VARIANTS=0 IKAN_UNSURE=0`.

Deploying it means:
- adding `rapidocr` to `requirements-space.txt`;
- making `deploy_space.py` upload the new folders;
- about 35 MB of models that download on first use.

Not done: no deploy and no push.

## Follow-ups found along the way

1. ~~**Filter out simplified Chinese.**~~ Done (postprocess/charset.py, 2026-11-11): a kanji outside JIS X 0208 that
   no JMdict word uses is swapped for the recognizer's own Japanese runner-up (顺 → 順, 级 → 級), or dropped. On
   the four sets: 30 images better, 2 worse (one word each), synthetic unchanged. Alternatives on cards are
   filtered the same way.
2. **Phone test:**
   - full-resolution capture on Android Chrome and on an iPhone;
   - the "move closer" hint.
3. **"Curved surface" and "read more (tiles)" choices in the UI**, if wanted, since neither helps on every photo.
4. **A furigana test set and an old-forms test set** (from public sources such as Aozora Bunko scans), to measure
   3.4 and 3.5 properly.

## Caveats

- The overnight matrix ran 7 evaluations side by side, each pinned to 2 logical cores. Use the timings in its
  summaries only to compare within one run. The speed section above comes from separate runs.
- The real-photo set is small: 12 photos, 136 words. A change of 1–2 words is noise.
- Results: `eval/results/<label>/` (not in git). Runs: baseline, v6, v6_nofilt, v6_upscale, v6_tiles, v6_correct,
  v6_furigana, v6_orient, v6_oretry(_lines), v6_ocheck(_lines), v6_flat_retry, v6_flat_always, m_upscale, m_tiles,
  m_correct, ndl, ndl_correct, ppv6_small (raw, `try_rapidocr.py`).
