# OCR baseline (Phase 0)

The starting numbers every later change is compared against. Produced by

    cd backend
    PYTHONUTF8=1 venv/Scripts/python eval/run_eval.py --label baseline

on 2026-11-10, pinned to 2 CPU cores like the Hugging Face Space. The code under test is meikiocr plus our
clean-up steps exactly as deployed (commit 91b2429). Every image is sent as the app sends it (`as_sent()`: at
most 2400 px, plus the small-crop margin from `CropView.tsx`). The test sets are described in
[sets/README.md](sets/README.md).

| set | images | CER | exact | kanji words | JP chars | fake kanji | s median | s max | crashes |
|---|---|---|---|---|---|---|---|---|---|
| jaw_lines | 500 | 39.2% | 34.2% | 470/773 (60.8%) | 2837/3685 (77.0%) | 168 | 1.11 | 38.10 | 0 |
| jaw_photos | 78 | - | - | 2229/3824 (58.3%) | 14810/23466 (63.1%) | 829 | 33.13 | 179.44 | 0 |
| real | 12 | - | - | 81/136 (59.6%) | 96/134 (71.6%) | 6 | 22.31 | 67.61 | 0 |
| synthetic | 156 | - | - | 665/675 (98.5%) | 2677/2700 (99.1%) | 13 | 2.93 | 16.02 | 0 |

- **CER / exact:** character error rate and the share of lines read exactly. These are only computed for
  single-line crops, where the reading order is known.
- **kanji words:** reference kanji words (or the listed words) found in the output.
- **JP chars:** reference kana and kanji found, ignoring order.
- **fake kanji:** kanji runs in the output that are not in the reference.
- **s:** seconds per image on 2 cores, with the models already loaded.

## Recognizer confidence vs correctness

Each raw character meikiocr returned on `jaw_lines` (before our filters), matched against the reference
line.

| confidence | characters | correct |
|---|---|---|
| 0.0-0.3 | 471 | 5.5% |
| 0.3-0.5 | 683 | 13.6% |
| 0.5-0.7 | 639 | 38.7% |
| 0.7-0.9 | 2070 | 68.5% |
| 0.9-1.0 | 2405 | 76.2% |

## What the baseline says

- **Real photos are the problem, not clean print.** About 60% of kanji words are found on JaWildText lines,
  JaWildText photos and our own photos, against 98.5% on the synthetic set. The synthetic set can only guard
  against regressions; it cannot show progress.
- **Whole photos are far too slow.** The median is 33 s and the worst 179 s per photo on 2 cores; 76 of 78
  photos take over 10 s. The slowest photos contain lots of small text, so the time likely goes to the
  per-box re-reads and the rotation retries (`engines/meiki.py`). Phase 1 tiling and any extra pass must be
  timed against this, and making it faster is a goal in its own right.
- **Confidence does rank errors, but it is not a probability.** Characters below 0.5 are almost always wrong
  (under 14% correct), so a low-confidence gate for corrections (Phase 3) is usable. Even above 0.9, one in four
  characters is wrong, so high confidence can't be read as "certainly right". Each bin's accuracy is lower
  than its confidence.
- **Fake kanji on photos:** 829 output kanji runs on 78 photos are not in the reference. Some are real text
  the annotators skipped, so this is an upper bound, but it is the number Phase 3 filtering should lower.
- **Empty results:** 87 of 500 line crops returned no text at all. 32 synthetic images also returned none;
  these are the photos without Japanese, where empty is correct.
