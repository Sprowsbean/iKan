# OCR accuracy roadmap

## Context

iKan reads Japanese text from phone photos with
meikiocr on a free Hugging Face Space and turns it into kanji cards. It works well on clean printed text but
misses on real photos: small dense label text, glare, tilted or decorative text, old character forms, furigana,
and context errors (先土 for 先生, 急 read as 食).

Four research passes (November 2026) looked for free, CPU-only ways to get clearly better results. This is
the compiled result, written as a build brief: what to do, in which order, with the evidence for each step.
The full notes with sources are kept outside the repo, in `research_notes/Free OCR accuracy improvements/`
next to the project folder (`preprocessing.md`, `ocr_models.md`, `post_correction.md`, `pipeline_practices.md`).
## Progress

Work happens on the branch `ocr-accuracy`, one phase per commit. Nothing from it is deployed yet; the live app
still runs the code from `main`.

| Step | Status | Result |
|---|---|---|
| Phase 0: measurement | done (commit 3414dc9) | `backend/eval/run_eval.py`, 4 test sets, baseline in [backend/eval/BASELINE.md](../backend/eval/BASELINE.md) |
| Split `main.py` into modules | done (commit d51fb11) | 746 of 746 test images give identical text, cards and errors |
| Phase 1: better input | done (commit 972320f) | full-res capture, "move closer" hint; upscale / tiles measured, off |
| Phase 2: second reader | done (commit f4f7b5c) | PP-OCRv6 small: real photos 59.6% → 78.7% of kanji words; NDLOCR-Lite below it |
| Phase 3: fix after reading | done (commit 8ddf952) | correction: 19 images better, 0 worse with PP-OCRv6; old forms, furigana, unsure marks |
| Phase 4: curved surfaces | done (commit c17d7d0) | UVDoc helps curved items, hurts flat ones: offer as a choice |
| Card building speed | done (commit 9121bc4) | photos 33 s → 3.4 s median, identical output |
| Decision: what to switch on | **waiting for review** | all results and the recommendation: [backend/eval/RESULTS.md](../backend/eval/RESULTS.md) |

What the baseline showed, to keep in mind for the next phases:

- About 60% of kanji words are found on real photos (JaWildText and our own), against 98.5% on synthetic.
  Progress has to show up on the real sets.
- Whole photos take 26–33 s median and up to about 3 minutes on 2 cores. Speed is a goal too, and timings
  must be compared between runs made back to back (medians moved by up to 1.7× between two runs of
  identical code, because of load on the PC).
- Recognizer confidence ranks errors (below 0.5 is almost always wrong) but is overconfident (above 0.9 is
  still wrong 1 time in 4). A low-confidence gate is usable; "high confidence = correct" is not.
- 87 of 500 line crops return no text at all.

Open questions to reflect on before Phase 1:

- Should speed come first? A 30 s scan on the free Space may matter more to users than a few points of
  accuracy, and Phase 1 tiling would make photos slower still.
- Why do 87 line crops return nothing (detector misses on tight crops, or filters removing real text)?
  Looking at the per-image results in `backend/eval/results/` would answer this without tuning on them.
- After deploying this branch, the Space has to be updated with `deploy_space.py`. Uploading only `main.py`
  no longer works.

## Rules that bound every step

- **Zero cost:** open models, ONNX Runtime on the Space's CPU (2 vCPU, 16 GB). No API keys. ZeroGPU is never
  required (visitors get only 2 min/day).
- **Do not delete or recreate the Space** `Okutsu1/kanji_ocr`. Hugging Face docs now say new Gradio/Docker
  Spaces need a paid plan; free accounts keep up to 2 Gradio ZeroGPU Spaces, which is what this one is.
- **No tuning on our own images.** Use existing models and published defaults. A change ships only if it
  improves the real-photo test set (Phase 0) and doesn't lower the synthetic sets.
- **A few seconds per scan.** Extra passes run only when the first pass looks weak (gated), capped at 1–2 retries.
- **Never invent text:** a corrector may only choose among characters the recognizer itself proposed, or a
  known look-alike, and only where confidence is low and the word isn't in the dictionary.

## Target structure

`backend/main.py` is one ~450-line file doing OCR, fixes and cards. Split it so engines and steps can be
swapped and measured one at a time. `main.py` stays as the entry point (`analyze`, `analyze_image`, `app`) so
`app.py` and the Space keep working.

```
backend/
  main.py                 thin: FastAPI route + analyze_image(), wires the pipeline
  pipeline.py             run(img, options) -> lines; gated retries live here
  engines/
    base.py               Engine protocol: read(img) -> [Line(text, chars[char, conf, box, alts], box, vertical)]
    meiki.py              current code moved here (_ocr_pass, _read_vertical, _oriented_pass, _pick ...)
    paddle_v6.py          PP-OCRv6 small det+rec via onnxruntime (Phase 2)
    ndl_lite.py           NDLOCR-Lite PARSeq recognizer (Phase 2, optional)
  preprocess/
    scale.py              text-height-aware upscale; 2x2 tiling with overlap + box merge
    flatten.py            UVDoc (already written on branch experiment/curved: main.flatten)
  postprocess/
    lookalike.py          fix_lookalikes + script-context rule (existing LOOKALIKE table moves here)
    candidates.py         top-k alternates, OOV-triggered substitution (JMdict/Sudachi check)
    variants.py           old form -> modern form lookup (戰 -> 戦), keeps "printed as"
    furigana.py           size/position rule: small glyphs beside a line = reading hint, not text
    filters.py            existing fake-kanji filters (_junk_line, _keep, LATIN_JUNK_CONF)
  cards.py                make_card, char_cards, glosses, reading, kanji_info (unchanged logic)
  eval/
    run_eval.py           one command: every set x every pipeline config -> CER, word recall, fake cards, seconds
    sets/README.md        where each test set comes from; real photos stay out of git
  deploy_space.py         must upload the new folders too (today it uploads 3 files)
```

Deploy note: after the split, updating the Space by uploading only `main.py` in the web UI no longer works.
Use `deploy_space.py` (needs `hf auth login` typed by the user in their own terminal) or upload the folders.

## Phase 0 — measure first (prerequisite for everything)

- Build `backend/eval/run_eval.py` from the throwaway scripts used so far (logic: detect → crop → OCR →
  count words; see `backend/experiments/curved_eval.py` on `experiment/curved`).
- Test sets:
  1. **JaWildText** (real photos taken in Japan, incl. handwriting; Apache-2.0, on Hugging Face; paper
     arXiv 2603.27942). Neutral set; use a fixed random subset of a few hundred images. Metric: character error rate.
  2. **The user's real photos** in `test_images/curved/` with `words.json` (10 photos, 109 words) plus
     `manga_vertical.png` and `user_grid.png`. Grow this set with full-size phone photos.
  3. Existing synthetic sets (`test_images/gt_all.json`) as a regression guard only.
- Report per set: CER, kanji-word recall, fake cards, median and worst seconds on 2 threads.
- Also measure whether meikiocr's confidence tracks correctness on photos (needed before any confidence gate).

## Phase 1 — better input (cheapest, most likely gains)

The detector shrinks every image to 960×544 (or 320×192 in the browser), so small text is lost before reading.

1. **Capture at full resolution.** `src/components/CameraCapture.tsx` grabs a 1920×1080 video frame. Use
   `ImageCapture.takePhoto()` where available (reliable on Android Chrome; iOS Safari support is unclear, so
   keep the video-frame path as fallback and test on a real iPhone).
2. **Retake prompt instead of repair.** In the camera view, measure sharpness (variance of the Laplacian) on
   the detected text area and overexposure (glare) and show "Hold steady" / "Too much glare". Published
   thresholds vary (100–400), so calibrate on Phase 0 photos rather than picking a number.
3. **Text-height-aware scaling** (`preprocess/scale.py`): if detected text is small, upscale 2× with plain
   bicubic and re-read (our manga test: 11 → 13 of 19; 3× gave nothing). No GAN super-resolution: it can invent strokes.
4. **Tiled reading for big photos with small text:** 2×2 tiles with overlap, map boxes back, merge
   duplicates. Trigger when text is small or a pass returns close to meikiocr's 64-box cap. No published OCR
   numbers for tiling, so this must earn its place in Phase 0.
5. Keep sending color JPEG at quality ≥ 0.9; `MAX_SIDE = 2400` in `src/components/CropView.tsx` stays.

## Phase 2 — a second reader

meikiocr says of itself that it is tuned for game text and may do poorly on real-world photos (and caps at 64
boxes, 48 characters per line, vertical "beta").

1. **PP-OCRv6 small** (PaddleOCR 3.7, June 2026; Apache-2.0; ~20 MB recognizer; ONNX exports on Hugging Face;
   ~0.6 s/image on a server CPU). Vendor's own Japanese score: 88.2 vs 75.8 for the v5 mobile generation the
   earlier comparison used. No independent Japanese benchmark exists, so: implement `engines/paddle_v6.py`,
   run Phase 0, then decide between (a) replace, (b) keep meikiocr and use v6 as a second opinion.
   Second-opinion rule: where both agree, accept; where they differ, decide by dictionary check, else by
   confidence, else show both to the learner.
2. **NDLOCR-Lite** (National Diet Library, 2026; CC BY 4.0; ONNX; built for CPU): candidate specialist for
   book pages, vertical text, old character forms and handwriting (CER 0.268 on JaWildText handwriting, near
   2–8B vision-language models). Only a command-line tool is documented, so wiring its recognizer onto our
   line crops needs reading its source. Check the DINOv3 licence on its layout model before bundling.
3. Not now: manga-ocr / Baberu (120–400 MB, invent text on non-text crops; revisit if manga becomes a
   focus), small vision-language models (≈30 s per page on CPU), YomiToku (non-commercial licence),
   comic-text-detector (GPL).

## Phase 3 — fix errors after reading

In order; each is small and measurable.

1. **Expose top-k alternates** from the meikiocr recognizer (its README says the raw output has them; inspect
   the ONNX output shape first). Add `alts` and `conf` to each character.
2. **Look-alike rule with script context:** カ/力, ロ/口, ー/一, ニ/二 decided by neighbours (katakana run vs
   kanji compound) and dictionary. Extends the existing `LOOKALIKE` table and `fix_lookalikes()` in `main.py`.
3. **Dictionary-triggered substitution:** when Sudachi/JMdict doesn't know a low-confidence span, try the
   alternates plus visually similar kanji (Yencken kanji-confusion sets, CC BY 3.0,
   https://lars.yencken.org/datasets/kanji-confusion). Accept only if the result is a dictionary word and the
   original is not. (先土 → 先生.)
4. **Old forms:** if a kanji has no card data, look up its modern form (Unicode 18 `kJapaneseNewVariant`, or
   `dahlia/shinjitai-table` / `kyujipy` after checking their licences) and show the modern card with
   "printed as 戰".
5. **Furigana by geometry:** glyphs under about half the main text height, sitting beside/above a line, are
   furigana: remove from the text and keep as a reading hint to confirm the kanji reading. Published rule,
   76–91% F1; it slightly hurts pages with little furigana, so gate on detection.
6. **Show uncertainty:** API adds `confidence` and `alternates` per card; the card UI
   (`src/components/KanjiFlashcard.tsx`, `src/types/kanji.ts`) marks unsure characters with "or: 食?".
7. Optional, measure latency first: a character-level masked language model (`tohoku-nlp/bert-base-japanese-char-v3`,
   Apache-2.0, or `sbintuitions/modernbert-ja-30m`, MIT) as a scorer over the candidate set only. A published
   2026 test recovered 27% of look-alike errors this way.
8. No generative LLM correction: small models rewrote correct text in every study found.

## Phase 4 — curved surfaces, gated

`experiment/curved` already has `flatten()` (UVDoc, Apache-2.0, 32 MB, ~0.5 s): real photos 67 → 85 of 109
words, flat synthetic 216 → 203. Merge it only as a gated retry (first pass weak, or the user taps "Curved
surface"), applied to the full-resolution photo. No published automatic gate exists.

## What to avoid (evidence says it hurts or doesn't help)

Binarization and grayscale conversion; "whiten the page" cleanup (our test: 1121 → 1114); int8 quantizing the
detector (120 → 61 of 157 lines); GAN super-resolution; PP-OCRv5 server (slow, no Japanese gain); always-on
multi-pass (small gains, cost doubles per pass); letting any corrector change high-confidence text.

## Licences to carry into the README

meikiocr code Apache-2.0, its models LGPL-3.0 · PP-OCRv6 Apache-2.0 · NDLOCR-Lite CC BY 4.0 (+ DINOv3 terms,
unreviewed) · UVDoc Apache-2.0/MIT · Yencken sets CC BY 3.0 · JaWildText Apache-2.0.

## Suggested order and kickoff prompt

Order: Phase 0 → restructure (behaviour unchanged, verified by Phase 0 numbers) → Phase 1 → Phase 2.1 →
Phase 3.1–3.6 → Phase 4 → Phase 2.2 / 3.7 if still needed.

Prompt for the build session:

> Read `docs/OCR_ROADMAP.md` and `docs/SCANNER.md`. Do Phase 0 first: build `backend/eval/run_eval.py` with
> the three test sets and record the baseline. Then split `backend/main.py` into the target structure with
> no behaviour change, and prove it with identical Phase 0 numbers. Work on a branch, one phase per commit;
> report the table after each phase and stop for review before starting the next. Don't tune on our own
> images, don't recreate the Space, and don't deploy until I say so.

## How to verify each phase

For each phase: `python backend/eval/run_eval.py` before and after; `npm run typecheck`
  and `npm run lint`; local `uvicorn main:app --port 8001` + a scan through the built site; then one scan
  against the Space after deploy.
