# Test sets for `eval/run_eval.py`

The images are kept outside the repo, in `test_images/` next to the project folder: the real photos are
personal, and the downloaded dataset is large. This file says where each set comes from so it can be rebuilt.

| Set | Images | Source | Reference |
|---|---|---|---|
| `jaw_lines` | 500 single text lines | [JaWildText](https://huggingface.co/datasets/llm-jp/jawildtext) (real photos taken in Japan, Apache-2.0), via its per-line crops [`nagohachi/jawildtext_cropped`](https://huggingface.co/datasets/nagohachi/jawildtext_cropped) shard 00000 | the dataset's transcription of each line |
| `jaw_photos` | 78 whole photos | JaWildText `board_vqa` shard 00005 (signboards, posters, products) | the text of every annotated region |
| `real` | 12 phone photos and images | the developer's photos (`test_images/curved/`: books, bottles, a tube, a bag) plus `manga_vertical.png` and `user_grid.png` | `curved/words.json` (kanji words read off each photo) and `extra_gt.json` (full text) |
| `synthetic` | 156 generated images | `test_images/make_*.py` (signs, menus, vertical books, rotated photos, marked pages, photos without Japanese) | `test_images/gt_all.json` |

Rules:

- **JaWildText and `real` decide whether a change ships.** `synthetic` only guards against breaking what
  already works; never tune on it.
- **Don't tune on `real` either.** It is ten photos; use published defaults and existing models, and treat a gain
  here as a reason to check JaWildText, not as proof.
- **Rebuilding JaWildText:** the download command and the fixed-seed sampling are in `eval/prepare_jawildtext.py`.
  Same seed, same images.
