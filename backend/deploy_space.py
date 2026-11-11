"""Upload the backend to a Hugging Face Space (Gradio SDK: free; Docker Spaces are paid).

One-time: log in with a *write* token from https://huggingface.co/settings/tokens
    hf auth login
Then, from backend/:
    venv\\Scripts\\python deploy_space.py [space-name] ["commit message"]   (default name: kanji-interpreter)
All files go up in one commit, so the Space rebuilds once.
Creates the Space if it doesn't exist (public, CPU basic). Hugging Face installs the packages and starts
app.py (a few minutes the first time). The API is served at https://<username>-<space-name>.hf.space ;
the Space must be public so the website can call it.
"""
import re
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

# what the Space runs: the app's modules and folders (nothing else: no venv, no eval, no experiments).
# Since the backend was split into modules, uploading main.py alone in the Space's web page is not enough.
CODE = ["app.py", "main.py", "pipeline.py", "cards.py", "jp.py",
        "engines/*.py", "preprocess/*.py", "postprocess/*.py", "data/*.tsv"]
REQUIREMENTS = {"requirements-space.txt": "requirements.txt"}
ABOUT = """
Backend for the Kanji Interpreter app: `POST /analyze` with a photo (multipart field `file`) returns the kanji
words found, with readings, romaji, meanings and JLPT level. The page here is a test form for the same thing.
Free and local: PaddleOCR PP-OCRv6 via RapidOCR (OCR; meikiocr as an option), SudachiPy, pykakasi and
JMdict/KANJIDIC2 via jamdict.
Source: https://github.com/Sprowsbean/Kanji_Intepreter
"""


def front_matter(repo_id: str, api: HfApi) -> str:
    """Keep the Space's own settings (title, emoji, Gradio sdk_version...), make sure it runs Python 3.11.
    The defaults below are only for a new Space."""
    meta = {"title": "Kanji Interpreter API", "emoji": "📷", "colorFrom": "pink", "colorTo": "gray",
            "sdk": "gradio", "app_file": "app.py", "pinned": "false"}
    try:
        old = Path(hf_hub_download(repo_id, "README.md", repo_type="space")).read_text(encoding="utf-8")
        m = re.match(r"---\n(.*?)\n---", old, re.S)
        for line in (m.group(1).splitlines() if m else []):
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    except Exception:  # new Space without a README yet: Hugging Face picks the Gradio version
        pass
    meta["python_version"] = '"3.11"'  # the pinned packages (numpy, opencv, sudachi) need 3.11+
    return "---\n" + "\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n---\n"


name = sys.argv[1] if len(sys.argv) > 1 else "kanji-interpreter"
api = HfApi()
user = api.whoami()["name"]
repo_id = f"{user}/{name}"
api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
here = Path(__file__).parent
readme = front_matter(repo_id, api) + ABOUT
files = {p.relative_to(here).as_posix(): p.relative_to(here).as_posix() for pat in CODE for p in sorted(here.glob(pat))}
# one commit, so the Space rebuilds once and never starts with half old, half new code
ops = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=here / local)
       for local, remote in {**files, **REQUIREMENTS}.items()]
ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode()))
api.create_commit(repo_id, ops, commit_message=sys.argv[2] if len(sys.argv) > 2 else "Update backend",
                  repo_type="space")
host = repo_id.replace("/", "-").replace("_", "-").replace(".", "-").lower()
print(f"Uploaded. Build log: https://huggingface.co/spaces/{repo_id}")
print(f"API URL (set as VITE_API_URL on Render): https://{host}.hf.space")
