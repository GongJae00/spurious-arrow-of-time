#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 src/visualization/submission_figures.py
make -C paper/neurocomputing clean all

python3 - <<'PY'
from pathlib import Path
import os
import shutil
import subprocess

pdf = Path("paper/neurocomputing/main.pdf")
out_dir = Path(os.environ.get("RENDER_OUT", "/tmp/spurious_arrow_manuscript_pages"))
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
pages = text.split("\f")
fig_pages: dict[int, int] = {}
for fig_no in (1, 3, 4, 5):
    needle = f"Figure {fig_no}:"
    for idx, page in enumerate(pages, start=1):
        if needle in page:
            fig_pages[fig_no] = idx
            break
    if fig_no not in fig_pages:
        raise SystemExit(f"Could not locate {needle} in {pdf}")

for fig_no, page_no in fig_pages.items():
    prefix = out_dir / f"fig{fig_no:02d}_page"
    subprocess.check_call(
        [
            "pdftoppm",
            "-f",
            str(page_no),
            "-l",
            str(page_no),
            "-png",
            "-r",
            "220",
            str(pdf),
            str(prefix),
        ]
    )
    rendered = sorted(out_dir.glob(f"fig{fig_no:02d}_page-*.png"))
    latest = rendered[-1] if rendered else Path(f"{prefix}-{page_no}.png")
    print(f"Figure {fig_no}: page {page_no} -> {latest}")
PY

pdfinfo paper/neurocomputing/main.pdf | grep -E "Pages|File size"
