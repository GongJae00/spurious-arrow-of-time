"""Small plotting helpers used by experiment scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_current_figure(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=350)
