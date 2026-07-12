"""Figure: per-frame core-label accuracy in the sequence-dependent core
configuration, with multi-frame reference lines.

Usage:
  python -m src.visualization.hardcore_figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.paper_style import (
    CORE,
    MUTED_TEXT,
    SEQUENCE,
    TEXT,
    apply_style,
    panel_title,
    save_figure,
    style_axis,
)

RES = Path("results/extended")


def main() -> None:
    apply_style()
    pf = json.load(open(RES / "hardcore_perframe.json"))["per_frame"]
    fin = json.load(open(RES / "hardcore_probe_final.json"))

    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    ts = np.arange(len(pf))
    means = np.array([a["mean"] for a in pf])
    stds = np.array([a["std"] for a in pf])
    ax.errorbar(ts, means, yerr=stds, color=CORE, linewidth=1.7, marker="o",
                markersize=4.2, capsize=2.0, zorder=3, label="single-frame probe")
    gru = fin["gru_iid"]["mean"]
    mean_probe = fin["mean_probe"]["mean"]
    ax.axhline(gru, color=SEQUENCE, linewidth=1.3, linestyle=(0, (5, 2)))
    ax.text(0.05, gru + 0.015, f"full multi-frame sequence ({gru:.3f})",
            fontsize=6.6, color=SEQUENCE)
    ax.axhline(mean_probe, color=MUTED_TEXT, linewidth=1.1, linestyle=(0, (2, 2)))
    ax.text(0.05, mean_probe + 0.015, f"temporal-mean probe ({mean_probe:.3f})",
            fontsize=6.6, color=MUTED_TEXT)
    ax.axhline(0.5, color=MUTED_TEXT, linewidth=0.9, linestyle=(0, (4, 2)))
    ax.text(len(pf) - 1.0, 0.515, "chance", fontsize=6.4, color=MUTED_TEXT, ha="right")
    style_axis(ax)
    ax.set_xlabel("frame index $t$", fontsize=7.5)
    ax.set_ylabel("core-label accuracy", fontsize=7.5)
    ax.set_xticks(ts)
    ax.set_ylim(0.45, 1.03)
    panel_title(ax, "", "Multi-frame core", "clean core channel, 5 seeds")
    fig.subplots_adjust(left=0.14, right=0.985, top=0.82, bottom=0.19)
    save_figure(fig, RES / "figures", "ext_fig8_hardcore_core")
    print("saved ext_fig8_hardcore_core")


if __name__ == "__main__":
    main()
