"""Figure: frame-local vs order-encoded directional cue audit.

Panel A: per-frame direction decodability (single-frame probe accuracy vs t)
         for the trail variant and the order-encoded variant.
Panel B: nuisance-only sequence model OOD accuracy under order interventions
         (ordered / frame-shuffled / order-reversed) in both variants.

Inputs:
  results/extended/temporal_evidence_audit.json      (trail variant)
  results/extended/temporal_evidence_audit_oe.json   (order-encoded variant)
  results/extended/nuisance_order_probe.json

Usage:
  python -m src.visualization.temporal_audit_figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.paper_style import (
    CORE,
    GRID,
    MUTED_TEXT,
    NUISANCE,
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
    trail = json.load(open(RES / "temporal_evidence_audit.json"))
    oe = json.load(open(RES / "temporal_evidence_audit_oe.json"))
    probe = json.load(open(RES / "nuisance_order_probe.json"))

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.6))

    # Panel A: per-frame direction decodability.
    ax = axes[0]
    L = len(trail["per_frame"]["dir_iid"])
    ts = np.arange(L)
    for data, color, label in [
        (trail, NUISANCE, "trail variant (frame-local residue)"),
        (oe, SEQUENCE, "order-encoded variant"),
    ]:
        means = np.array([d["mean"] for d in data["per_frame"]["dir_iid"]])
        stds = np.array([d["std"] for d in data["per_frame"]["dir_iid"]])
        ax.errorbar(ts, means, yerr=stds, color=color, linewidth=1.6, marker="o",
                    markersize=4.0, capsize=2.0, label=label, zorder=3)
    ax.axhline(0.5, color=MUTED_TEXT, linewidth=0.9, linestyle=(0, (4, 2)))
    ax.text(L - 1.0, 0.53, "chance", fontsize=6.6, color=MUTED_TEXT, ha="right")
    style_axis(ax)
    ax.set_xlabel("frame index $t$", fontsize=7.5)
    ax.set_ylabel("single-frame direction accuracy", fontsize=7.5)
    ax.set_ylim(0.4, 1.05)
    ax.set_xticks(ts)
    ax.legend(fontsize=6.4, loc="center left", frameon=False)
    panel_title(ax, "A", "Where does the directional cue live?",
                "per-frame probes (10 seeds)")

    # Panel B: order interventions on the nuisance-only model (OOD accuracy).
    ax = axes[1]
    conditions = ["ood", "ood_shuffled", "ood_reversed_order"]
    cond_labels = ["ordered", "frame-\nshuffled", "order-\nreversed"]
    width = 0.36
    xs = np.arange(len(conditions))
    for k, (variant, color, label) in enumerate([
        ("trail", NUISANCE, "trail variant"),
        ("order_encoded", SEQUENCE, "order-encoded variant"),
    ]):
        means = [probe[variant][c]["mean"] for c in conditions]
        stds = [probe[variant][c]["std"] for c in conditions]
        ax.bar(xs + (k - 0.5) * width, means, width, yerr=stds, capsize=2.0,
               color=color, edgecolor="white", linewidth=0.6, label=label, zorder=3)
        for x, m in zip(xs + (k - 0.5) * width, means, strict=True):
            ax.text(x, m + 0.05, f"{m:.2f}", fontsize=6.2, ha="center", color=TEXT)
    ax.axhline(0.5, color=MUTED_TEXT, linewidth=0.9, linestyle=(0, (4, 2)))
    style_axis(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(cond_labels, fontsize=7.0)
    ax.set_ylabel("OOD accuracy (nuisance-only)", fontsize=7.5)
    ax.set_ylim(0.0, 1.13)
    ax.legend(fontsize=6.4, loc="upper left", frameon=False)
    panel_title(ax, "B", "Does the cue need temporal order?",
                "test-time order interventions (10 seeds)")

    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.17, wspace=0.26)
    save_figure(fig, RES / "figures", "ext_fig7_temporal_audit")
    print("saved ext_fig7_temporal_audit")


if __name__ == "__main__":
    main()
