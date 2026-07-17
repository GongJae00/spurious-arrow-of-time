"""Figures: frame-local vs order-encoded directional cue audit.

ext_fig7a_perframe_probe:      per-frame direction decodability
                               (single-frame probe accuracy vs t) for the
                               trail variant and the order-encoded variant.
ext_fig7b_order_interventions: nuisance-only sequence model OOD accuracy
                               under order interventions (ordered /
                               frame-shuffled / order-reversed).

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
    MUTED_TEXT,
    NUISANCE,
    SEQUENCE,
    TEXT,
    apply_style,
    save_figure,
    style_axis,
)

RES = Path("results/extended")


def main() -> None:
    apply_style()
    trail = json.load(open(RES / "temporal_evidence_audit.json"))
    oe = json.load(open(RES / "temporal_evidence_audit_oe.json"))
    probe = json.load(open(RES / "nuisance_order_probe.json"))

    # Figure A: per-frame direction decodability.
    fig, ax = plt.subplots(figsize=(3.6, 2.55))
    L = len(trail["per_frame"]["dir_iid"])
    ts = np.arange(L)
    for data, color, label in [
        (trail, NUISANCE, "trail variant (frame-local residue)"),
        (oe, SEQUENCE, "order-encoded variant"),
    ]:
        means = np.array([d["mean"] for d in data["per_frame"]["dir_iid"]])
        stds = np.array([d["std"] for d in data["per_frame"]["dir_iid"]])
        ax.errorbar(ts, means, yerr=stds, color=color, linewidth=1.7,
                    marker="o", markersize=4.2, capsize=2.0, label=label,
                    zorder=3)
    ax.axhline(0.5, color=MUTED_TEXT, linewidth=0.9, linestyle=(0, (4, 2)))
    ax.text(L - 1.0, 0.525, "chance", fontsize=7.6, color=MUTED_TEXT,
            ha="right")
    style_axis(ax)
    ax.set_xlabel("frame index $t$", fontsize=9.0)
    ax.set_ylabel("single-frame direction acc.", fontsize=9.0)
    ax.tick_params(labelsize=8.2)
    ax.set_ylim(0.4, 1.05)
    ax.set_xticks(ts)
    ax.legend(fontsize=7.8, loc="center left", frameon=False)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.97, bottom=0.19)
    save_figure(fig, RES / "figures", "ext_fig7a_perframe_probe")
    print("saved ext_fig7a_perframe_probe")

    # Figure B: order interventions on the nuisance-only model.
    fig, ax = plt.subplots(figsize=(3.6, 2.55))
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
               color=color, edgecolor="white", linewidth=0.6, label=label,
               zorder=3)
        for xpos, m in zip(xs + (k - 0.5) * width, means, strict=True):
            ax.text(xpos, m + 0.05, f"{m:.2f}", fontsize=7.4, ha="center",
                    color=TEXT)
    ax.axhline(0.5, color=MUTED_TEXT, linewidth=0.9, linestyle=(0, (4, 2)))
    style_axis(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(cond_labels, fontsize=8.6)
    ax.set_ylabel("OOD accuracy (nuisance-only)", fontsize=9.0)
    ax.tick_params(axis="y", labelsize=8.2)
    ax.set_ylim(0.0, 1.16)
    ax.legend(fontsize=7.8, loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.97, bottom=0.15)
    save_figure(fig, RES / "figures", "ext_fig7b_order_interventions")
    print("saved ext_fig7b_order_interventions")


if __name__ == "__main__":
    main()
