# -*- coding: utf-8 -*-
"""Six-gate audit decision-flow figure (paper Fig. 2).

Usage: python -m src.visualization.audit_flow_figure --out fig06_audit_flow.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="fig06_audit_flow.pdf")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(4.2, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 21)
    ax.axis("off")

    BOX = dict(boxstyle="round,pad=0.32", fc="#eef3fb", ec="#33518a", lw=1.1)
    FAIL = dict(boxstyle="round,pad=0.28", fc="#fbeeee", ec="#8a3333", lw=1.0)
    OUT = dict(boxstyle="round,pad=0.32", fc="#eefbef", ec="#2f7a3a", lw=1.1)

    gates = [
        (19.6, "G1: Core learnable\nalone?", "reject /\ntask too hard"),
        (16.9, "G2: Nuisance\npredictive alone?", "reject /\nweak shortcut cue"),
        (14.2, "G3: Endpoint leakage\ncontrolled?", "reject /\nendpoint leakage"),
        (11.5, "G4: Reference-learner\nrecovery (no-spurious)?",
         "reject /\ncore not recovered\nunder ref. budget"),
        (8.8, "G5: Reversal collapse\n(signature match)?",
         "inconclusive /\nsignature mismatch"),
    ]
    cx = 3.9
    for y, q, fail in gates:
        ax.text(cx, y, q, ha="center", va="center", fontsize=8.3, bbox=BOX)
        ax.text(7.0, y, fail, ha="left", va="center", fontsize=6.8, bbox=FAIL)
        ax.add_patch(FancyArrowPatch((6.1, y), (6.82, y), arrowstyle="-|>",
                                     mutation_scale=9, color="#8a3333",
                                     lw=0.9))
        ax.text(6.45, y + 0.32, "no", fontsize=6.8, color="#8a3333",
                ha="center")

    for (y1, _, _), (y2, _, _) in zip(gates, gates[1:]):
        ax.add_patch(FancyArrowPatch((cx, y1 - 0.62), (cx, y2 + 0.62),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="#33518a", lw=1.1))
        ax.text(cx + 0.28, (y1 + y2) / 2, "yes", fontsize=6.8,
                color="#33518a")

    y6 = 6.0
    ax.text(cx, y6, "G6: Cue-locality audit\n(probes + order interventions;\n"
            "Route A construction-certified /\nRoute B intervention-certified)",
            ha="center", va="center", fontsize=7.4, bbox=BOX)
    ax.add_patch(FancyArrowPatch((cx, gates[-1][0] - 0.85), (cx, y6 + 0.95),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color="#33518a", lw=1.1))
    ax.text(cx + 0.28, (gates[-1][0] + y6) / 2 - 0.1, "yes", fontsize=6.8,
            color="#33518a")

    outs = [(1.15, 3.45, "frame-local\n(single-frame\nprobe $\\geq$ 0.8)"),
            (4.6, 3.7, "order-invariant\nmulti-frame\n(set probe\n$\\geq$ 0.8)"),
            (8.3, 3.45, "order-encoded\n(only ordered\nreadout $\\geq$ 0.8)")]
    for x, ye, t in outs:
        ax.text(x, 2.6, t, ha="center", va="center", fontsize=6.6, bbox=OUT)
        ax.add_patch(FancyArrowPatch((cx, y6 - 1.05), (x, ye),
                                     arrowstyle="-|>", mutation_scale=9,
                                     color="#2f7a3a", lw=1.0))
    ax.text(4.7, 0.7, "shortcut attribution with cue class",
            ha="center", fontsize=8.0, style="italic")

    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
