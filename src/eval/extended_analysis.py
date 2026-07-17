"""Analyze and visualize the extended (revision) ablation experiments.

Reads the four extended-profile summaries and produces markdown tables plus
manuscript-style figures for:
  1. nuisance-label correlation sweep
  2. nuisance scale sweep
  3. model-family comparison
  4. channel-mixing (two-channel vs additive)

Usage:
  python -m src.eval.extended_analysis --root results/extended --out results/extended/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.visualization import paper_style as ps


def load_summary(root: Path, profile: str) -> dict[str, Any]:
    path = root / profile / "summary.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get(scen: dict[str, Any], method: str, metric: str) -> tuple[float, float, int]:
    """Return (mean, std, n) or (nan, nan, 0) if missing."""
    node = scen.get(method, {}).get(metric)
    if not node:
        return float("nan"), float("nan"), 0
    return float(node["mean"]), float(node["std"]), int(node["n"])


# ----------------------------------------------------------------------------- tables
def md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([line, sep, body])


def fmt(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "--"
    return f"{mean:.3f} ({std:.3f})"


# ----------------------------------------------------------------------------- fig 1
def fig_corr_sweep(summary: dict[str, Any], out: Path) -> str:
    order = [("corr_0p55", 0.55), ("corr_0p70", 0.70), ("corr_0p85", 0.85), ("corr_0p97", 0.97)]
    scen = summary["scenarios"]
    xs = [c for _, c in order]
    series = {
        "sequence_erm": ("Sequence ERM (OOD)", ps.SEQUENCE),
        "nuisance_only_oracle": ("Nuisance-only reference (OOD)", ps.NUISANCE),
        "core_only_oracle": ("Core-only reference (OOD)", ps.CORE),
        "counterfactual_invariance": ("Counterfactual (OOD)", ps.COUNTERFACTUAL),
    }
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    # ERM IID reference (dashed)
    iid = [get(scen[s], "sequence_erm", "iid_test_accuracy")[0] for s, _ in order]
    ax.plot(xs, iid, "--", color=ps.SEQUENCE, lw=1.0, alpha=0.6, label="Sequence ERM (IID)")
    for method, (label, color) in series.items():
        means = [get(scen[s], method, "ood_test_accuracy")[0] for s, _ in order]
        stds = [get(scen[s], method, "ood_test_accuracy")[1] for s, _ in order]
        ax.errorbar(xs, means, yerr=stds, marker="o", ms=4, lw=1.4, color=color,
                    capsize=2.5, label=label)
    ax.axhline(0.5, color=ps.LIGHT_TEXT, lw=0.7, ls=":")
    ax.set_xlabel("Train nuisance-label correlation")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(xs)
    ps.style_axis(ax)
    ax.legend(loc="center left", fontsize=7.2)
    ps.panel_title(ax, "", "Correlation strength sweep",
                   "OOD accuracy vs how strongly the nuisance arrow is correlated in training")
    ps.save_figure(fig, out, "ext_fig1_corr_sweep")
    # table
    rows = []
    for s, corr in order:
        rows.append([
            f"{corr:.2f}",
            fmt(*get(scen[s], "sequence_erm", "iid_test_accuracy")[:2]),
            fmt(*get(scen[s], "sequence_erm", "ood_test_accuracy")[:2]),
            fmt(*get(scen[s], "sequence_erm", "ood_gap")[:2]),
            fmt(*get(scen[s], "nuisance_only_oracle", "ood_test_accuracy")[:2]),
            fmt(*get(scen[s], "core_only_oracle", "ood_test_accuracy")[:2]),
            fmt(*get(scen[s], "counterfactual_invariance", "ood_test_accuracy")[:2]),
        ])
    return md_table(
        ["corr", "ERM IID", "ERM OOD", "ERM gap", "Nuis-only OOD", "Core-only OOD", "CF OOD"], rows)


# ----------------------------------------------------------------------------- fig 2
def fig_nuisance_scale(summary: dict[str, Any], out: Path) -> str:
    order = [("scale_1p0", 1.0), ("scale_1p5", 1.5), ("scale_2p0", 2.0), ("scale_2p8", 2.8)]
    scen = summary["scenarios"]
    xs = [v for _, v in order]
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    erm_iid = [get(scen[s], "sequence_erm", "iid_test_accuracy")[0] for s, _ in order]
    erm_iid_s = [get(scen[s], "sequence_erm", "iid_test_accuracy")[1] for s, _ in order]
    erm_ood = [get(scen[s], "sequence_erm", "ood_test_accuracy")[0] for s, _ in order]
    erm_ood_s = [get(scen[s], "sequence_erm", "ood_test_accuracy")[1] for s, _ in order]
    core_ood = [get(scen[s], "core_only_oracle", "ood_test_accuracy")[0] for s, _ in order]
    nuis_ood = [get(scen[s], "nuisance_only_oracle", "ood_test_accuracy")[0] for s, _ in order]
    ax.errorbar(xs, erm_iid, yerr=erm_iid_s, marker="o", ms=4, lw=1.4, color=ps.SEQUENCE,
                capsize=2.5, label="Sequence ERM (IID)")
    ax.errorbar(xs, erm_ood, yerr=erm_ood_s, marker="s", ms=4, lw=1.4, color=ps.NUISANCE,
                capsize=2.5, label="Sequence ERM (OOD)")
    ax.plot(xs, core_ood, "--", marker="^", ms=4, lw=1.2, color=ps.CORE, label="Core-only reference (OOD)")
    ax.plot(xs, nuis_ood, ":", marker="v", ms=4, lw=1.2, color=ps.LIGHT_TEXT, label="Nuisance-only reference (OOD)")
    ax.axhline(0.5, color=ps.LIGHT_TEXT, lw=0.7, ls=":")
    ax.set_xlabel("Nuisance scale (signal strength)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(xs)
    ps.style_axis(ax)
    ax.legend(loc="center left", fontsize=7.2)
    ps.panel_title(ax, "", "Nuisance scale sweep",
                   "High IID + low OOD => shortcut selected; dropping IID => core occluded")
    ps.save_figure(fig, out, "ext_fig2_nuisance_scale")
    rows = []
    for s, v in order:
        rows.append([
            f"{v:.1f}",
            fmt(*get(scen[s], "sequence_erm", "iid_test_accuracy")[:2]),
            fmt(*get(scen[s], "sequence_erm", "ood_test_accuracy")[:2]),
            fmt(*get(scen[s], "sequence_erm", "ood_gap")[:2]),
            fmt(*get(scen[s], "core_only_oracle", "iid_test_accuracy")[:2]),
            fmt(*get(scen[s], "nuisance_only_oracle", "iid_test_accuracy")[:2]),
        ])
    return md_table(
        ["scale", "ERM IID", "ERM OOD", "ERM gap", "Core-only IID", "Nuis-only IID"], rows)


# ----------------------------------------------------------------------------- fig 3
ARCH_ORDER = [
    ("sequence_erm", "GRU"),
    ("sequence_erm_lstm", "LSTM"),
    ("sequence_erm_tcn", "TCN"),
    ("sequence_erm_transformer", "Transformer"),
    ("sequence_erm_temporal_pool", "CNN+pool"),
]


def fig_model_family(summary: dict[str, Any], out: Path) -> str:
    main = summary["scenarios"]["model_family_main"]
    nospur = summary["scenarios"].get("model_family_no_spurious", {})
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    labels = [lab for _, lab in ARCH_ORDER]
    x = np.arange(len(labels))
    w = 0.38
    iid = [get(main, m, "iid_test_accuracy")[0] for m, _ in ARCH_ORDER]
    iid_s = [get(main, m, "iid_test_accuracy")[1] for m, _ in ARCH_ORDER]
    ood = [get(main, m, "ood_test_accuracy")[0] for m, _ in ARCH_ORDER]
    ood_s = [get(main, m, "ood_test_accuracy")[1] for m, _ in ARCH_ORDER]
    ax.bar(x - w / 2, iid, w, yerr=iid_s, color=ps.SEQUENCE_LIGHT, edgecolor=ps.SEQUENCE,
           lw=0.8, capsize=2.5, label="IID (main)")
    ax.bar(x + w / 2, ood, w, yerr=ood_s, color=ps.NUISANCE_LIGHT, edgecolor=ps.NUISANCE,
           lw=0.8, capsize=2.5, label="OOD (main reversal)")
    # no-spurious OOD as markers
    if nospur:
        ns = [get(nospur, m, "ood_test_accuracy")[0] for m, _ in ARCH_ORDER]
        ax.scatter(x, ns, marker="D", s=22, color=ps.CORE, zorder=5, label="OOD (no-spurious)")
    ax.axhline(0.5, color=ps.LIGHT_TEXT, lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ps.style_axis(ax)
    ax.legend(loc="lower center", ncol=3, fontsize=7.2)
    ps.panel_title(ax, "", "Model-family comparison",
                   "OOD vulnerability across architectures; Transformer is less collapsed on average but highly variable")
    ps.save_figure(fig, out, "ext_fig3_model_family")
    rows = []
    for m, lab in ARCH_ORDER:
        rows.append([
            lab,
            fmt(*get(main, m, "iid_test_accuracy")[:2]),
            fmt(*get(main, m, "ood_test_accuracy")[:2]),
            fmt(*get(main, m, "ood_gap")[:2]),
            fmt(*get(nospur, m, "ood_test_accuracy")[:2]) if nospur else "--",
        ])
    return md_table(["Arch", "IID (main)", "OOD (main)", "gap", "OOD (no-spurious)"], rows)


# ----------------------------------------------------------------------------- fig 4
def fig_channel_mixing(summary: dict[str, Any], out: Path) -> str:
    scen = summary["scenarios"]
    layouts = [("two_channel", "Two-channel"), ("additive_single_channel", "Additive\n(1-channel)")]
    methods = [
        ("core_only_oracle", "Core-only reference", ps.CORE),
        ("sequence_erm", "Sequence ERM", ps.SEQUENCE),
        ("nuisance_only_oracle", "Nuisance-only reference", ps.NUISANCE),
        ("final_frame_mlp", "Final frame", ps.FINAL),
        ("counterfactual_invariance", "Counterfactual", ps.COUNTERFACTUAL),
    ]
    ps.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.0), sharey=True)
    for ax, (lay, lay_label) in zip(axes, layouts):
        s = scen.get(lay, {})
        x = np.arange(len(methods))
        w = 0.38
        iid = [get(s, m, "iid_test_accuracy")[0] for m, _, _ in methods]
        ood = [get(s, m, "ood_test_accuracy")[0] for m, _, _ in methods]
        ood_s = [get(s, m, "ood_test_accuracy")[1] for m, _, _ in methods]
        ax.bar(x - w / 2, iid, w, color="#Dfe4ea", edgecolor=ps.MUTED_TEXT, lw=0.7, label="IID")
        ax.bar(x + w / 2, ood, w, yerr=ood_s, color=[c for _, _, c in methods],
               edgecolor=ps.TEXT, lw=0.5, capsize=2, alpha=0.9, label="OOD")
        ax.axhline(0.5, color=ps.LIGHT_TEXT, lw=0.7, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab, _ in methods], rotation=40, ha="right", fontsize=7.0)
        ax.set_title(lay_label, fontsize=7.6)
        ax.set_ylim(0, 1.05)
        ps.style_axis(ax)
    axes[0].set_ylabel("Accuracy")
    axes[1].legend(loc="upper right", fontsize=7.2)
    fig.suptitle("Two-channel vs additive single-channel mixing", fontsize=8.2, fontweight="bold", y=1.02)
    ps.save_figure(fig, out, "ext_fig4_channel_mixing")
    rows = []
    for lay, lay_label in layouts:
        s = scen.get(lay, {})
        rows.append([
            lay_label.replace("\n", " "),
            fmt(*get(s, "sequence_erm", "iid_test_accuracy")[:2]),
            fmt(*get(s, "sequence_erm", "ood_test_accuracy")[:2]),
            fmt(*get(s, "sequence_erm", "ood_gap")[:2]),
            fmt(*get(s, "final_frame_mlp", "ood_test_accuracy")[:2]),
            fmt(*get(s, "counterfactual_invariance", "ood_test_accuracy")[:2]),
        ])
    return md_table(
        ["Layout", "ERM IID", "ERM OOD", "ERM gap", "Final-frame OOD", "CF OOD"], rows)


def fig_signature_across(summary: dict[str, Any], order: list[tuple[str, str]], out: Path,
                         fname: str, title: str, subtitle: str) -> str:
    """Grouped bars showing the IID-high / OOD-collapse signature across variants,
    with core-only (robust) and final-frame (chance) reference markers."""
    scen = summary["scenarios"]
    labels = [lab for _, lab in order]
    x = np.arange(len(labels))
    w = 0.38
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    iid = [get(scen[k], "sequence_erm", "iid_test_accuracy")[0] for k, _ in order]
    iid_s = [get(scen[k], "sequence_erm", "iid_test_accuracy")[1] for k, _ in order]
    ood = [get(scen[k], "sequence_erm", "ood_test_accuracy")[0] for k, _ in order]
    ood_s = [get(scen[k], "sequence_erm", "ood_test_accuracy")[1] for k, _ in order]
    ax.bar(x - w / 2, iid, w, yerr=iid_s, color=ps.SEQUENCE_LIGHT, edgecolor=ps.SEQUENCE,
           lw=0.8, capsize=2.5, label="Sequence ERM (IID)")
    ax.bar(x + w / 2, ood, w, yerr=ood_s, color=ps.NUISANCE_LIGHT, edgecolor=ps.NUISANCE,
           lw=0.8, capsize=2.5, label="Sequence ERM (OOD)")
    core = [get(scen[k], "core_only_oracle", "ood_test_accuracy")[0] for k, _ in order]
    fin = [get(scen[k], "final_frame_mlp", "ood_test_accuracy")[0] for k, _ in order]
    ax.scatter(x, core, marker="D", s=30, color=ps.CORE, zorder=5, label="Core-only reference (OOD)")
    ax.scatter(x, fin, marker="_", s=200, color=ps.FINAL, zorder=5, linewidths=2.2,
               label="Final-frame (OOD)")
    ax.axhline(0.5, color=ps.LIGHT_TEXT, lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.4)
    ax.set_ylabel("Accuracy", fontsize=9.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 1.08)
    ps.style_axis(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              fontsize=8.2, frameon=False, columnspacing=1.4,
              handletextpad=0.6)
    ps.save_figure(fig, out, fname)
    rows = []
    for k, lab in order:
        rows.append([
            lab,
            fmt(*get(scen[k], "sequence_erm", "iid_test_accuracy")[:2]),
            fmt(*get(scen[k], "sequence_erm", "ood_test_accuracy")[:2]),
            fmt(*get(scen[k], "core_only_oracle", "ood_test_accuracy")[:2]),
            fmt(*get(scen[k], "nuisance_only_oracle", "ood_test_accuracy")[:2]),
            fmt(*get(scen[k], "final_frame_mlp", "ood_test_accuracy")[:2]),
        ])
    return md_table(["Variant", "ERM IID", "ERM OOD", "Core OOD", "Nuis OOD", "Final OOD"], rows)


def fig_real_video(sum4k: dict[str, Any], sum8k: dict[str, Any], out: Path) -> str:
    """Two-panel mechanism signature for the real-video-arrow benchmark."""
    methods = [
        ("core_only_oracle", "Core-only\nref."),
        ("sequence_erm", "Sequence\nERM"),
        ("nuisance_only_oracle", "Nuisance-only\nref."),
        ("final_frame_mlp", "Final\nframe"),
        ("counterfactual_invariance", "Counter-\nfactual"),
    ]
    ps.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, summary, title in [(axes[0], sum4k, "4,096 train"), (axes[1], sum8k, "8,192 train")]:
        scen = summary["scenarios"]["rv_main"]
        x = np.arange(len(methods))
        w = 0.38
        iid = [get(scen, m, "iid_test_accuracy")[0] for m, _ in methods]
        iid_s = [get(scen, m, "iid_test_accuracy")[1] for m, _ in methods]
        ood = [get(scen, m, "ood_test_accuracy")[0] for m, _ in methods]
        ood_s = [get(scen, m, "ood_test_accuracy")[1] for m, _ in methods]
        ax.bar(x - w / 2, iid, w, yerr=iid_s, color=ps.SEQUENCE_LIGHT, edgecolor=ps.SEQUENCE,
               lw=0.8, capsize=2.5, label="IID")
        ax.bar(x + w / 2, ood, w, yerr=ood_s, color=ps.NUISANCE_LIGHT, edgecolor=ps.NUISANCE,
               lw=0.8, capsize=2.5, label="OOD (reversed)")
        ns = get(summary["scenarios"]["rv_no_spurious"], "sequence_erm", "ood_test_accuracy")[0]
        ax.axhline(ns, color=ps.CORE, lw=1.0, ls="--", label="ERM no-spurious OOD")
        ax.axhline(0.5, color=ps.LIGHT_TEXT, lw=0.7, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in methods], fontsize=7.2)
        ax.set_title(title, fontsize=7.8)
        ax.set_ylim(0, 1.08)
        ps.style_axis(ax)
    axes[0].set_ylabel("Accuracy")
    axes[1].legend(loc="upper right", fontsize=7.0)
    fig.suptitle("Real-video arrow as the nuisance", fontsize=8.4, fontweight="bold", y=1.03)
    ps.save_figure(fig, out, "ext_fig10_realvideo_results")
    rows = []
    for label, summary in [("4,096", sum4k), ("8,192", sum8k)]:
        scen = summary["scenarios"]["rv_main"]
        rows.append([
            label,
            fmt(*get(scen, "sequence_erm", "iid_test_accuracy")[:2]),
            fmt(*get(scen, "sequence_erm", "ood_test_accuracy")[:2]),
            fmt(*get(scen, "nuisance_only_oracle", "iid_test_accuracy")[:2]),
            fmt(*get(scen, "nuisance_only_oracle", "ood_test_accuracy")[:2]),
            fmt(*get(summary["scenarios"]["rv_no_spurious"], "sequence_erm", "ood_test_accuracy")[:2]),
            fmt(*get(scen, "counterfactual_invariance", "ood_test_accuracy")[:2]),
        ])
    return md_table(
        ["Train", "ERM IID", "ERM OOD", "Nuis IID", "Nuis OOD", "No-spur OOD", "CF OOD"], rows)


FAMILY_ORDER = [("fam_diffusion_translate", "Diff+Translate"),
                ("fam_diffusion_rotate", "Diff+Rotate"),
                ("fam_diffusion_diagonal", "Diff+Diagonal"),
                ("fam_advection_translate", "Advect+Translate")]
COMPLEXITY_ORDER = [("cx_grid32", "32x32"),
                    ("cx_clutter", "Clutter"),
                    ("cx_grid32_clutter", "32x32+Clutter+L10")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/extended")
    parser.add_argument("--out", default="results/extended/analysis")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    blocks = ["# Extended Ablation Results\n"]
    t1 = fig_corr_sweep(load_summary(root, "corr_sweep"), out)
    blocks += ["## 1. Nuisance-label correlation sweep\n", t1, ""]
    t2 = fig_nuisance_scale(load_summary(root, "nuisance_scale_sweep"), out)
    blocks += ["## 2. Nuisance scale sweep\n", t2, ""]
    t3 = fig_model_family(load_summary(root, "model_family"), out)
    blocks += ["## 3. Model-family comparison\n", t3, ""]
    t4 = fig_channel_mixing(load_summary(root, "channel_mixing"), out)
    blocks += ["## 4. Channel-mixing comparison\n", t4, ""]

    if (root / "family" / "summary.json").exists():
        t5 = fig_signature_across(
            load_summary(root, "family"), FAMILY_ORDER, out, "ext_fig5_family",
            "Benchmark-family generality",
            "The IID-high / OOD-collapse signature holds across distinct core/nuisance pairings")
        blocks += ["## 5. Benchmark-family generality\n", t5, ""]
    if (root / "complexity" / "summary.json").exists():
        t6 = fig_signature_across(
            load_summary(root, "complexity"), COMPLEXITY_ORDER, out, "ext_fig6_complexity",
            "Complexity scale-up and gate boundary",
            "OOD collapse persists, but the no-spurious gate limits shortcut attribution")
        blocks += ["## 6. Complexity scale-up\n", t6, ""]

    if (root / "real_video_4k" / "summary.json").exists() and (root / "real_video_8k" / "summary.json").exists():
        t7 = fig_real_video(load_summary(root, "real_video_4k"), load_summary(root, "real_video_8k"), out)
        blocks += ["## 7. Real-video arrow nuisance\n", t7, ""]

    (out / "extended_results.md").write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print("wrote", out / "extended_results.md")
    print("figures in", out / "figures")


if __name__ == "__main__":
    main()
