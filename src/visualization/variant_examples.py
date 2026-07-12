"""Render example sequences for the extended benchmark variants.

Produces two manuscript figures showing what the new data actually looks like:
  ext_fig7_family_examples   - mixed observations for the four core/nuisance pairings
  ext_fig8_complexity_examples - mixed observations for the complexity scale-ups

Each panel renders the two-channel mixed observation as a composite image
(core channel in teal, nuisance channel in orange), matching the visual
language of the original benchmark-construction figure.

Usage:
  python -m src.visualization.variant_examples --out results/extended/analysis
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.irreversible_source_inference import IrreversibleSourceConfig, generate_split
from src.visualization import paper_style as ps

TEAL = np.array([0.10, 0.55, 0.51])
ORANGE = np.array([0.85, 0.44, 0.25])


def composite(mixed: np.ndarray) -> np.ndarray:
    """Two-channel [L, 2, H, W] -> RGB [L, H, W, 3] composite on black."""
    core = mixed[:, 0]
    nuis = mixed[:, 1]
    core = np.clip(core / max(core.max(), 1e-6), 0, 1)
    nuis = np.clip(nuis / max(nuis.max(), 1e-6), 0, 1)
    rgb = core[..., None] * TEAL[None, None, None, :] + nuis[..., None] * ORANGE[None, None, None, :]
    return np.clip(rgb, 0, 1)


# Match the experimental base configuration from irreversible_source_extended.yaml
BASE = dict(
    grid_size=16, length=8,
    diffusion_alpha=0.22, diffusion_start_step=0, diffusion_steps_between_frames=4,
    core_noise_std=0.006, core_noise_growth_power=1.0,
    observation_noise_std=0.04, core_scale=1.0, nuisance_scale=1.2,
    nuisance_sigma=1.15, nuisance_speed=2.0, nuisance_trail_decay=0.78,
    nuisance_correlation=0.97,
)


def example_sequence(overrides: dict, seed: int = 11) -> np.ndarray:
    params = {**BASE, **overrides}
    cfg = IrreversibleSourceConfig(
        n_train=4, n_val_iid=4, n_iid_test=4, n_ood_test=4, seed=seed,
        benchmark_variant="endpoint_matched", observation_layout="two_channel",
        **params,
    )
    split = generate_split(cfg, "train")
    return composite(np.asarray(split.mixed[0]))


def render_grid(rows: list[tuple[str, dict]], out: Path, fname: str, title: str) -> None:
    ps.apply_style()
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    seqs = [(label, example_sequence(ov)) for label, ov in rows]
    max_len = max(s.shape[0] for _, s in seqs)
    fig, axes = plt.subplots(
        len(seqs), max_len,
        figsize=(0.62 * max_len + 1.5, 0.72 * len(seqs) + 0.5),
        squeeze=False,
    )
    for r, (label, seq) in enumerate(seqs):
        for t in range(max_len):
            ax = axes[r][t]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if t < seq.shape[0]:
                ax.imshow(seq[t], interpolation="nearest")
                if r == 0:
                    ax.set_title(f"t={t}", fontsize=6.4, color=ps.MUTED_TEXT, pad=2)
            else:
                ax.set_visible(False)
        axes[r][0].set_ylabel(label, fontsize=7.2, rotation=0, ha="right", va="center", labelpad=6)
    fig.suptitle(title, fontsize=8.4, fontweight="bold", y=1.0)
    fig.subplots_adjust(wspace=0.06, hspace=0.12, left=0.16, right=0.99, top=0.86, bottom=0.02)
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{fname}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(figure_dir / f"{fname}.png", bbox_inches="tight", pad_inches=0.03, dpi=400)
    plt.close(fig)


def render_real_video_examples(out: Path, cache: str) -> None:
    """Rows: real crop forward, same crop backward, core-only, mixed composite."""
    ps.apply_style()
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    crops = np.load(cache)["crops"]
    # Pick a crop with high, spatially localized motion so the direction is visible.
    sample = crops[:4000].astype(np.float32) / 255.0
    motion = np.abs(np.diff(sample, axis=1)).mean(axis=(1, 2, 3))
    spatial = sample.std(axis=(1, 2, 3))
    crop = sample[int(np.argmax(motion * spatial))]
    cfg = IrreversibleSourceConfig(
        n_train=4, n_val_iid=4, n_iid_test=4, n_ood_test=4, seed=11,
        benchmark_variant="endpoint_matched", observation_layout="two_channel",
        nuisance_motion="real_video", real_video_cache=cache, **{
            k: v for k, v in BASE.items()},
    )
    split = generate_split(cfg, "train")
    mixed_rgb = composite(np.asarray(split.mixed[0]))
    core = np.asarray(split.core_only[0])
    core = np.clip(core / max(core.max(), 1e-6), 0, 1)
    L = crop.shape[0]
    rows = [
        ("실영상 정재생 (d=+1)", [np.repeat(crop[t][..., None], 3, -1) * ORANGE for t in range(L)]),
        ("실영상 역재생 (d=-1)", [np.repeat(crop[L - 1 - t][..., None], 3, -1) * ORANGE for t in range(L)]),
        ("core 확산 (라벨 결정)", [np.repeat(core[t][..., None], 3, -1) * TEAL for t in range(L)]),
        ("혼합 관측", [mixed_rgb[t] for t in range(L)]),
    ]
    fig, axes = plt.subplots(len(rows), L, figsize=(0.62 * L + 1.9, 0.72 * len(rows) + 0.5), squeeze=False)
    for r, (label, imgs) in enumerate(rows):
        for t in range(L):
            ax = axes[r][t]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.imshow(np.clip(imgs[t], 0, 1), interpolation="nearest")
            if r == 0:
                ax.set_title(f"t={t}", fontsize=6.4, color=ps.MUTED_TEXT, pad=2)
        axes[r][0].set_ylabel(label, fontsize=7.0, rotation=0, ha="right", va="center", labelpad=6)
    fig.suptitle("실영상 시간의 화살을 nuisance로 사용한 벤치마크 예시", fontsize=8.4, fontweight="bold", y=1.0)
    fig.subplots_adjust(wspace=0.06, hspace=0.12, left=0.22, right=0.99, top=0.85, bottom=0.02)
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "ext_fig9_realvideo_examples.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(figure_dir / "ext_fig9_realvideo_examples.png", bbox_inches="tight", pad_inches=0.03, dpi=400)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/extended/analysis")
    args = parser.parse_args()
    out = Path(args.out)

    render_grid(
        [
            ("확산+이동", dict(core_process="diffusion", nuisance_motion="translate")),
            ("확산+회전", dict(core_process="diffusion", nuisance_motion="rotate")),
            ("확산+대각", dict(core_process="diffusion", nuisance_motion="diagonal")),
            ("이류+이동", dict(core_process="advection", advection_shift=1, nuisance_motion="translate")),
        ],
        out, "ext_fig7_family_examples",
        "벤치마크 패밀리 예시: 혼합 관측 (청록=core, 주황=nuisance)",
    )
    render_grid(
        [
            ("32×32", dict(grid_size=32, observation_noise_std=0.06)),
            ("클러터", dict(background_clutter=0.6, background_clutter_count=4, observation_noise_std=0.06)),
            ("32×32+클러터+L10", dict(grid_size=32, length=10, background_clutter=0.6,
                                      background_clutter_count=5, observation_noise_std=0.06)),
        ],
        out, "ext_fig8_complexity_examples",
        "복잡도 확대 예시: 혼합 관측 (청록=core, 주황=nuisance)",
    )
    rv_cache = Path("data/real_video/cache_g16_L8_s5.npz")
    if rv_cache.exists():
        render_real_video_examples(out, str(rv_cache))
        print("wrote", out / "figures" / "ext_fig9_realvideo_examples.png")
    print("wrote", out / "figures" / "ext_fig7_family_examples.png")
    print("wrote", out / "figures" / "ext_fig8_complexity_examples.png")


if __name__ == "__main__":
    main()
