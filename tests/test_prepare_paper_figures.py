from pathlib import Path

import matplotlib.image as mpimg

from src.eval.audit_smoke import EXPECTED_BENCHMARKS, REQUIRED_METHODS
from src.eval.prepare_paper_figures import prepare_paper_figures
from src.train.common import save_json


def _method_metrics(method: str, value: float) -> dict:
    if method in {"ocp_style", "lens_like_arrow_classifier"}:
        return {
            "n_runs": 2,
            "fine_tuned_encoder_iid_test_accuracy_mean": value,
            "fine_tuned_encoder_iid_test_accuracy_std": 0.01,
            "fine_tuned_encoder_ood_test_accuracy_mean": value - 0.1,
            "fine_tuned_encoder_ood_test_accuracy_std": 0.02,
            "fine_tuned_encoder_ood_gap_mean": 0.1,
            "fine_tuned_encoder_ood_gap_std": 0.02,
        }
    return {
        "n_runs": 2,
        "iid_test_accuracy_mean": value,
        "iid_test_accuracy_std": 0.01,
        "ood_test_accuracy_mean": value - 0.1,
        "ood_test_accuracy_std": 0.02,
        "ood_gap_mean": 0.1,
        "ood_gap_std": 0.02,
    }


def _write_result_root(root: Path) -> None:
    for bench_idx, benchmark in enumerate(EXPECTED_BENCHMARKS):
        aggregate = {
            method: _method_metrics(method, 0.8 - 0.01 * i - 0.02 * bench_idx)
            for i, method in enumerate(REQUIRED_METHODS)
        }
        save_json(root / benchmark / "aggregate.json", aggregate)
    save_json(
        root / "result_interpretation.json",
        {
            "claim_mode": "diagnostic_or_negative_evidence",
            "primary_method": "itm",
            "benchmark_gates": [
                {
                    "benchmark": benchmark,
                    "passed": False,
                }
                for benchmark in EXPECTED_BENCHMARKS
            ],
        },
    )


def _write_ink_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "seed: 3",
                "benchmark_name: ink_advection_diffusion",
                "data:",
                "  length: 6",
                "  grid_size: 12",
                "  core_diffusion: 0.16",
                "  spurious_diffusion: 0.12",
                "  pre_observation_steps: 2",
                "observation:",
                "  noise_std: 0.003",
                "  core_scale: 1.0",
                "  spur_scale: 0.9",
                "counterfactual:",
                "  spurious_cf_mode: randomized",
                "  reuse_noise: true",
            ]
        ),
        encoding="utf-8",
    )


def test_prepare_paper_figures_writes_manifest_and_assets(tmp_path):
    result_root = tmp_path / "results"
    output_dir = tmp_path / "visuals"
    ink_config = tmp_path / "ink.yaml"
    _write_result_root(result_root)
    _write_ink_config(ink_config)

    manifest = prepare_paper_figures(
        result_root=result_root,
        output_dir=output_dir,
        ink_config=ink_config,
        sample_n=12,
    )

    assert manifest["figures"] == [
        "mechanism_schematic.png",
        "ink_advection_diffusion_decomposition.png",
        "results_summary.png",
    ]
    for figure in manifest["figures"]:
        png = output_dir / figure
        pdf = png.with_suffix(".pdf")
        source = png.with_suffix(".json")
        assert png.exists()
        assert png.stat().st_size > 0
        image = mpimg.imread(png)
        assert image.shape[0] >= 1000
        assert image.shape[1] >= 1000
        assert pdf.exists()
        assert pdf.stat().st_size > 0
        assert source.exists()
    assert (output_dir / "paper_figures_manifest.json").exists()
    assert not any("contact_sheet" in figure for figure in manifest["figures"])
    assert not any("v1" in figure.lower() for figure in manifest["figures"])
