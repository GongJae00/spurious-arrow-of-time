import json
from pathlib import Path

import matplotlib.image as mpimg

from src.eval.aggregate_results import aggregate_results
from src.eval.ink_advection_diffusion_diagnostics import diagnose_config as diagnose_ink
from src.eval.interpret_results import interpret_results
from src.eval.plot_results import plot_ood_bars
from src.eval.sta_benchmark_diagnostics import diagnose_config as diagnose_sta
from src.train.common import load_config


def test_aggregate_results_reads_manifested_success_runs(tmp_path):
    run = tmp_path / "sta" / "erm" / "run_seed0"
    run.mkdir(parents=True)
    (run / "final_metrics.json").write_text(
        json.dumps({"iid_test_accuracy": 0.8, "ood_test_accuracy": 0.4, "status_code": 1})
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"runs": [{"method": "erm", "status": "success", "run_dir": str(run)}]})
    )

    summary = aggregate_results(tmp_path)

    assert summary["erm"]["n_runs"] == 1
    assert summary["erm"]["iid_test_accuracy_mean"] == 0.8
    assert summary["erm"]["ood_test_accuracy_mean"] == 0.4


def test_sta_diagnostics_reports_required_quality_gates():
    result = diagnose_sta(load_config("configs/sta_smoke.yaml"))

    required = {
        "same_mixing_matrix",
        "train_threshold_reused",
        "core_oracle_high_iid",
        "core_oracle_high_ood",
        "spurious_rule_high_iid",
        "spurious_rule_breaks_ood",
    }
    assert required.issubset(result["quality_gates"])
    assert all(isinstance(result["quality_gates"][name], bool) for name in required)


def test_ink_advection_diagnostics_reports_physical_and_task_gates():
    result = diagnose_ink(load_config("configs/ink_advection_diffusion_smoke.yaml"))

    required = {
        "mass_conservation",
        "nonnegative_concentration",
        "spread_increase",
        "entropy_increase",
        "visible_signal",
        "core_oracle_high_iid",
        "core_oracle_high_ood",
        "spurious_rule_high_iid",
        "spurious_rule_breaks_ood",
        "counterfactual_preserves_core_and_label",
        "counterfactual_changes_spurious_flow",
    }
    assert required.issubset(result["quality_gates"])
    assert all(isinstance(result["quality_gates"][name], bool) for name in required)


def _write_minimal_result_root(root: Path) -> None:
    for benchmark in ("sta", "ink_advection_diffusion"):
        bench = root / benchmark
        bench.mkdir(parents=True)
        (bench / "aggregate.json").write_text(
            json.dumps(
                {
                    "erm": {
                        "n_runs": 1,
                        "iid_test_accuracy_mean": 0.8,
                        "ood_test_accuracy_mean": 0.4,
                    },
                    "sid": {
                        "n_runs": 1,
                        "iid_test_accuracy_mean": 0.7,
                        "ood_test_accuracy_mean": 0.45,
                    },
                    "itm": {
                        "n_runs": 1,
                        "iid_test_accuracy_mean": 0.7,
                        "ood_test_accuracy_mean": 0.45,
                    },
                }
            )
        )
    (root / "evidence_audit.json").write_text(
        json.dumps({"passed": True, "n_failed": 0, "n_checks": 12, "failed_checks": []})
    )


def test_interpret_results_blocks_positive_claim_without_full_audit(tmp_path):
    _write_minimal_result_root(tmp_path)

    interpretation = interpret_results(tmp_path, min_seeds=1, min_epochs=1)

    assert interpretation["evidence_audit_passed"] is False
    assert interpretation["positive_primary_claim_allowed"] is False
    assert interpretation["claim_mode"] == "diagnostic_or_negative_evidence"


def test_plot_ood_bars_writes_png_pdf_and_source(tmp_path):
    aggregate = tmp_path / "aggregate.json"
    output = tmp_path / "figures" / "ood.png"
    aggregate.write_text(
        json.dumps(
            {
                "erm": {
                    "iid_test_accuracy_mean": 0.9,
                    "ood_test_accuracy_mean": 0.4,
                },
                "ocp_style": {
                    "fine_tuned_encoder_iid_test_accuracy_mean": 0.8,
                    "fine_tuned_encoder_ood_test_accuracy_mean": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )

    plot_ood_bars(aggregate, output)

    assert output.exists()
    assert output.stat().st_size > 0
    image = mpimg.imread(output)
    assert image.shape[0] >= 1000
    assert image.shape[1] >= 1000
    assert output.with_suffix(".pdf").exists()
    assert output.with_suffix(".json").exists()
