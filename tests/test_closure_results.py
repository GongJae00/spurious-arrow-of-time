import json

from src.eval.interpret_closure_results import SCHEMA
from src.eval.interpret_closure_results import interpret_closure_results


def _method_payload(iid: float, ood: float, gap: float) -> dict:
    return {
        "n_runs": 2,
        "iid_test_accuracy_mean": iid,
        "ood_test_accuracy_mean": ood,
        "ood_gap_mean": gap,
    }


def _aggregate() -> dict:
    return {
        "by_condition": {
            "closure_spurious_causality/correlated_reversed_ood": {
                "erm": _method_payload(0.98, 0.20, 0.78),
                "sib": _method_payload(0.96, 0.88, 0.08),
                "sid": _method_payload(0.97, 0.92, 0.05),
            },
            "closure_spurious_causality/correlated_no_shift": {
                "erm": _method_payload(0.98, 0.97, 0.01),
                "sib": _method_payload(0.96, 0.95, 0.01),
                "sid": _method_payload(0.97, 0.96, 0.01),
            },
            "closure_spurious_causality/randomized_no_shortcut": {
                "erm": _method_payload(0.80, 0.79, 0.01),
                "sib": _method_payload(0.80, 0.78, 0.02),
                "sid": _method_payload(0.81, 0.79, 0.02),
            },
        }
    }


def test_interpret_closure_results_marks_diagnostic_gate(tmp_path):
    for benchmark in ("sta", "ink_advection_diffusion"):
        bench_dir = tmp_path / benchmark
        bench_dir.mkdir(parents=True)
        (bench_dir / "aggregate.json").write_text(json.dumps(_aggregate()))
    (tmp_path / "sid_conditional_factor_audit_summary.json").write_text(
        json.dumps(
            {
                "schema": "sid_conditional_factor_audit_v1",
                "passed": True,
                "aggregate": {
                    "ood_test.z_ir_spur.spurious_dynamic.raw_orientation_free_auc": {
                        "mean": 0.95,
                        "n": 4,
                    }
                },
            }
        )
    )

    report = interpret_closure_results(tmp_path)

    assert report["schema"] == SCHEMA
    assert report["passed"] is True
    assert report["claim_mode"] == "diagnostic_closure_evidence"
    assert report["appendix_diagnostic_benchmarks"] == []
    assert len(report["rows"]) == 18
    assert report["closure_gate"]["passed"] is True
    assert "clean SID factorization" in report["interpretation_lock"]
    assert report["conditional_factor_audit"]["selected_metrics"]
    assert (tmp_path / "closure_result_interpretation.json").exists()
    assert (tmp_path / "closure_result_interpretation.md").exists()


def test_interpret_closure_results_rejects_weak_trap_control_contrast(tmp_path):
    weak = _aggregate()
    weak["by_condition"]["closure_spurious_causality/correlated_no_shift"]["erm"] = (
        _method_payload(0.98, 0.30, 0.68)
    )
    for benchmark in ("sta", "ink_advection_diffusion"):
        bench_dir = tmp_path / benchmark
        bench_dir.mkdir(parents=True)
        (bench_dir / "aggregate.json").write_text(json.dumps(weak))

    report = interpret_closure_results(tmp_path)

    assert report["passed"] is False
    assert report["closure_gate"]["passed"] is False
