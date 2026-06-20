import json
from pathlib import Path

from src.eval.main_results import (
    final_gate_checks,
    load_jsonl,
    plot_scenario_heatmap,
    write_scenario_table,
)


def test_scenario_tables_and_heatmaps_are_generated(tmp_path: Path) -> None:
    rows = [
        {
            "scenario": "main_spurious_arrow",
            "method": "sequence_erm",
            "iid_test_accuracy": 0.95,
            "ood_test_accuracy": 0.05,
            "ood_gap": 0.90,
        },
        {
            "scenario": "main_spurious_arrow",
            "method": "core_only_oracle",
            "iid_test_accuracy": 0.99,
            "ood_test_accuracy": 0.98,
            "ood_gap": 0.01,
        },
        {
            "scenario": "ood_randomized",
            "method": "sequence_erm",
            "iid_test_accuracy": 0.95,
            "ood_test_accuracy": 0.50,
            "ood_gap": 0.45,
        },
        {
            "scenario": "ood_randomized",
            "method": "core_only_oracle",
            "iid_test_accuracy": 0.99,
            "ood_test_accuracy": 0.98,
            "ood_gap": 0.01,
        },
    ]
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    loaded = load_jsonl(metrics_path)

    table_path = tmp_path / "scenario_table.md"
    heatmap_path = tmp_path / "scenario_ood_gap_heatmap.png"
    write_scenario_table(loaded, table_path)
    plot_scenario_heatmap(
        loaded,
        metric="ood_gap",
        path=heatmap_path,
        title="OOD gap across scenarios",
        colorbar_label="OOD gap",
    )

    assert "ood randomized" in table_path.read_text(encoding="utf-8")
    assert heatmap_path.exists()
    assert heatmap_path.stat().st_size > 0


def test_final_gate_checks_require_no_spurious_and_main_controls() -> None:
    rows = [
        {
            "scenario": "no_spurious_correlation",
            "method": "sequence_erm",
            "iid_test_accuracy": 0.86,
            "ood_test_accuracy": 0.84,
            "ood_gap": 0.02,
        },
        {
            "scenario": "main_spurious_arrow",
            "method": "sequence_erm",
            "iid_test_accuracy": 0.91,
            "ood_test_accuracy": 0.40,
            "ood_gap": 0.51,
            "dataset_config": {"benchmark_variant": "endpoint_matched"},
        },
        {
            "scenario": "main_spurious_arrow",
            "method": "nuisance_only_oracle",
            "iid_test_accuracy": 0.95,
            "ood_test_accuracy": 0.05,
            "ood_gap": 0.90,
            "dataset_config": {"benchmark_variant": "endpoint_matched"},
        },
        {
            "scenario": "main_spurious_arrow",
            "method": "core_only_oracle",
            "iid_test_accuracy": 0.98,
            "ood_test_accuracy": 0.97,
            "ood_gap": 0.01,
            "dataset_config": {"benchmark_variant": "endpoint_matched"},
        },
        {
            "scenario": "main_spurious_arrow",
            "method": "final_frame_mlp",
            "iid_test_accuracy": 0.80,
            "ood_test_accuracy": 0.50,
            "ood_gap": 0.06,
            "dataset_config": {"benchmark_variant": "endpoint_matched"},
        },
    ]
    checks = final_gate_checks(rows)
    by_name = {check["name"]: check for check in checks}
    assert by_name["gate_a_no_spurious_iid"]["passed"]
    assert by_name["gate_a_no_spurious_seed_success_rate"]["passed"]
    assert by_name["gate_b_main_gap"]["passed"]
    assert by_name["gate_e_endpoint_final_gap"]["passed"]
