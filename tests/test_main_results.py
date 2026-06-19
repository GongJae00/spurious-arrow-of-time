import json
from pathlib import Path

from src.eval.main_results import (
    load_jsonl,
    plot_scenario_heatmap,
    write_scenario_table,
)


def test_scenario_tables_and_heatmaps_are_generated(tmp_path: Path) -> None:
    rows = [
        {
            "scenario": "main_reversed",
            "method": "sequence_erm",
            "iid_test_accuracy": 0.95,
            "ood_test_accuracy": 0.05,
            "ood_gap": 0.90,
        },
        {
            "scenario": "main_reversed",
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

    assert "ood_randomized" in table_path.read_text(encoding="utf-8")
    assert heatmap_path.exists()
    assert heatmap_path.stat().st_size > 0
