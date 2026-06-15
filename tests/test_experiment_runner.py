import json
from pathlib import Path

import pytest

from src.experiments import run_experiment as runner
from src.train.common import load_config


def _base_config():
    return {
        "seed": 0,
        "experiment": "spurious_arrow_trap",
        "data": {
            "p_core": 0.35,
            "q_core": 0.25,
            "p_spur": 0.45,
            "q_spur": 0.15,
        },
    }


def test_ep_ratio_expansion_logs_actual_solution_metadata():
    config = {
        **_base_config(),
        "experiment": "ep_ratio_sweep",
        "ratios": [0.5, 2.0],
        "move_rate": 0.6,
    }
    expanded = runner.expand_configs(config)

    assert [name for name, _ in expanded] == ["ratio_0.5", "ratio_2.0"]
    for _, cfg in expanded:
        assert "actual_ratio" in cfg["sweep"]
        assert "actual_ep" in cfg["sweep"]["solution"]
        assert cfg["data"]["p_spur"] > 0.0
        assert cfg["data"]["q_spur"] > 0.0


def test_closure_spurious_causality_expansion_marks_conditions():
    config = {
        **_base_config(),
        "experiment": "closure_spurious_causality",
        "benchmark_name": "sta_bench",
    }
    expanded = runner.expand_configs(config)

    assert [name for name, _ in expanded] == [
        "correlated_reversed_ood",
        "correlated_no_shift",
        "randomized_no_shortcut",
    ]
    roles = {name: cfg["closure_condition"]["role"] for name, cfg in expanded}
    assert roles["correlated_reversed_ood"] == "main_spurious_arrow_trap"
    assert roles["correlated_no_shift"] == "no_distribution_shift_control"
    assert roles["randomized_no_shortcut"] == "no_spurious_arrow_control"


def test_clean_output_dir_only_allows_scoped_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    safe = tmp_path / "results" / "smoke_run"
    safe.mkdir(parents=True)
    (safe / "old.json").write_text("{}")

    runner.clean_output_dir(safe)
    assert not safe.exists()

    with pytest.raises(ValueError):
        runner.clean_output_dir(tmp_path)
    with pytest.raises(ValueError):
        runner.clean_output_dir(tmp_path / "results")


def test_scripts_reference_only_public_benchmarks():
    script_paths = [
        Path("experiments/smoke_suite.sh"),
        Path("experiments/full_suite.sh"),
        Path("experiments/preflight.sh"),
        Path("experiments/21_closure_spurious_causality_suite.sh"),
    ]
    combined = "\n".join(path.read_text() for path in script_paths)

    assert "configs/sta_" in combined
    assert "configs/ink_advection_diffusion_" in combined
    assert "closure_ink_advection_diffusion" in combined
    assert "main_submission" not in combined


def test_final_configs_load_without_retired_benchmarks():
    config_paths = [
        "configs/sta_smoke.yaml",
        "configs/ink_advection_diffusion_smoke.yaml",
        "configs/closure_sta.yaml",
        "configs/closure_ink_advection_diffusion.yaml",
    ]
    for path in config_paths:
        config = load_config(path)
        assert config.get("benchmark_name", config.get("data", {}).get("benchmark_name")) in {
            None,
            "sta_bench",
            "ink_advection_diffusion",
        }


def test_run_manifest_records_success_and_failure_rows(tmp_path):
    manifest = {
        "runs": [
            {"method": "erm", "seed": 0, "status": "success", "run_dir": "r0"},
            {"method": "sib", "seed": 1, "status": "failed", "run_dir": "r1"},
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    loaded = json.loads(path.read_text())
    assert [row["status"] for row in loaded["runs"]] == ["success", "failed"]
