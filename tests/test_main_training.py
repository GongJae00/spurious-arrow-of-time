from pathlib import Path

import yaml

from src.train.main_experiment import run_experiment, stable_run_seed


def minimal_config() -> dict:
    return {
        "device": "cpu",
        "data": {
            "grid_size": 8,
            "length": 5,
            "n_train": 64,
            "n_val_iid": 32,
            "n_iid_test": 32,
            "n_ood_test": 32,
            "diffusion_start_step": 2,
            "diffusion_steps_between_frames": 2,
            "nuisance_scale": 2.0,
            "nuisance_correlation": 0.95,
            "ood_mode": "reversed",
        },
        "model": {"hidden_dim": 16, "num_layers": 1, "dropout": 0.0},
        "training": {
            "batch_size": 32,
            "epochs": 1,
            "patience": 1,
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "lambda_cf_task": 1.0,
            "lambda_pred": 0.1,
        },
        "methods": ["sequence_erm"],
        "profiles": {
            "smoke": {
                "runtime_limited": True,
                "seeds": [0],
                "methods": ["sequence_erm", "counterfactual_invariance"],
            }
        },
    }


def test_main_training_smoke_writes_outputs(tmp_path: Path) -> None:
    config = minimal_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_experiment(config_path, tmp_path / "out", "smoke", write_docs_summary=False)

    assert (tmp_path / "out" / "metrics.jsonl").exists()
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
    assert "sequence_erm" in result["summary"]["methods"]
    assert "counterfactual_invariance" in result["summary"]["methods"]


def test_runtime_limited_profile_does_not_update_docs_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(minimal_config()), encoding="utf-8")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    docs_summary = docs_dir / "latest_result_summary.md"
    docs_summary.write_text("keep full result summary\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    run_experiment(config_path, tmp_path / "out", "smoke", write_docs_summary=True)

    assert docs_summary.read_text(encoding="utf-8") == "keep full result summary\n"


def test_stable_run_seed_is_method_and_scenario_specific() -> None:
    seed = stable_run_seed(3, "main_spurious_arrow", "sequence_erm")
    assert seed == stable_run_seed(3, "main_spurious_arrow", "sequence_erm")
    assert seed != stable_run_seed(4, "main_spurious_arrow", "sequence_erm")
    assert seed != stable_run_seed(3, "ood_randomized", "sequence_erm")
    assert seed != stable_run_seed(3, "main_spurious_arrow", "counterfactual_invariance")
