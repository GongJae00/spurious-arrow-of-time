from pathlib import Path

import yaml

from src.train.main_experiment import run_experiment


def test_main_training_smoke_writes_outputs(tmp_path: Path) -> None:
    config = {
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_experiment(config_path, tmp_path / "out", "smoke", write_docs_summary=False)

    assert (tmp_path / "out" / "metrics.jsonl").exists()
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
    assert "sequence_erm" in result["summary"]["methods"]
    assert "counterfactual_invariance" in result["summary"]["methods"]
