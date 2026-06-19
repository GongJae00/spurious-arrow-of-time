from pathlib import Path

from src.data.irreversible_source_inference import IrreversibleSourceConfig
from src.eval.benchmark_diagnostics import run_diagnostics


def test_diagnostics_write_required_artifacts(tmp_path: Path) -> None:
    cfg = IrreversibleSourceConfig(
        grid_size=12,
        length=6,
        n_train=96,
        n_val_iid=48,
        n_iid_test=48,
        n_ood_test=48,
        seed=7,
        diffusion_start_step=5,
        diffusion_steps_between_frames=3,
    )
    payload = run_diagnostics(cfg, tmp_path)
    assert (tmp_path / "diagnostics.json").exists()
    assert (tmp_path / "benchmark_gate.json").exists()
    assert (tmp_path / "candidate_trials.jsonl").exists()
    assert (tmp_path / "smoke_report.md").exists()
    metrics = payload["metrics"]
    for key in [
        "final_frame_core_oracle_accuracy",
        "full_sequence_core_oracle_accuracy",
        "core_only_ood_accuracy",
        "nuisance_only_iid_accuracy",
        "nuisance_only_ood_accuracy",
        "mixed_feature_probe_iid_accuracy",
        "mixed_feature_probe_ood_accuracy",
        "mixed_feature_probe_ood_gap",
        "static_feature_accuracy",
        "corr_y_nuisance_arrow_train",
        "corr_y_nuisance_arrow_ood_test",
    ]:
        assert key in metrics
    assert "checks" in payload["gate"]
