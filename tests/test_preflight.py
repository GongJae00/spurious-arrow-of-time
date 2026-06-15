import json
import os
from pathlib import Path
import subprocess

from src.eval.preflight import run_preflight
from src.eval.audit_smoke import REQUIRED_METHODS
from src.eval.validate_preflight_reuse import validate_preflight_reuse


def _failed_checks(report: dict) -> set[str]:
    return {check["name"] for check in report["checks"] if not check["passed"]}


def test_preflight_passes_cpu_noncanonical_path(tmp_path):
    report = run_preflight(
        out=str(tmp_path / "full_run"),
        device="cpu",
        seeds="0 1",
        min_seeds=2,
        min_final_seeds_required=2,
        epochs=3,
        min_epochs_required=3,
        finalize_paper="0",
        min_disk_free_gb=0.0,
        require_canonical_paths=False,
    )

    assert report["launch_recommended"] is True
    assert report["n_errors_failed"] == 0
    assert report["environment"]["methods"] == list(REQUIRED_METHODS)
    assert report["environment"]["launch_authorization"] == "maintenance_or_diagnostic"


def test_preflight_rejects_smoke_low_seed_and_missing_method(tmp_path):
    methods = " ".join(method for method in REQUIRED_METHODS if method != "sid")
    report = run_preflight(
        out=str(tmp_path / "smoke_run"),
        device="cpu",
        seeds="0",
        min_seeds=1,
        min_final_seeds_required=2,
        epochs=1,
        min_epochs_required=3,
        methods=methods,
        finalize_paper="0",
        min_disk_free_gb=0.0,
        require_canonical_paths=False,
    )

    failed = _failed_checks(report)
    assert report["launch_recommended"] is False
    assert "out_not_smoke" in failed
    assert "final_seed_count" in failed
    assert "final_epoch_count" in failed
    assert "required_methods" in failed


def test_preflight_rejects_noncanonical_full_path():
    report = run_preflight(
        out="results/alternate_full_run",
        device="cpu",
        seeds="0 1 2 3 4",
        min_seeds=5,
        epochs=50,
        finalize_paper="0",
        min_disk_free_gb=0.0,
        require_canonical_paths=True,
    )

    assert "out_canonical" in _failed_checks(report)


def test_preflight_rejects_cpu_when_final_full_requires_cuda():
    report = run_preflight(
        out="results/full_run",
        device="cpu",
        seeds="0 1 2 3 4",
        min_seeds=5,
        epochs=50,
        finalize_paper="0",
        min_disk_free_gb=0.0,
        require_cuda_for_full_run="1",
    )

    failed = _failed_checks(report)
    assert report["launch_recommended"] is False
    assert "full_run_requires_cuda_capable_device" in failed
    assert any(
        check["name"] == "torch_cuda_available_for_full_suite"
        for check in report["checks"]
    )
    assert report["environment"]["require_cuda_for_full_run"] == "1"
    assert report["environment"]["launch_authorization"] == "final_cuda_launch"


def test_preflight_cli_writes_json(tmp_path):
    output = tmp_path / "preflight.json"
    completed = subprocess.run(
        [
            "python",
            "-m",
            "src.eval.preflight",
            "--output",
            str(output),
            "--out",
            str(tmp_path / "full_run"),
            "--device",
            "cpu",
            "--seeds",
            "0",
            "--min-seeds",
            "1",
            "--min-final-seeds-required",
            "1",
            "--epochs",
            "1",
            "--min-epochs-required",
            "1",
            "--min-disk-free-gb",
            "0",
            "--finalize-paper",
            "0",
            "--no-require-canonical-paths",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    payload = json.loads(output.read_text())
    assert payload["launch_recommended"] is True
    assert payload["environment"]["launch_authorization"] == "maintenance_or_diagnostic"


def test_preflight_cli_defaults_to_final_min_epoch_floor(tmp_path):
    output = tmp_path / "preflight.json"
    completed = subprocess.run(
        [
            "python",
            "-m",
            "src.eval.preflight",
            "--output",
            str(output),
            "--out",
            str(tmp_path / "full_run"),
            "--device",
            "cpu",
            "--seeds",
            "0",
            "--min-seeds",
            "1",
            "--min-final-seeds-required",
            "1",
            "--epochs",
            "24",
            "--min-disk-free-gb",
            "0",
            "--finalize-paper",
            "0",
            "--no-require-canonical-paths",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    payload = json.loads(output.read_text())
    assert payload["environment"]["min_epochs_required"] == 25
    assert "final_epoch_count" in _failed_checks(payload)


def test_final_evidence_clis_default_to_final_epoch_floor():
    sources = [
        Path("src/eval/preflight.py").read_text(),
        Path("src/eval/audit_evidence.py").read_text(),
        Path("src/eval/interpret_results.py").read_text(),
        Path("src/eval/prepare_paper_assets.py").read_text(),
    ]

    for source in sources:
        assert "default=10" not in source
        assert '"MIN_EPOCHS", 10' not in source
        assert "25" in source


def test_wait_wrapper_enforces_preflight_before_suite():
    script = Path("experiments/wait_for_preflight_and_run.sh").read_text()

    assert "preflight.sh" in script
    assert "full_suite.sh" in script
    assert "must not bypass preflight" in script
    assert 'REQUIRE_CUDA_FOR_FULL_RUN="${REQUIRE_CUDA_FOR_FULL_RUN:-1}"' in script
    assert "REQUIRE_CUDA_FOR_FULL_RUN must be 0 or 1" in script
    assert "export REQUIRE_CUDA_FOR_FULL_RUN" in script
    assert 'ATTEMPT_PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT}.attempt"' in script
    assert 'mv -- "${ATTEMPT_PREFLIGHT_OUTPUT}" "${PREFLIGHT_OUTPUT}"' in script
    assert "RUN_PREFLIGHT=0 REUSE_PASSED_PREFLIGHT=1" in script
    assert script.index("preflight.sh") < script.index(
        "full_suite.sh"
    )


def test_full_suite_runs_preflight_before_clean_and_audit():
    script = Path("experiments/full_suite.sh").read_text()

    assert 'RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"' in script
    assert 'REQUIRE_CUDA_FOR_FULL_RUN="${REQUIRE_CUDA_FOR_FULL_RUN:-1}"' in script
    assert "REQUIRE_CUDA_FOR_FULL_RUN must be 0 or 1" in script
    assert 'REUSE_PASSED_PREFLIGHT="${REUSE_PASSED_PREFLIGHT:-0}"' in script
    assert 'PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:-results/preflight.json}"' in script
    assert "RUN_PREFLIGHT=0 is diagnostic only" in script
    assert "src.eval.validate_preflight_reuse" in script
    assert "expected-require-cuda-for-full-run" in script
    assert "-u REQUIRE_CUDA_FOR_FULL_RUN" in script
    assert "-u RUN_PREFLIGHT" in script
    assert "-u REUSE_PASSED_PREFLIGHT" in script
    assert "-u FINALIZE_PAPER" in script
    assert "preflight.sh" in script
    assert "configs/sta_full.yaml" in script
    assert "configs/ink_advection_diffusion_full.yaml" in script
    assert "configs/ink_diffusion_full.yaml" not in script
    assert 'FINALIZE_PAPER="${FINALIZE_PAPER:-1}"' in script
    assert "ALLOW_UNFINALIZED_RUN" in script
    assert "finalize_paper.sh" in script
    assert "configs/sta_smoke.yaml" not in script
    assert "configs/ink_diffusion_smoke.yaml" not in script
    assert "configs/ink_advection_diffusion_smoke.yaml" not in script
    assert "--preflight-path" in script
    assert 'cp -- "${PREFLIGHT_OUTPUT}" "${OUT}/preflight.json"' in script
    assert '--preflight-path "${OUT}/preflight.json"' in script
    assert 'PREFLIGHT_OUTPUT="${OUT}/preflight.json"' in script
    assert script.index("preflight.sh") < script.index(
        'if [[ "${CLEAN_OUT}" == "1" ]]'
    )
    assert script.index("REUSE_PASSED_PREFLIGHT") < script.index(
        'if [[ "${CLEAN_OUT}" == "1" ]]'
    )
    assert script.index("preflight.sh") < script.index(
        "src.eval.audit_evidence"
    )
    assert "src.eval.itm_mechanism_audit" in script
    assert script.index("src.eval.itm_mechanism_audit") < script.index(
        "src.eval.audit_evidence"
    )
    assert script.index("src.eval.interpret_results") < script.index(
        "finalize_paper.sh"
    )


def test_full_suite_rejects_reused_maintenance_preflight(tmp_path):
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "pass": True,
                "launch_recommended": True,
                "environment": {
                    "out": str(tmp_path / "full_run"),
                    "require_cuda_for_full_run": "0",
                    "launch_authorization": "maintenance_or_diagnostic",
                },
            }
        )
    )

    completed = subprocess.run(
        [
            "bash",
            "experiments/full_suite.sh",
        ],
        check=False,
        text=True,
        capture_output=True,
        env={
            "PATH": os.environ["PATH"],
            "PYENV_VERSION": os.environ.get("PYENV_VERSION", "rppg-310"),
            "PYTHON_BIN": "python",
            "OUT": str(tmp_path / "full_run"),
            "PREFLIGHT_OUTPUT": str(preflight),
            "RUN_PREFLIGHT": "0",
            "REUSE_PASSED_PREFLIGHT": "1",
            "REQUIRE_CUDA_FOR_FULL_RUN": "1",
            "CLEAN_OUT": "0",
            "RUN_TESTS": "0",
            "FINALIZE_PAPER": "0",
            "ALLOW_UNFINALIZED_RUN": "1",
        },
    )

    assert completed.returncode == 2
    assert "Refusing to reuse preflight artifact" in completed.stderr
    assert "launch_authorization must be final_cuda_launch" in completed.stderr


def test_validate_preflight_reuse_rejects_mismatched_runtime_args(tmp_path):
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "pass": True,
                "launch_recommended": True,
                "environment": {
                    "out": str(tmp_path / "full_run"),
                    "device": "cuda",
                    "seeds": "0 1 2 3 4",
                    "min_seeds": 5,
                    "epochs": 50,
                    "min_epochs_required": 25,
                    "methods": list(REQUIRED_METHODS),
                    "require_cuda_for_full_run": "1",
                    "launch_authorization": "final_cuda_launch",
                },
            }
        )
    )

    report = validate_preflight_reuse(
        preflight,
        expected_out=str(tmp_path / "full_run"),
        expected_device="cuda",
        expected_seeds="0 1 2",
        expected_min_seeds=5,
        expected_epochs=50,
        expected_min_epochs=25,
        expected_methods=" ".join(REQUIRED_METHODS),
        expected_require_cuda_for_full_run="1",
    )

    assert report["pass"] is False
    assert any("preflight seeds=" in error for error in report["errors"])


def test_validate_preflight_reuse_accepts_matching_final_cuda_artifact(tmp_path):
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "pass": True,
                "launch_recommended": True,
                "environment": {
                    "out": str(tmp_path / "full_run"),
                    "device": "cuda",
                    "seeds": "0 1 2 3 4",
                    "min_seeds": 5,
                    "epochs": 50,
                    "min_epochs_required": 25,
                    "methods": list(REQUIRED_METHODS),
                    "require_cuda_for_full_run": "1",
                    "launch_authorization": "final_cuda_launch",
                },
            }
        )
    )

    report = validate_preflight_reuse(
        preflight,
        expected_out=str(tmp_path / "full_run"),
        expected_device="cuda",
        expected_seeds="0 1 2 3 4",
        expected_min_seeds=5,
        expected_epochs=50,
        expected_min_epochs=25,
        expected_methods=" ".join(REQUIRED_METHODS),
        expected_require_cuda_for_full_run="1",
    )

    assert report["pass"] is True
    assert report["errors"] == []


def test_preflight_uses_full_configs_not_smoke_configs():
    source = Path("src/eval/preflight.py").read_text()

    assert "configs/sta_full.yaml" in source
    assert "configs/ink_advection_diffusion_full.yaml" in source
    assert "configs/ink_diffusion_full.yaml" not in source
    assert "configs/sta_smoke.yaml" not in source
    assert "configs/ink_diffusion_smoke.yaml" not in source
    assert "configs/ink_advection_diffusion_smoke.yaml" not in source
    assert "latexmk_available_for_final_paper" in source
    assert "finalize_paper" in source
    assert "require_cuda_for_full_run" in source
    assert "torch_cuda_available_for_full_suite" in source


def test_finalizer_generates_assets_policy_audit_and_pdf():
    script = Path("experiments/finalize_paper.sh").read_text()

    assert 'RESULT_ROOT="${RESULT_ROOT:-results/full_run}"' in script
    assert (
        'PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:-${RESULT_ROOT}/preflight.json}"'
        in script
    )
    assert 'GENERATED_DIR="${GENERATED_DIR:-generated_irreversibility_trust}"' in script
    assert 'RESULT_MAIN_FILE="${RESULT_MAIN_FILE:-main_irreversibility_trust.tex}"' in script
    assert "src.eval.prepare_paper_assets" in script
    assert "--output-dir \"${PAPER_ASSET_OUT}\"" in script
    assert '"${LATEXMK_BIN}" -g -pdf' in script
    assert script.index("src.eval.prepare_paper_assets") < script.index(
        '"${LATEXMK_BIN}" -g -pdf'
    )
    assert "Refusing to finalize final paper assets from smoke" in script
    assert "PAPER_ASSET_OUT must match PAPER_DIR/GENERATED_DIR" in script


def test_docs_explain_preflight_artifact_lifecycle():
    for doc_path in [Path("README.md"), Path("EXPERIMENT_RUNBOOK.md")]:
        text = doc_path.read_text()
        assert "preflight" in text.lower(), doc_path
        assert "results/full_run" in text, doc_path
        assert (
            "PREFLIGHT_OUTPUT=results/full_run/preflight.json"
            in text
        ), doc_path
        assert "REQUIRE_CUDA_FOR_FULL_RUN=1" in text, doc_path
        assert "CPU maintenance preflight" in text, doc_path
