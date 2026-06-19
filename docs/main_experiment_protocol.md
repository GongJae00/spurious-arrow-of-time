# Main Experiment Protocol

## Purpose

The main experiment tests whether raw neural sequence models trust the
task-causal irreversible source process or the stronger spurious nuisance arrow.

Diagnostic feature probes are benchmark checks only. They are never reported as
neural model results.

## Splits

```text
train: model fitting
val_iid: early stopping and checkpoint selection only
iid_test: final IID report
ood_test: final OOD report
```

No hyperparameter, checkpoint, threshold, or model choice may be selected using
`ood_test`.

## Run Profiles

| Profile | Purpose | Claim Status |
|---|---|---|
| `smoke` | Code and artifact correctness | No paper claim |
| `pilot` | Training dynamics and leakage check | Diagnostic only |
| `main` | Primary table | Claim allowed only if gates pass |
| `full` | Robustness/camera-ready extension | Strongest evidence if completed |

Recommended `main` split sizes:

```text
train >= 8192
val_iid >= 2048
iid_test >= 4096
ood_test >= 4096
```

## Required Methods

```text
final_frame_mlp
sequence_erm
core_only_oracle
nuisance_only_oracle
time_reversed_sequence
counterfactual_invariance
```

All methods consume raw tensors only. Metadata and diagnostic feature functions
are forbidden as neural-model inputs.

## Required Outputs

```text
results/main_experiments/<profile>/metrics.jsonl
results/main_experiments/<profile>/summary.json
results/main_experiments/<profile>/manifest.json
results/main_experiments/<profile>/summary.md
docs/latest_result_summary.md
```

Generated result directories are ignored by git. The tracked summary in `docs/`
is the public lightweight record.
