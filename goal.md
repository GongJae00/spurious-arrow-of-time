# Active Goal: Q1-Grade Spurious Arrow Redesign

This is the only active `goal.md`.

The previous evidence package established a strong but imperfect phenomenon:
raw neural predictors collapse under a spurious irreversible nuisance process.
However, the current benchmark is too nuisance-dominant for the strongest paper
claim. The next phase must turn the project from a useful stress test into a
sharper research contribution.

## /goal Objective

```text
Redesign and validate the irreversible inverse inference benchmark so it proves
model reliance on a tempting spurious arrow rather than mere nuisance
occlusion. Build a Q1-grade final evidence package where the mixed observation
contains a learnable task-causal core signal when the nuisance is
label-independent, but standard neural predictors switch to the non-causal
irreversible nuisance when it is correlated with the label. Add endpoint-matched
and residue-controlled variants to separate true temporal-arrow shortcutting
from final-frame leakage. Run calibrated smoke, pilot, main, and full profiles;
produce final figures, tables, audits, and manuscript-ready claim boundaries;
and keep the public repository clean. The work is complete only when the final
benchmark gates, neural results, controls, sweeps, and limitations support a
non-ambiguous paper claim without relying on diagnostic feature probes.
The intended end state is that manuscript writing can begin immediately from
tracked docs and final logged artifacts without another design pass.
```

Short command form:

```text
Build the final Q1-grade benchmark and evidence package: make the core learnable
in mixed data without spurious correlation, make the spurious arrow tempting
only when correlated, isolate temporal-arrow effects from endpoint residue, run
full evidence, and update paper-ready docs/figures.
```

## Core Thesis

The paper should test this precise hypothesis:

```text
In irreversible inverse inference, the task-causal process may be learnable, but
a stronger non-causal arrow can still become the evidence a neural predictor
trusts when it is spuriously correlated with the label.
```

The strongest desired result is not:

```text
nuisance is so large that the core is invisible
```

The strongest desired result is:

```text
the model can learn the core when the nuisance is non-predictive, but it chooses
the spurious irreversible arrow when the nuisance becomes predictive in train.
```

## Current Limitations To Fix

The previous residue-visible result must be preserved only as diagnostic
history. The active main benchmark is now endpoint-matched and two-channel.

Previous residue-visible diagnostic result:

```text
core_only_oracle:
  IID = 1.000
  OOD = 1.000

sequence_erm:
  IID = 0.968
  OOD = 0.032

nuisance_only_oracle:
  IID = 0.968
  OOD = 0.032

final_frame_mlp:
  IID = 0.968
  OOD = 0.032

counterfactual_invariance:
  IID = 0.533
  OOD = 0.465
```

Interpretation:

```text
Supported:
  Strong nuisance-dominant stress test.

Insufficient:
  Proof that ERM chooses nuisance despite a learnable mixed core.

Insufficient:
  Proof that the shortcut requires temporal trajectory reasoning rather than
  final-frame nuisance residue.

Failed:
  Simple counterfactual invariance as a solved method.
```

This goal must directly repair these limitations.

Current non-runtime-limited full result after redesign:

```text
main_spurious_arrow, endpoint-matched:
  final_frame_mlp:      IID 0.501, OOD 0.500, gap 0.000
  sequence_erm:         IID 0.971, OOD 0.124, gap 0.848
  nuisance_only_oracle: IID 0.969, OOD 0.031, gap 0.938
  core_only_oracle:     IID 1.000, OOD 1.000, gap 0.000
  counterfactual_invariance:
                         IID 0.943, OOD 0.756, gap 0.187
                         seed success rate 7/10

no_spurious_correlation:
  sequence_erm:         IID 0.898, OOD 0.899, gap -0.001
                         seed success rate 8/10

residue_visible_control:
  final_frame_mlp:      IID 0.969, OOD 0.031, gap 0.938
  sequence_erm:         IID 0.969, OOD 0.031, gap 0.938
```

This satisfies the ten-seed full phenomenon gates, including seed-level
success-rate checks. The counterfactual mitigation gate fails seed stability
and must not be claimed as a solved method.

## Non-Negotiable Scientific Gates

The final benchmark phenomenon claim requires Gates A-E. Gate F is a separate
mitigation-method honesty gate.

If Gates A-E fail, the spurious-arrow phenomenon claim must be lowered
immediately in the result documents. If Gate F fails, the method claim must be
lowered immediately while preserving the benchmark phenomenon claim if Gates
A-E pass. Do not keep a high claim in parallel with failed gates.

### Gate A: Core Learnability In Mixed Data

When the nuisance exists but is label-independent:

```text
scenario: no_spurious_correlation
input: mixed sequence
model: sequence_erm
required:
  iid_test_accuracy >= 0.80
  ood_test_accuracy >= 0.80
  ood_gap <= 0.10
  seed_success_rate(ood_test_accuracy >= 0.80) >= 0.80
```

Purpose:

```text
Prove that the core is not merely hidden by nuisance energy.
```

If this gate fails, the benchmark is still a stress test but not a strong
spurious-arrow selection benchmark.

### Gate B: Spurious Arrow Trap

When the nuisance arrow is correlated with the label in train/IID and reversed
or removed OOD:

```text
scenario: main_spurious_arrow
sequence_erm:
  iid_test_accuracy >= 0.80
  ood_gap >= 0.25

nuisance_only_oracle:
  iid_test_accuracy >= 0.80
  ood_test_accuracy <= 0.40 under reversed OOD

core_only_oracle:
  iid_test_accuracy >= 0.80
  ood_test_accuracy >= 0.80
```

Purpose:

```text
Prove that the shortcut is predictive in train but unstable OOD.
```

### Gate C: Selection Rather Than Occlusion

Compare `no_spurious_correlation` to `main_spurious_arrow`.

Required pattern:

```text
no_spurious_correlation sequence_erm:
  high IID and high OOD

main_spurious_arrow sequence_erm:
  high IID and low OOD
```

This is the central Q1-grade evidence pattern.

### Gate D: Endpoint-Residue Audit

The final-frame model must be treated as a central audit, not an afterthought.

Required metrics:

```text
final_frame_mlp_iid
final_frame_mlp_ood
final_frame_ood_gap
sequence_erm_iid
sequence_erm_ood
sequence_ood_gap
sequence_minus_final_frame_gap
```

Interpretation rules:

```text
If final_frame_mlp also collapses strongly:
  claim visible irreversible residue, not pure temporal reasoning.

If sequence_erm collapses but final_frame_mlp does not:
  stronger claim: temporal-arrow trajectory shortcut.

If both collapse:
  still valid only as directed irreversible residue shortcut.
```

### Gate E: Endpoint-Matched Main Variant

The main spurious-arrow scenario must use an endpoint-matched variant that
controls endpoint residue.

Required idea:

```text
Keep final-frame nuisance statistics matched across labels and splits as much
as possible, while preserving trajectory direction as the train-correlated
spurious signal.
```

Allowed mechanisms:

```text
1. nuisance pulse returns to matched endpoint after a directed loop
2. pair trajectories with similar final nuisance frame but opposite path
3. subtract or normalize final nuisance endpoint while preserving sequence
   displacement history
4. create a trajectory-only input channel for the nuisance audit, but do not use
   it as the main model input unless clearly labeled
```

Required result for strong temporal claim:

```text
endpoint_matched_final_frame_mlp_ood_gap <= 0.15
endpoint_matched_sequence_erm_ood_gap >= 0.20
```

If this variant fails in the non-runtime-limited run, preserve the result and
keep the paper at the irreversible-residue shortcut claim level.

### Gate F: Counterfactual Method Honesty

Counterfactual invariance may be included only as a control unless it satisfies:

```text
counterfactual_iid >= max(0.75, sequence_erm_iid - 0.15)
counterfactual_ood > sequence_erm_ood
counterfactual_ood_gap < sequence_erm_ood_gap
seed success rate for OOD >= 0.80 is >= 0.80
```

If IID collapses or seed stability fails, report it as a partial mitigation
diagnostic or negative method result, not as a solved method.

### Gate G: Final Result Lock

After the final profile is chosen:

```text
freeze config
freeze method list
freeze seed list
freeze figure scripts
record git commit hash
record hardware/device
record exact command lines
```

No benchmark difficulty, model, or threshold may be changed after inspecting
final OOD results unless the run is explicitly invalidated and renamed.

## Benchmark Redesign Requirements

### Signal Balance

Add calibrated profiles that sweep:

```text
core_scale: [0.45, 0.65, 0.85, 1.0]
nuisance_scale: [1.0, 1.4, 1.8, 2.2, 2.8]
diffusion_steps_between_frames: [4, 6, 8, 10, 12]
observation_noise_std: [0.04, 0.06, 0.08]
```

Search is allowed only in calibration profiles, not by looking at final OOD
test results for the selected main profile.

Selection criteria for final benchmark profile:

```text
1. no_spurious_correlation mixed sequence_erm learns core.
2. main_spurious_arrow sequence_erm collapses OOD.
3. final-frame leakage is quantified and either controlled or honestly framed.
4. core_only_oracle remains high.
5. nuisance_only_oracle remains high IID and low OOD.
```

Record selected profile in:

```text
configs/irreversible_source_main.yaml
docs/benchmark_design_decision.md
docs/latest_result_summary.md
```

### Data Modes

Support:

```text
train_nuisance_mode:
  correlated
  randomized

ood_mode:
  reversed
  randomized
  partial_shift

benchmark_variant:
  residue_visible
  endpoint_matched
  core_learnability
```

The existing `residue_visible` variant may remain as a stress test, but the
final paper should prioritize the calibrated benchmark that passes Gate A.

### Metadata

Every run must log:

```text
core_scale
nuisance_scale
diffusion parameters
noise parameters
train nuisance correlation
OOD nuisance correlation
final-frame static leakage metrics
endpoint matching metrics if applicable
gate pass/fail
config hash
git commit hash
runtime_limited flag
exact command
profile name
scenario name
method list
seed list
validation selection metric
```

The final summary must include enough metadata that another user can rerun the
main table from a clean clone.

## Model And Baseline Requirements

Required raw neural models:

```text
final_frame_mlp
sequence_erm
core_only_oracle
nuisance_only_oracle
time_reversed_sequence
counterfactual_invariance
group_invariance_light
```

Optional but useful if time permits:

```text
small temporal transformer
2D CNN over frame differences
sequence model with final-frame dropout
```

Forbidden for main neural baselines:

```text
metadata-derived features
source_center
source_orientation
nuisance_direction
hand-engineered arrow features
diagnostic feature probes mixed into the neural result table
```

Diagnostic probes remain allowed only in separate audit tables.

## Experiment Matrix

### Stage 0: Calibration

Purpose:

```text
Find a benchmark regime that passes Gate A and Gate B without using final OOD
test performance as a hidden tuning target.
```

Use small/medium runs:

```text
seeds: 3
train: 1024-4096
val_iid: 512-1024
test: 1024-2048
epochs: 10-20
```

Report:

```text
profile table
Gate A/B/C status
selected final profile
rejected profiles and reason
calibration-only metrics clearly separated from final metrics
```

Calibration anti-leakage rule:

```text
Use calibration runs to choose a benchmark regime, not to optimize method
hyperparameters for OOD performance. Once a final profile is selected, rerun it
from scratch under a new output path.
```

### Stage 1: Main Final Run

Run:

```text
profile: main
seeds: at least 5
train >= 8192
val_iid >= 2048
iid_test >= 4096
ood_test >= 4096
epochs >= 30 or documented early stopping
```

Required scenarios:

```text
main_spurious_arrow
no_spurious_correlation
ood_randomized
ood_partial_shift
core_only_no_nuisance
core-label-randomized nuisance sanity control, if runtime permits
```

Required methods:

```text
sequence_erm
final_frame_mlp
core_only_oracle
nuisance_only_oracle
counterfactual_invariance
group_invariance_light
time_reversed_sequence for main_spurious_arrow
```

### Stage 2: Full Final Run

Run if hardware permits:

```text
profile: full
seeds: 10
all main scenarios
selected nuisance/core scale sweeps
endpoint_matched variant if implemented
```

The final paper should use `full` if available. It is now available and should
be the main reported result unless a later run supersedes it.

For Q1-level submission, prefer running `full` unless hardware is genuinely
unavailable. If `full` is deferred, create a clear note explaining why the
available evidence is still sufficient for the target venue.

### Stage 3: Endpoint / Residue Control

The endpoint-matched condition is now the main scenario:

```text
scenario: main_spurious_arrow
benchmark_variant: endpoint_matched
methods:
  sequence_erm
  final_frame_mlp
  core_only_oracle
  nuisance_only_oracle
seeds: at least 5
```

Also run:

```text
scenario: residue_visible_control
benchmark_variant: residue_visible
```

This control demonstrates what final-frame leakage looks like and prevents the
paper from confusing endpoint residue with temporal-arrow shortcutting.

## Required Figures

Figure 1:

```text
Problem schematic:
  hidden cause -> irreversible core diffusion -> inverse label
  independent nuisance arrow -> train shortcut -> OOD reversal/removal
```

Figure 2:

```text
Benchmark calibration/gate figure:
  no_spurious mixed ERM learns core
  main correlated ERM collapses OOD
  core-only and nuisance-only oracles separate task and shortcut
```

Figure 3:

```text
Main neural result:
  IID/OOD bars for final selected profile
  include final_frame_mlp audit
```

Figure 4:

```text
OOD shift sweep:
  reversed vs randomized vs partial_shift
```

Figure 5 if endpoint-matched variant works:

```text
Final-frame vs sequence gap under endpoint matching
```

All figures must be generated from code and logged artifacts. No manual
spreadsheet figure assembly.

Each figure must have:

```text
caption draft
source artifact path
exact script/command
claim supported by the figure
claim not supported by the figure
```

## Required Tables

Table 1:

```text
Benchmark gates and calibration profile
```

Table 2:

```text
Main neural results
```

Table 3:

```text
Controls:
  no_spurious_correlation
  core_only_no_nuisance
  randomized labels if run
```

Table 4:

```text
Endpoint leakage / endpoint-matched audit
```

Each table must report:

```text
mean
standard deviation or standard error
number of seeds
runtime_limited flag
scenario name
method name
```

Do not report only best seed or selected successful seeds.

## Claim Policy

Allowed only if Gate A/B/C pass:

```text
The benchmark shows that models can learn the task-causal irreversible source
from mixed observations when the nuisance is not predictive, yet switch to the
spurious irreversible nuisance when it is correlated with the label.
```

Allowed only if endpoint-matched variant passes:

```text
The failure cannot be explained by final-frame residue alone; temporal
trajectory evidence is sufficient to mislead the model.
```

Allowed if endpoint-matched variant fails but visible-residue results remain:

```text
Directed irreversible nuisance processes can leave misleading observable
residue that dominates learned predictors in inverse inference.
```

Not allowed:

```text
the method solves robustness
pure temporal reasoning failure if final-frame baseline also collapses
physical entropy production is measured
the result proves a universal law of time or neural networks
pilot sweeps are final evidence
```

## Manuscript-Ready Deliverables

Create or update tracked documents that make paper writing immediate without
duplicating stale files:

```text
docs/paper_handoff.md
docs/latest_result_summary.md
docs/main_result_interpretation.md
docs/rejection_risk.md
docs/result_claim_policy.md
docs/figure_plan.md
docs/table_plan.md
```

Required content:

```text
abstract draft
one-sentence contribution
three to five contribution bullets
main claim and evidence table
negative/control result table
limitations
reviewer attack -> answer matrix
which figures/tables go into main paper vs appendix
which results are final vs diagnostic
exact unsupported claims
```

The handoff must be written in paper-author language, not agent log language.

## Reviewer Attack Checklist

Before the final response, answer these in tracked docs:

```text
1. Is this just generic spurious correlation?
2. Is this just endpoint leakage?
3. Is this just nuisance occlusion?
4. Is this just synthetic benchmark overfitting?
5. Did OOD tuning leak into benchmark or method selection?
6. Is the core actually learnable from mixed observations?
7. Does the method claim outrun the result?
8. What is different from arrow-of-time self-supervision?
9. What is different from entropy-production estimation?
10. What is different from inverse diffusion/source localization?
```

Each answer must point to a figure, table, diagnostic, or limitation statement.

## Public Repository Hygiene

Keep the public surface small:

```text
README.md
RESEARCH.md
configs/
docs/
experiments/
src/
tests/
pyproject.toml
requirements.txt
.gitignore
```

Generated artifacts remain ignored:

```text
results/
figures/
paper/
logs/
```

Before final commit:

```bash
git diff --check
PYTHONDONTWRITEBYTECODE=1 PYENV_VERSION=rppg-310 python -m pytest -q -p no:cacheprovider
PYENV_VERSION=rppg-310 PYTHONDONTWRITEBYTECODE=1 bash experiments/smoke_benchmark.sh
PROFILE=smoke PYENV_VERSION=rppg-310 PYTHONDONTWRITEBYTECODE=1 bash experiments/main_experiments.sh
find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.ruff_cache' -o -name '.mypy_cache' \) -print
rg -n "[T]ODO|[F]IXME|final[_]final|newnew|\\bv[0-9]+\\b|dirty|garbage" README.md RESEARCH.md docs src tests configs experiments
```

The find command must print nothing. The search command must print nothing.
The search intentionally excludes `goal.md` because this file documents the
forbidden patterns as text.

Do not commit generated `results/`, `figures/`, `paper/`, or `logs/` contents
unless the user explicitly asks for release artifacts. Public code should
reproduce them.

## Completion Criteria

This goal is complete only when:

```text
Gate A passes:
  no-spurious mixed sequence ERM learns core.

Gate B passes:
  correlated spurious arrow causes high IID and low OOD for sequence ERM.

Gate C passes:
  the difference between no-spurious and spurious scenarios demonstrates model
  selection, not mere nuisance occlusion.

Gate D is documented:
  final-frame leakage/residue is quantified.

Endpoint-matched variant is implemented and evaluated.

Main final profile is run.

Full final profile is run and made the main evidence source.

All figures/tables are regenerated from logged artifacts.

Manuscript-ready docs exist:
  paper handoff
  latest result summary
  main result interpretation
  rejection risk audit
  result claim policy

Docs state exactly:
  supported claims
  unsupported claims
  limitations
  final experimental status

Tests and smoke checks pass.

The public repo remains clean.

A final commit is ready, with generated heavy artifacts ignored.
```

Current completion status:

```text
Phenomenon Gates A-E:
  satisfied by the ten-seed full run.

Gate F:
  failed for simple counterfactual invariance because only 7/10 seeds reach the
  OOD success threshold.
```

The paper-ready claim is therefore the benchmark phenomenon, not a solved
method. If any later gate fails, preserve the result and reduce the claim. Do
not tune the benchmark to force a desired curve without recording the
calibration decision.
