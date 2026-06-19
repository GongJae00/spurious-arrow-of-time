# Active Goal: Paper-Ready Evidence Closure

This is the only active `goal.md`.

The repository is now organized around one clean research question:

```text
When the true task is to infer the hidden cause of an irreversible forward
process, can a stronger but non-causal irreversible process become the arrow
that a learned sequence predictor trusts?
```

This goal is not another redesign cycle. It is the final evidence-closure pass:
align the benchmark, code, results, figures, tables, and written claims so the
project can move into manuscript writing without hidden contradictions.

## /goal Objective

```text
Finalize the paper-ready evidence package for irreversible inverse inference
under spurious arrow shortcuts. Reconcile the executed main and sweep results
with the research claim, harden the benchmark audits against static leakage and
trivial label coding, upgrade result and component visualizations to manuscript
quality, generate final tables and claim-status documents from logged artifacts,
run the required tests and smoke checks, clean public-repository clutter, and
produce a clear paper-writing handoff that states exactly what is supported,
what failed, what remains a limitation, and what must not be claimed. Do not add
new exploratory method stacks or revive deleted exploratory stacks.
```

Short command form:

```text
Close the study for manuscript writing: verify the benchmark, reconcile main
results, improve figures/tables, document supported claims and failures, run
tests/smoke, and clean the public repo.
```

## Current Evidence To Preserve

The latest completed main run supports a strong benchmark phenomenon but not a
successful robustness method.

Main profile:

```text
profile: main
scenario: main_reversed
seeds: 5
runtime_limited: false
```

Logged result summary:

```text
core_only_oracle:
  IID = 1.000
  OOD = 1.000
  gap = 0.000

sequence_erm:
  IID = 0.968
  OOD = 0.032
  gap = 0.936

nuisance_only_oracle:
  IID = 0.968
  OOD = 0.032
  gap = 0.936

final_frame_mlp:
  IID = 0.968
  OOD = 0.032
  gap = 0.936

time_reversed_sequence:
  IID = 0.968
  OOD = 0.032
  gap = 0.936

group_invariance_light:
  IID = 0.968
  OOD = 0.032
  gap = 0.936

counterfactual_invariance:
  IID = 0.533
  OOD = 0.465
  gap = 0.068
```

Interpretation:

```text
Supported:
  A raw neural sequence predictor can lock onto the non-causal nuisance arrow
  and fail under OOD reversal in this controlled benchmark.

Supported:
  The task-causal core sequence contains enough information for high IID/OOD
  performance, as shown by the core-only oracle.

Supported:
  The nuisance-only signal is sufficient for IID success and OOD failure.

Not supported:
  Counterfactual invariance is a successful method here. It reduces the OOD gap
  mainly by collapsing IID performance toward chance.

High-risk observation:
  final_frame_mlp also collapses like the nuisance shortcut. The paper must
  audit and explain whether this is acceptable visible nuisance residue or
  evidence of static leakage that weakens the temporal-arrow claim.
```

The sweep pilot is diagnostic only because it is runtime-limited, but it must be
used to guide the final writeup:

```text
reversed OOD: large ERM gap
randomized OOD: medium ERM gap
partial_shift OOD: intermediate ERM gap
no_spurious_correlation: raw mixed ERM does not recover core well
core_only_no_nuisance: sequence model improves but remains below oracle in the
pilot budget
```

Do not overwrite these facts with optimistic wording.

## Claim Boundaries

Allowed if the final audits remain consistent:

```text
We introduce a controlled benchmark for irreversible inverse inference under
spurious arrow shortcuts.

In the main reversed-shift setting, standard raw sequence predictors learn a
non-causal irreversible nuisance arrow strongly enough to achieve high IID
accuracy and near-complete OOD failure.

Core-only oracle performance verifies that the true irreversible source signal
is learnable, while nuisance-only performance verifies that the shortcut itself
is sufficient and unstable.
```

Allowed only with explicit caveat:

```text
Counterfactual invariance reduces the OOD gap, but in the current implementation
it does so by sacrificing IID accuracy. This is a diagnostic failure of the
simple control, not a solved robustness method.
```

Not allowed:

```text
the method solves spurious arrow reliance
counterfactual invariance succeeds without tradeoff
learned scores measure physical entropy production or heat
the benchmark proves a universal law of neural networks
feature probes are presented as neural ERM results
OOD test performance was used to choose hyperparameters
```

## Non-Negotiable Locks

The goal run must obey:

```text
1. No new broad method family unless a bug makes current models invalid.
2. No revival of deleted STA/SIB/SID/ITM-era stacks.
3. No revision-numbered scratch names, exploratory scratch names, or
   agent-style clutter.
4. No unsupported positive method claim.
5. No OOD-guided tuning.
6. Every plotted result must trace to logged metrics or deterministic
   diagnostics.
7. Public repo cleanliness is part of correctness.
```

## Workstream 1: Repository Hygiene

Make the public surface coherent.

Keep:

```text
README.md
RESEARCH.md
goal.md
docs/
configs/
experiments/
src/
tests/
pyproject.toml
requirements.txt
.gitignore
```

Check and clean:

```text
obsolete generated result folders if they are not needed locally
Python bytecode
pytest/ruff caches
ad hoc temporary files
stale references to deleted method stacks
documents that still say only smoke-level evidence exists
```

Allowed ignored artifact roots:

```text
results/
figures/
paper/
logs/
```

Tracked docs must be lightweight summaries, not raw experiment dumps.

## Workstream 2: Evidence Reconciliation

Update tracked documents so they match the actual completed runs.

Required files:

```text
README.md
docs/latest_result_summary.md
docs/main_result_interpretation.md
docs/result_claim_policy.md
docs/rejection_risk.md
docs/table_plan.md
docs/figure_plan.md
```

They must clearly separate:

```text
benchmark diagnostics
main neural results
runtime-limited sweep diagnostics
negative controls
method failure
allowed paper claims
limitations
next paper-writing steps
```

`docs/main_result_interpretation.md` must no longer say that only smoke-level
neural execution exists if main results are present.

`docs/latest_result_summary.md` may point to the latest run, but the docs must
also preserve the non-runtime-limited main profile result.

## Workstream 3: Benchmark Audits

Before paper claims are strengthened, audit the benchmark for the exact reviewer
attacks most likely to matter.

Required audits:

```text
static leakage audit:
  Does final-frame or static residue alone solve the nuisance shortcut?
  If final_frame_mlp succeeds/fails OOD like nuisance-only, explain whether the
  nuisance arrow leaves a visible endpoint trace and how this affects the claim.

dynamic shortcut audit:
  Verify the reported shortcut is a trajectory-level direction process, not
  label-coded noise or a static initial marker.

core recoverability audit:
  Verify full core sequence is learnable and final core-only frame remains weak
  enough to preserve inverse ambiguity.

negative control audit:
  no_spurious_correlation
  random_labels if available
  core_only_no_nuisance

OOD mode audit:
  reversed
  randomized
  partial_shift
```

If an audit fails, do not patch around the result. Record the failure and update
the claim boundary.

## Workstream 4: Figures And Tables

Upgrade figures to manuscript standard.

Minimum figure set:

```text
Figure 1:
  Compact problem schematic. Show hidden source, irreversible core diffusion,
  nuisance arrow, mixed observation, counterfactual replacement, and OOD
  reversal/removal. It must be readable without inspecting raw arrays.

Figure 2:
  Benchmark gate figure. Show final-frame ambiguity, full-sequence core
  recoverability, nuisance IID/OOD collapse, and dynamic shortcut strength.

Figure 3:
  Main neural result bars. Show IID and OOD accuracy with uncertainty across
  seeds. Feature probes must not appear in this table/figure.

Figure 4:
  OOD-mode or nuisance-strength sweep. Show that the failure changes
  systematically with the shortcut shift, or explicitly state if the sweep is
  only diagnostic.
```

Figure requirements:

```text
generated by code
numbers trace to metrics.jsonl or diagnostics.json
clear axis labels
chance/reference lines where relevant
error bars across seeds
consistent colors
legible at paper column width
no overlapping labels
caption-ready titles
no unsupported success language
```

If current plotting code cannot produce Figure 4, patch `src/eval/main_results.py`
instead of assembling the figure manually.

## Workstream 5: Scale Decision

Decide whether to run another full-scale profile before manuscript writing.

Run or document:

```text
main:
  already completed, 5 seeds, not runtime-limited

sweep_pilot:
  completed, runtime-limited diagnostic

full:
  run only if hardware/time permits and the config is frozen
```

If `full` is not run, the paper package must say:

```text
Primary evidence uses the non-runtime-limited 5-seed main profile.
OOD-mode and negative-control sweeps are diagnostic unless rerun at full scale.
```

Do not treat `sweep_pilot` as camera-ready full evidence.

## Workstream 6: Paper Handoff

Create or update a tracked handoff document under `docs/` that lets manuscript
writing start immediately.

It must include:

```text
one-paragraph abstract draft without overclaiming
main contribution list
exact result claims supported by logs
exact method claims not supported
figure/table checklist
limitations
reviewer-risk answers
which experiments are final and which are diagnostic
```

Suggested file:

```text
docs/paper_handoff.md
```

The handoff must be written for a human paper author, not as an internal agent
log.

## Workstream 7: Validation Commands

Before final response, run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYENV_VERSION=rppg-310 python -m pytest -q -p no:cacheprovider
PYENV_VERSION=rppg-310 PYTHONDONTWRITEBYTECODE=1 bash experiments/smoke_benchmark.sh
git diff --check
find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.ruff_cache' \) -print
rg -n "[T]ODO|[F]IXME|final[_]final|newnew|\\bv[0-9]+\\b|dirty|garbage" README.md RESEARCH.md docs src tests configs experiments
```

If time permits, also run:

```bash
PROFILE=smoke OUT=results/main_experiments/smoke_final \
  PYENV_VERSION=rppg-310 PYTHONDONTWRITEBYTECODE=1 \
  bash experiments/main_experiments.sh
```

Do not run an expensive full profile without an explicit decision that hardware
is available.

## Completion Criteria

This goal is complete only when:

```text
tests pass
smoke benchmark passes
docs reflect the completed main result
main and sweep-pilot evidence are not confused
paper claims are bounded by actual results
static leakage and dynamic shortcut risks are explicitly audited
figures/tables are regenerated or precisely marked as remaining work
public repo clutter is cleaned
validation commands are reported
the final answer provides the exact next action for manuscript writing or full
run execution
```

If the final state is not paper-ready, say exactly why. Do not call the study
complete just because the code runs.
