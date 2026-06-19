# Paper Handoff

This document is the current manuscript handoff. It records what the repository
can support now and what must not be claimed.

## Working Title

```text
Spurious Arrows of Time in Irreversible Inverse Inference
```

## Abstract Draft

Many sequence tasks involve irreversible forward dynamics: diffusion, spreading,
mixing, or damage can make the hidden cause harder to recover than the forward
process is to recognize. We study a controlled setting where the true task is to
infer the hidden source of an irreversible core process, while an independent
non-causal nuisance process has a stronger visible arrow of time and is
correlated with the label only in training. In this benchmark, raw neural
sequence predictors achieve high IID accuracy but fail almost completely when
the nuisance arrow reverses OOD, while a core-only oracle remains accurate. The
result isolates a failure mode in which the most visible arrow is not the
task-causal arrow. A simple counterfactual invariance control reduces the OOD
gap but sacrifices IID accuracy, so we report it as a diagnostic tradeoff rather
than a solved method.

## Main Contributions

```text
1. A controlled irreversible inverse benchmark where the task-causal process
   and the nuisance shortcut are both dynamic processes.

2. A main neural result showing that raw sequence ERM can follow the nuisance
   arrow and collapse under reversed OOD shift.

3. Oracle and control results separating task-causal recoverability from
   shortcut reliance.

4. A negative method result showing that simple counterfactual invariance is not
   enough when it destroys IID performance.
```

## Main Supported Result

From the completed five-seed main run:

```text
core_only_oracle:      IID 1.000, OOD 1.000
sequence_erm:          IID 0.968, OOD 0.032
nuisance_only_oracle:  IID 0.968, OOD 0.032
counterfactual control: IID 0.533, OOD 0.465
```

Supported claim:

```text
In this controlled benchmark, a standard raw sequence model can prefer a
non-causal irreversible nuisance arrow over the task-causal source process.
```

## Not Supported

Do not claim:

```text
the counterfactual method solves the problem
the model failure is purely temporal rather than endpoint/static residue
the learned model estimates physical entropy production
the result universally transfers to real-world time series
```

## Reviewer Risks To Address

```text
1. Final-frame MLP collapse may indicate visible nuisance residue.
   Current audit: docs/static_leakage_audit.md.
2. Sweep-pilot results are runtime-limited and must not be treated as final
   camera-ready sweep evidence.
3. Core-difficulty pilot settings still show shortcut domination, so the mixed
   benchmark should be framed as a strong stress test.
4. The benchmark must be framed as a controlled stress test, not a natural-data
   proof.
5. Method failure must be reported honestly.
```

## Figures Needed

```text
Figure 1: compact problem schematic
Figure 2: benchmark gate diagnostics
Figure 3: main neural IID/OOD result bars
Figure 4: OOD-mode or nuisance-strength sweep, diagnostic unless full-scale
  Current diagnostic source: results/main_experiments/sweep_pilot_goal/
```

## Next Manuscript Step

Finish the evidence-closure goal:

```text
audit static leakage
inspect or regenerate paper-quality figures
sync all docs with main/sweep results
run tests and smoke
decide whether to run a full sweep or write with main + diagnostic sweep
```
