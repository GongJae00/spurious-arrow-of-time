# Smoke Interpretation

Smoke command:

```bash
PYENV_VERSION=rppg-310 bash experiments/smoke_benchmark.sh
```

Artifact directory:

```text
results/smoke_benchmark/
```

## Overall Decision

```text
benchmark gate passed: true
```

This is evidence that the benchmark construction is coherent enough for later
full method experiments. It is not a method success claim.

## Single-Seed Smoke Gate

| Gate | Value | Interpretation |
|---|---:|---|
| final-frame core oracle accuracy | 0.609375 | Final diffused core observation is weak enough to count as an inverse ambiguity smoke test. |
| full-sequence core oracle accuracy | 1.000000 | The core sequence carries recoverable source-pattern information. |
| core-only OOD accuracy | 1.000000 | The task-causal source signal remains valid under OOD nuisance shift. |
| nuisance-only IID accuracy | 0.949219 | The nuisance arrow is a strong train/IID shortcut. |
| nuisance-only OOD accuracy | 0.023438 | The reversed nuisance arrow fails badly OOD. |
| mixed feature-probe OOD gap | 0.925781 | A mixed diagnostic probe using visible dynamic features is strongly misled by the nuisance arrow. |
| static feature accuracy | 0.472656 | Static/final leakage does not solve the task in this smoke run. |
| core forward/reverse arrow accuracy | 1.000000 | The core process is temporally asymmetric. |
| nuisance forward/reverse arrow accuracy | 1.000000 | The nuisance is now an actual temporal-arrow process, not only a left/right moving object. |
| counterfactual core residual max abs | 0.000000 | Counterfactual replacement preserves the core and observation noise exactly under the generator identity. |

## Twenty-Seed Stability Check

The smoke configuration was checked for seeds 0 through 19.

```text
passed seeds: 20 / 20
```

Observed ranges:

| Metric | Min | Mean | Max |
|---|---:|---:|---:|
| final-frame core oracle accuracy | 0.484375 | 0.566406 | 0.644531 |
| full-sequence core oracle accuracy | 1.000000 | 1.000000 | 1.000000 |
| core-only OOD accuracy | 1.000000 | 1.000000 | 1.000000 |
| nuisance-only IID accuracy | 0.949219 | 0.970313 | 0.992188 |
| nuisance-only OOD accuracy | 0.007812 | 0.030859 | 0.054688 |
| mixed feature-probe OOD gap | 0.917969 | 0.939453 | 0.968750 |
| static feature accuracy | 0.460938 | 0.509766 | 0.589844 |
| core forward/reverse arrow accuracy | 1.000000 | 1.000000 | 1.000000 |
| nuisance forward/reverse arrow accuracy | 1.000000 | 1.000000 | 1.000000 |
| counterfactual core residual max abs | 0.000000 | 0.000000 | 0.000000 |

## What This Supports

The smoke run supports the benchmark-level statement:

```text
The constructed task contains a recoverable task-causal irreversible source
signal and a stronger non-causal nuisance arrow that breaks under OOD reversal.
```

The updated nuisance process is not merely a moving object. It leaves a temporal
trail, making forward and reverse sequences distinguishable while still carrying
a direction statistic that is spuriously correlated with the label.

## What This Does Not Support Yet

Do not claim:

```text
any proposed method solves the problem
the mixed feature-probe result is a neural ERM result
factorized latent-mechanism methods work
the benchmark is already publication-complete
learned scores are physical entropy production
```

## Remaining Paper-Readiness Requirements

Before writing the final paper from main experiments:

```text
1. Run actual neural ERM and method baselines on the benchmark.
2. Keep mixed_feature_probe separate from neural ERM in tables.
3. Report multi-seed means and intervals, not only seed 0.
4. Include component visualizations and gate tables.
5. Keep the claim at benchmark/method evidence level, not physical ontology.
```
