# Benchmark Specification

The benchmarks isolate one question:

```text
Can a dynamic arrow of time become a spurious shortcut under distribution shift?
```

The current method question is sharper:

```text
Can a model identify the transition mechanism that remains invariant when the
nuisance arrow is counterfactually changed?
```

## Shared Protocol

Every benchmark returns:

```text
train
val_iid
iid_test
ood_test
```

`train` is used for fitting. `val_iid` is used for model selection. `iid_test`
and `ood_test` are final reporting splits only. OOD results must not be used
for thresholds, checkpoints, hyperparameters, or setpoints.

## STA-Bench

STA-Bench is an analytic biased-ring control. It contains:

```text
core process:
  label-generating Markov chain

nuisance process:
  dynamic trajectory statistic correlated with y in train/IID and reversed or
  removed in OOD
```

For a ring transition model:

```text
analytic_ep = (p_forward - p_backward) * log(p_forward / p_backward)
signed_drift = p_forward - p_backward
```

`analytic_ep` is an irreversibility magnitude. `signed_drift` and realized net
displacement encode direction.

Required gates:

```text
same_mixing_matrix
train_threshold_reused
core_oracle_high_iid
core_oracle_high_ood
spurious_rule_high_iid
spurious_rule_breaks_ood
dynamic spurious-y correlation computed from trajectory statistics
```

## Ink Advection-Diffusion

Ink Advection-Diffusion is a controlled passive-scalar task. It is not a full
fluid simulator. It exists to express the original ink-spreading intuition in a
visually inspectable and numerically constrained way.

Core:

```text
sample a core ink source
evolve a passive scalar under diffusion
define y from the core source statistic
```

Nuisance:

```text
sample a nuisance source
evolve a passive scalar under advection-diffusion
correlate nuisance flow direction with y in train/IID
reverse or randomize that correlation in OOD
```

Observation:

```text
x_t = core_scale * core_field_t + spur_scale * nuisance_field_t + noise_t
```

Counterfactual:

```text
preserve core field and label
resample nuisance source/flow
reuse observation noise by default
```

This counterfactual pair is not just an augmentation. It is the mechanism
intervention used by ITM:

```text
core mechanism should remain stable
nuisance mechanism should change
task prediction should follow the stable core mechanism
```

Required gates:

```text
mass_conservation
nonnegative_concentration
spread_increase
entropy_increase
visible_signal
train_threshold_reused
core_oracle_high_iid
core_oracle_high_ood
spurious_rule_high_iid
spurious_rule_breaks_ood
dynamic_spurious_corr_train
dynamic_spurious_corr_iid
dynamic_spurious_corr_ood_reversed
counterfactual_preserves_core_and_label
counterfactual_changes_spurious_flow
```

Additional inverse-ambiguity diagnostics:

```text
mass_near_source_observed_start/final
peak_contrast_observed_start/final
peak_error_observed_final
center_error_observed_final
inverse_ambiguity_claim_ready
```

These diagnostics are claim gates, not just engineering metrics. If the source
peak or center remains trivially identifiable, the benchmark may still be a
valid spurious-arrow stress test, but it must not be described as a strong
many-to-one inverse-problem benchmark.

## Interpretation

A benchmark is valid for this paper only when the core oracle works IID/OOD,
the spurious train rule works IID, and the same spurious rule fails under OOD
shift. This distinguishes a spurious-arrow stress test from a broken task.

Allowed wording:

```text
controlled synthetic diagnostic
dynamic spurious-arrow stress test
passive-scalar conceptual experiment
```

Forbidden wording:

```text
exact physical entropy production measurement
realistic fluid simulation benchmark
universal robustness proof
clean SID factorization without factor-role evidence
ITM success without result_interpretation support
```

`result_interpretation` depends on both task metrics and
`itm_mechanism_audit.json`. High OOD accuracy alone is not enough for the
mechanism claim.
