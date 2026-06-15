# Theory Definitions

Date: 2026-06-05

These definitions control terminology in code comments, documentation, and the
paper. They are intentionally narrower than the motivating physics language.

## Trajectory

A trajectory is an ordered sequence of states or observations:

```text
x_0:L = (x_0, x_1, ..., x_{L-1})
```

`L` is the number of observed time points. The number of transitions is:

```text
n_transitions = L - 1
```

## Time Reversal

Time reversal reverses only the temporal axis:

```text
reverse(x_0:L) = (x_{L-1}, ..., x_1, x_0)
```

It does not change feature dimensions, task labels, or hidden state values.
For arrow-pretraining and probing:

```text
1 = original forward trajectory
0 = reversed trajectory
```

Task labels must not be recomputed from reversed sequences unless the experiment
is explicitly labeled as a separate diagnostic task.

## Dynamic Statistic

A dynamic statistic is computed from transitions or trajectory-level evolution,
not from a single static state.

Examples:

```text
signed drift
net displacement
flow direction
diffusion/advection direction
forward/reverse arrow evidence
```

Static initial sector is a static shortcut control, not a main spurious-arrow
mechanism.

## Irreversibility Proxy

An irreversibility proxy is a learned or analytic score that detects asymmetric
trajectory structure. In this project, learned proxies are not physical heat and
are not exact thermodynamic entropy production.

Allowed names:

```text
latent_arrow_score
transition_irreversibility_score
latent_transition_ep_proxy
```

## Latent Arrow Evidence

The implemented latent score is transition-level evidence:

```text
sigma_transition =
  sum_t log p_theta(z_{t+1}|z_t) - log r_omega(z_t|z_{t+1})
```

Unless boundary or marginal density terms are explicitly estimated and
validated, do not call this exact path entropy production.

Use:

```text
learned latent path-wise irreversibility score inspired by stochastic thermodynamics
transition-level latent arrow evidence
```

Do not use:

```text
physical heat dissipation
exact entropy production of the observed system
```

## Spurious Arrow

A spurious arrow is a nuisance dynamic process whose direction-sensitive
irreversibility statistic is predictive in train/IID environments but is not the
task-relevant robust signal under OOD shift.

It differs from ordinary temporal correlation:

```text
ordinary temporal correlation:
  a feature varies with time and is correlated with y.

spurious arrow:
  the direction of nuisance evolution is correlated with y in train/IID and
  reverses, disappears, or randomizes under OOD.
```

## Task-Relevant Arrow

A task-relevant arrow is irreversible trajectory information belonging to the
core process that is causally or procedurally tied to the label.

Example:

```text
core net displacement determines y
ink source dynamics determine y
```

## Nuisance Counterfactual

A nuisance counterfactual preserves the core trajectory and label while changing
the nuisance dynamics:

```text
x = mix(core, nuisance)
x_cf = mix(same core, resampled nuisance)
y_cf = y
```

This assumption is valid in controlled synthetic benchmarks. For real data, it
requires known interventions, simulation, valid augmentation, or a learned
counterfactual generator.

## OOD Arrow Shift

An OOD arrow shift changes the label relationship of the nuisance dynamic
statistic:

```text
train/IID:
  corr(y, nuisance_dynamic_stat) is nontrivial

OOD:
  the correlation is reversed, removed, or randomized
```

OOD data is final-report only and must not be used for model selection,
checkpointing, setpoint selection, or hyperparameter tuning.

## Many-To-One Inverse Ambiguity

Many-to-one inverse ambiguity means many prior histories can lead to similar
coarse observations.

Examples:

```text
diffused ink observation -> many possible source histories
```

This motivates the project, but the benchmarks are controlled ML tests, not
full physical simulations.

## Behavioral Robustness

Behavioral robustness means final task performance remains high under OOD arrow
shift:

```text
small OOD gap = iid_test_accuracy - ood_test_accuracy
high ood_test_accuracy
```

Behavioral robustness does not imply interpretable factorization.

## Mechanism Factorization

Mechanism factorization means representations separate roles as intended:

```text
z_rev:
  reversible/configurational information

z_ir_task:
  task-relevant irreversible information

z_ir_spur:
  spurious/nuisance irreversible information
```

Current evidence does not support a clean mechanism factorization claim.
Conditional audits are needed to distinguish true factor failure from probe
confounding or orientation artifacts.
