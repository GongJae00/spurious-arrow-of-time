# The Spurious Arrow of Time

## Central Research Question

Can a neural sequence model mistake a nuisance arrow of time for the task
mechanism?

This repository studies sequence tasks where the observation contains multiple
irreversible processes:

```text
core transition mechanism:
  the irreversible process that carries task-relevant information

nuisance transition mechanism:
  an irreversible process that is correlated with the label during training
  but changes under distribution shift
```

The central hypothesis is:

```text
The strongest arrow of time may not be the right arrow of time.
Robust sequence learning should identify the invariant transition mechanism
behind a useful arrow, not merely maximize apparent irreversibility.
```

## Physical Intuition

The motivating example is ink in water. A localized ink drop spreads forward.
After enough diffusion and coarse observation, some information about the
initial state becomes hard to recover. The forward process is easy to observe;
the inverse problem can be ambiguous.

The machine-learning analogue is:

```text
forward temporal evidence exists,
but not all irreversible evidence is task-causal.
```

A model can learn a nuisance flow, drift, or expansion direction because it is a
strong arrow in the training data. That does not mean it has learned the task
mechanism.

## Main Scientific Shift

Earlier versions of this project tried to solve the problem mainly through
latent representation factorization:

```text
z_rev, z_ir_task, z_ir_spur
```

That is not sufficiently fundamental. A representation slot can be arbitrary
unless the transition law itself is constrained.

The current research direction is therefore:

```text
transition mechanism decomposition
```

The method should learn which latent transition mechanism is invariant under
nuisance-arrow interventions. This is stricter than matching representation
distances or fitting a scalar arrow score.

## Current Method Direction: ITM

ITM means Invariant Transition Mechanism.

ITM separates:

```text
z_core:
  candidate task-relevant transition mechanism

z_spur:
  candidate nuisance transition mechanism
```

The task head is allowed to use only the core mechanism representation. During
controlled synthetic training, counterfactual pairs preserve the core process
and label while resampling nuisance dynamics.

Core training pressures:

```text
1. task prediction from the core mechanism
2. counterfactual task consistency
3. core transition invariance across nuisance interventions
4. nuisance transition sensitivity across the same interventions
5. auxiliary core-stat preservation on controlled benchmarks
6. auxiliary nuisance-stat capture for diagnostics
7. adversarial suppression of nuisance statistics in the task mechanism
```

ITM is not allowed to claim success unless full non-smoke results and claim
gates support it.

## Baselines And Diagnostic Methods

Baselines:

```text
ERM
IB
EP-Min
EP-Max
OCP-style order pretraining
lens_like_arrow_classifier
```

Selective diagnostic methods:

```text
SIB:
  counterfactual arrow regularization

SID:
  representation factorization into reversible/task-irreversible/
  nuisance-irreversible factors

ITM:
  current primary method based on invariant transition mechanisms
```

SID remains useful as a diagnostic baseline. It must not be presented as the
main solved method unless its factor-role audit supports that claim.

## Current Benchmarks

### STA-Bench

STA-Bench is an analytic biased-ring control. It creates a label-generating
core Markov process and a nuisance Markov process whose dynamic trajectory
statistic is correlated with the label in train/IID but reversed or removed in
OOD.

It verifies:

```text
known analytic dynamics
dynamic spurious-label correlation
OOD reversal of the nuisance arrow
same observation mixing matrix across splits
train-only label threshold calibration
```

### Ink Advection-Diffusion

Ink Advection-Diffusion is a controlled passive-scalar benchmark. It is not a
full fluid simulator. It exists to express the ink intuition with visible
fields, mass conservation, source recoverability diagnostics, and nuisance flow
arrows.

It verifies:

```text
mass conservation
nonnegative concentration
increasing spread/entropy
core-source label structure
nuisance-flow shortcut structure
counterfactual nuisance resampling
source recoverability over time
```

If the source peak or center remains trivially identifiable, the benchmark may
still be a valid spurious-arrow stress test, but it must not be described as a
strong many-to-one inverse-problem benchmark.

## Safe Physics Language

Allowed:

```text
statistical irreversibility
transition-level arrow evidence
coarse observation
many-to-one information loss when diagnostics support it
invariant transition mechanism
nuisance-arrow intervention
```

Not allowed:

```text
physical heat measurement
exact entropy production in learned latent space
proof of what time is
universal real-world robustness without counterfactual construction
```

Analytic entropy-production quantities are used only in controlled Markov-chain
settings where the assumptions are explicit. Learned latent scores are proxies
or diagnostics, not physical heat.

## Paper Claim Discipline

Before full experiments finish, use hypothesis language:

```text
Irreversibility may become a spurious shortcut.
Robust sequence learning may require identifying invariant transition
mechanisms rather than maximizing the strongest arrow.
```

Allowed only if logged evidence supports it:

```text
ITM improves OOD robustness on the benchmark suite.
ITM learns the intended mechanism separation.
```

Forbidden without direct evidence:

```text
SIB/SID/ITM solves spurious irreversibility in general.
The latent score is physical entropy production.
Ink Advection-Diffusion is a strong inverse-ambiguity benchmark when
inverse_ambiguity_claim_ready=false.
```

## Minimum Paper-Ready Evidence

A strong submission needs:

```text
1. benchmark diagnostics proving a dynamic spurious-arrow trap
2. Ink diagnostics showing what source information does or does not decay
3. ERM/IB/arrow-pretraining/EP baselines under IID and OOD
4. ITM full-suite results over multiple seeds
5. SID/SIB reported as diagnostic baselines, not hidden
6. evidence_audit.json passing
7. result_interpretation.json allowing any positive method language
```
