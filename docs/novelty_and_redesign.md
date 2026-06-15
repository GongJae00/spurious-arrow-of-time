# Novelty And Redesign Plan

## Literature Position

The project touches four existing lines of work.

1. Arrow-of-time self-supervision:
   - Wei et al., "Learning and Using the Arrow of Time"
   - Seif et al., "Machine learning the thermodynamic arrow of time"
   - temporal order / pace / frame-order pretraining papers

2. Temporal self-supervision shortcut analysis:
   - Recent work shows temporal pretext tasks can be too easy and can rely on
     shortcuts such as local appearance statistics.

3. Spurious correlation and counterfactual invariance:
   - Counterfactual invariance formalizes stress tests where irrelevant input
     changes should not change predictions.

4. Inverse problems and diffusion/source localization:
   - Source localization after diffusion is an inverse problem and can be
     ill-posed or many-to-one under coarse/noisy observation.

## Distinct Claim

The strongest novelty is not simply:

```text
we use an arrow-of-time objective
we add another spurious-correlation benchmark
we propose another regularizer
```

The distinct claim is:

```text
Irreversibility itself can be spuriously predictive.
```

This differs from ordinary temporal shortcut learning because the nuisance is
not merely a static artifact or easy frame cue. It is a dynamic irreversible
trajectory statistic that is predictive in training and unreliable under OOD
shift.

The method-side novelty target is:

```text
The feature is not the arrow score itself.
The feature is the transition mechanism that produces task-relevant arrow
evidence and remains invariant under nuisance-arrow interventions.
```

This separates the project from:

```text
arrow-of-time pretraining:
  learns temporal direction as a useful signal

thermodynamic arrow estimation:
  estimates irreversibility or entropy-production-like quantities

ordinary spurious correlation work:
  studies static or generic nuisance correlations

this project:
  studies a nuisance irreversible transition mechanism that is predictive in
  training and unreliable under shift
```

## What Was Missing

The original ink intuition was stronger than the current operationalization.
The intuition was about:

```text
many-to-one information loss
inverse ambiguity
coarse observation
source recoverability decaying over time
```

The current Ink Advection-Diffusion generator expresses mass-conserving
spreading and spurious nuisance flow, but diagnostics reveal an important
limitation:

```text
source mass and peak contrast decay,
but the source peak/center can remain localizable in the core field.
```

Therefore, the current benchmark supports a controlled spurious-arrow shortcut
claim more strongly than a deep many-to-one inverse-problem claim.

## Required Redesign

The next research version should be information-loss first.

### 1. Add Source Recoverability Curves

For each time point, log:

```text
mass near true source
peak contrast
peak-location error
center-of-mass source error
source classifier accuracy from x_t only
source classifier accuracy from x_{0:t}
```

The paper should only claim inverse ambiguity when source recovery degrades
under the observation available to the model.

### 2. Make The Main Task A Controlled Inverse Problem

The core task should be:

```text
infer an initial source attribute from delayed, noisy, coarse observations
```

The nuisance shortcut should be:

```text
flow direction or another irreversible trajectory statistic correlated with y
in train/IID and reversed or removed in OOD
```

### 3. Avoid Trivial Source Localization

Use at least one of:

```text
delayed observation window
coarse sensor grid
partial observation mask
stronger observational noise
multi-source initial conditions with overlapping late fields
source attribute not equal to center of mass
```

Do not claim strong inverse ambiguity if the source peak or center-of-mass
remains an easy oracle.

### 4. Replace Representation-First Claims With Mechanism Claims

The main method claim should be:

```text
We evaluate whether invariant transition mechanism learning can identify the
task-relevant mechanism under nuisance-arrow intervention.
```

Not:

```text
representation slots alone solve irreversibility factorization.
```

Clean SID factorization requires explicit factor-role audit support. ITM
success requires the current primary claim gate to pass.

## Minimum Paper-Ready Evidence

A strong paper needs:

```text
1. Data diagnostics showing source information decay under the model observation.
2. A spurious-arrow trap where ERM/IB/arrow-pretraining fail OOD.
3. Closure controls showing failure is tied to dynamic nuisance-arrow shift.
4. Honest SIB/SID diagnostic results, including negative or mixed results.
5. ITM as the primary method, with full-suite claim-gated evidence and
   `itm_mechanism_audit.json`.
6. Clear wording that learned latent scores are not physical heat or exact EP.
```

## Current Decision

The repository should not yet frame Ink Advection-Diffusion as a fully solved
many-to-one inverse-problem benchmark. It should frame it as a controlled
spurious-arrow stress test with newly added source recoverability diagnostics.

The next full run should be regenerated after the information-loss diagnostics
and any source-ambiguity redesign are finalized.
