# Method Taxonomy

## Baselines

```text
ERM:
  supervised GRU classifier

IB:
  supervised GRU with stochastic pooled bottleneck

EP-Min:
  supervised task loss plus dynamics losses plus irreversibility suppression

EP-Max:
  supervised task loss plus dynamics losses plus irreversibility encouragement

OCP-style:
  order/arrow pretraining followed by frozen and fine-tuned evaluation

lens_like_arrow_classifier:
  forward/reverse classifier baseline inspired by arrow-of-time pretraining
```

## Selective Methods

```text
SIB:
  counterfactual arrow regularization with task consistency

SID:
  factorized representation with reversible, task-irreversible, and
  nuisance-irreversible subspaces

ITM:
  invariant transition mechanism learning; the task head consumes only the
  core mechanism representation, while nuisance dynamics are handled through
  counterfactual mechanism sensitivity and auxiliary diagnostics
```

SID role claims require explicit factor audits. The task head must not consume
the nuisance-irreversible factor directly.

ITM is the current primary method. Its claim is stronger than representation
matching: the core transition mechanism should remain stable under nuisance
counterfactuals, while the nuisance mechanism should change.

ITM claim language requires `itm_mechanism_audit.json`. The audit checks:

```text
task head excludes the spurious mechanism
task representation predicts label/core dynamics after residualization
task representation has low residual spurious-dynamic evidence
spurious representation captures spurious dynamics
core transition deltas are counterfactually stable
spurious transition deltas are counterfactually sensitive
```
