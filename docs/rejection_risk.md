# Rejection Risk Audit

This document lists the most likely reasons a reviewer could reject the project
and the gate that must answer each risk.

| Risk | Why it matters | Required answer |
|---|---|---|
| "This is just generic spurious correlation." | A static or label-coded shortcut would make the arrow language cosmetic. | The nuisance must be a trajectory-level direction statistic. Log dynamic correlations and show static leak probes are weak. |
| "This is just temporal-order self-supervision." | Arrow-of-time pretext work already exists. | The target task must be hidden-cause recovery, not forward/reverse classification. |
| "The inverse problem is fake; final frame gives away the answer." | Then there is no irreversible inverse ambiguity. | Final-frame core oracle must be weak while full-sequence core oracle is strong. |
| "The benchmark bakes in the desired failure." | If the shortcut is too artificial, method results are uninteresting. | Use a standard inverse diffusion/source-localization foundation and report nuisance-only, core-only, sequence ERM, final-frame, time-reversed, and static audits. |
| "The nuisance is not irreversible." | The central term "spurious arrow" would be unsupported. | Nuisance-only forward/reverse or direction-statistic probes must show strong directional evidence. |
| "OOD tuning leaked into design." | Invalidates robustness claims. | Candidate trials must be logged; final gates must not tune method hyperparameters using OOD performance. |
| "The visualization is unintelligible." | The problem must be understandable without trusting code. | Publishable component visualizations must show core, nuisance, mixed, counterfactual, train, and OOD examples with matched scales. |
| "Method claims outrun evidence." | The current method result is not a clean success. | Report counterfactual invariance as a tradeoff/failure unless it improves OOD without destroying IID. |
| "The shortcut is endpoint leakage, not an arrow." | The main run shows final-frame MLP collapses like the nuisance shortcut. | Audit and explain visible nuisance residue. Avoid claiming the failure is purely temporal unless a stricter static-leak audit supports it. |

## Current Decision

The repository has passed the benchmark smoke gate and completed a
non-runtime-limited five-seed main run.

The main result supports the benchmark phenomenon:

```text
sequence_erm: high IID, near-complete OOD collapse
nuisance_only_oracle: same pattern
core_only_oracle: high IID and OOD
```

The main result does not support a positive robustness-method claim:

```text
counterfactual_invariance reduces the OOD gap but sacrifices IID accuracy.
```

Allowed result:

```text
The controlled benchmark shows whether real sequence models are misled by a
spurious irreversible nuisance arrow.
```

Also acceptable:

```text
The simple counterfactual control fails as a robust method, and the failure is
localized without overstating claims.
```

Not acceptable:

```text
The method failed but the paper claim was kept unchanged.
```
