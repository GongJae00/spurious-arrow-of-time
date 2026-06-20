# Rejection Risk Audit

This document lists the most likely reasons a reviewer could reject the project
and the gate that must answer each risk.

| Risk | Why it matters | Required answer |
|---|---|---|
| "This is just generic spurious correlation." | A static or label-coded shortcut would make the arrow language cosmetic. | The nuisance must be a trajectory-level direction statistic. Log dynamic correlations and show static leak probes are weak. |
| "This is just temporal-order self-supervision." | Arrow-of-time pretext work already exists. | The target task must be hidden-cause recovery, not forward/reverse classification. |
| "The inverse problem is fake; final frame gives away the answer." | Then there is no irreversible inverse ambiguity. | Quantify final-frame core evidence, show full-sequence core evidence is stronger, and avoid claiming the final frame contains no core information. |
| "The benchmark bakes in the desired failure." | If the shortcut is too artificial, method results are uninteresting. | Use a standard inverse diffusion/source-localization foundation and report nuisance-only, core-only, sequence ERM, final-frame, time-reversed, and static audits. |
| "The nuisance is not irreversible." | The central term "spurious arrow" would be unsupported. | Nuisance-only forward/reverse or direction-statistic probes must show strong directional evidence. |
| "OOD tuning leaked into design." | Invalidates robustness claims. | Candidate trials must be logged; final gates must not tune method hyperparameters using OOD performance. |
| "The visualization is unintelligible." | The problem must be understandable without trusting code. | Publishable component visualizations must show core, nuisance, mixed, counterfactual, train, and OOD examples with matched scales. |
| "Method claims outrun evidence." | The current method result is not a clean success. | Report counterfactual invariance as a tradeoff/failure unless it improves OOD without destroying IID. |
| "The shortcut is endpoint leakage, not an arrow." | A final-frame model could exploit nuisance residue. | Use endpoint-matched main results where final-frame MLP is near chance, and keep residue-visible as a control. |

## Current Decision

The repository has passed the endpoint-matched smoke gate, completed the
non-runtime-limited five-seed `main` run, and completed the non-runtime-limited
ten-seed `full` run.

The full run supports the final-run design:

```text
no_spurious_correlation: sequence_erm learns the core on average, 8/10 seeds
main_spurious_arrow: sequence_erm follows the nuisance arrow
main_spurious_arrow: final_frame_mlp remains near chance
counterfactual_invariance improves mean OOD but fails the seed-stability gate
```

The strongest remaining caveats are:

```text
counterfactual success is not seed-stable enough for a primary method claim
the generator supplies valid nuisance counterfactuals
the full profile contains an exploratory core-label-randomized nuisance control
that should not be described as a pure random-label negative control
```

Allowed result:

```text
The controlled benchmark shows whether real sequence models are misled by a
spurious irreversible nuisance arrow.
```

Also acceptable:

```text
Synthetic counterfactual nuisance replacement improves mean robustness but is
not seed-stable enough to be claimed as a solved method.
```

Not acceptable:

```text
The method failed but the paper claim was kept unchanged.
```
