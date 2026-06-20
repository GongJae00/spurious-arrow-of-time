# Latest Main Result Summary

Profile: `full`
Primary scenario: `main_spurious_arrow`
Config: `configs/irreversible_source_main.yaml`
Command: `PROFILE=full OUT=results/main_experiments/full bash experiments/main_experiments.sh`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 10 |
| counterfactual_invariance | 0.944 | 0.943 | 0.756 | 0.187 | 10 |
| final_frame_mlp | 0.518 | 0.501 | 0.500 | 0.000 | 10 |
| group_invariance_light | 0.972 | 0.972 | 0.126 | 0.846 | 10 |
| nuisance_only_oracle | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm | 0.972 | 0.971 | 0.124 | 0.848 | 10 |
| time_reversed_sequence | 0.978 | 0.977 | 0.300 | 0.677 | 10 |

## Main Gap Reduction

Mean ERM-minus-counterfactual OOD-gap reduction: `0.6605712890625`
Consistent positive across common seeds: `False`

## Claim Status

Phenomenon supported: ERM shows a large OOD gap in the endpoint-matched
spurious-arrow setting. Counterfactual invariance improves mean OOD but is not
seed-stable enough for a primary method-success claim.

## Required Caveats

```text
no_spurious_correlation sequence_erm reaches the OOD threshold in 8/10 full
seeds.
counterfactual_invariance reaches the OOD threshold in 7/10 full seeds on the
main scenario.
The counterfactual method assumes generator-provided nuisance replacement.
The core-label-randomized nuisance sanity control is not a pure random-label
negative control.
```

This summary separates neural model evidence from diagnostic feature probes.
