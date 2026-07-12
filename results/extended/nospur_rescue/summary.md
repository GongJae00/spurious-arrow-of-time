# Latest Main Result Summary

Profile: `nospur_rescue`
Primary scenario: `no_spurious_correlation`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\nospur_rescue --profile nospur_rescue`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| sequence_erm | 0.992 | 0.991 | 0.990 | 0.000 | 9 |

## Claim Status

Partially supported: neural ERM did not show the target OOD gap.

This summary separates neural model evidence from diagnostic feature probes.
