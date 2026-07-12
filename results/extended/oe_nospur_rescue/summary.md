# Latest Main Result Summary

Profile: `oe_nospur_rescue`
Primary scenario: `oe_no_spurious`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\oe_nospur_rescue --profile oe_nospur_rescue`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| sequence_erm | 1.000 | 0.999 | 0.999 | 0.000 | 3 |

## Claim Status

Partially supported: neural ERM did not show the target OOD gap.

This summary separates neural model evidence from diagnostic feature probes.
