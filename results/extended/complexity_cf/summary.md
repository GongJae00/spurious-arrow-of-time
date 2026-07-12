# Latest Main Result Summary

Profile: `complexity_cf`
Primary scenario: `cx_grid32_clutter`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\complexity_cf --profile complexity_cf`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| counterfactual_invariance | 0.644 | 0.642 | 0.354 | 0.287 | 5 |

## Claim Status

Not supported: `sequence_erm` result is missing.

This summary separates neural model evidence from diagnostic feature probes.
