# Latest Main Result Summary

Profile: `complexity_controls`
Primary scenario: `cxns_grid32_clutter`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\complexity_controls --profile complexity_controls`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| sequence_erm | 0.512 | 0.503 | 0.497 | 0.006 | 5 |

## Claim Status

Partially supported: neural ERM did not show the target OOD gap.

This summary separates neural model evidence from diagnostic feature probes.
