# Latest Main Result Summary

Profile: `family_controls`
Primary scenario: `famns_diffusion_translate`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\family_controls --profile family_controls`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| sequence_erm | 0.713 | 0.701 | 0.700 | 0.001 | 5 |

## Claim Status

Partially supported: neural ERM did not show the target OOD gap.

This summary separates neural model evidence from diagnostic feature probes.
