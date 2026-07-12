# Latest Main Result Summary

Profile: `family_cf`
Primary scenario: `fam_diffusion_translate`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\family_cf --profile family_cf`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| counterfactual_invariance | 0.872 | 0.868 | 0.729 | 0.139 | 5 |

## Claim Status

Not supported: `sequence_erm` result is missing.

This summary separates neural model evidence from diagnostic feature probes.
