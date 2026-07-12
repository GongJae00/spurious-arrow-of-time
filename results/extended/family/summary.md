# Latest Main Result Summary

Profile: `family`
Primary scenario: `fam_diffusion_translate`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\family --profile family`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 5 |
| final_frame_mlp | 0.531 | 0.489 | 0.496 | -0.007 | 5 |
| nuisance_only_oracle | 0.971 | 0.972 | 0.029 | 0.943 | 5 |
| sequence_erm | 0.971 | 0.972 | 0.029 | 0.943 | 5 |

## Claim Status

Partially supported: ERM gap exists, but counterfactual result is missing.

This summary separates neural model evidence from diagnostic feature probes.
