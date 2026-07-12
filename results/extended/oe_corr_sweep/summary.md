# Latest Main Result Summary

Profile: `oe_corr_sweep`
Primary scenario: `oecorr_0p97`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\oe_corr_sweep --profile oe_corr_sweep`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 10 |
| final_frame_mlp | 0.521 | 0.497 | 0.502 | -0.005 | 10 |
| nuisance_only_oracle | 0.969 | 0.969 | 0.031 | 0.938 | 10 |
| sequence_erm | 0.969 | 0.969 | 0.031 | 0.938 | 10 |

## Claim Status

Partially supported: ERM gap exists, but counterfactual result is missing.

This summary separates neural model evidence from diagnostic feature probes.
