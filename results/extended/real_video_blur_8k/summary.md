# Latest Main Result Summary

Profile: `real_video_blur_8k`
Primary scenario: `rv_main`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\real_video_blur_8k --profile real_video_blur_8k`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 5 |
| final_frame_mlp | 0.540 | 0.519 | 0.476 | 0.042 | 5 |
| nuisance_only_oracle | 0.735 | 0.734 | 0.267 | 0.468 | 5 |
| sequence_erm | 0.999 | 0.998 | 0.998 | 0.000 | 5 |

## Claim Status

Partially supported: neural ERM did not show the target OOD gap.

This summary separates neural model evidence from diagnostic feature probes.
