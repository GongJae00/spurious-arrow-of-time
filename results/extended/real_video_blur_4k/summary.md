# Latest Main Result Summary

Profile: `real_video_blur_4k`
Primary scenario: `rv_main`
Config: `configs\irreversible_source_extended.yaml`
Command: `python -m src.train.main_experiment --config configs\irreversible_source_extended.yaml --out results\extended\real_video_blur_4k --profile real_video_blur_4k`
Device: `cuda`
Runtime-limited: `False`

## Method Table

| Method | Val IID | IID Test | OOD Test | OOD Gap | Seeds |
|---|---:|---:|---:|---:|---:|
| core_only_oracle | 1.000 | 1.000 | 1.000 | 0.000 | 5 |
| final_frame_mlp | 0.530 | 0.521 | 0.478 | 0.043 | 5 |
| nuisance_only_oracle | 0.676 | 0.653 | 0.337 | 0.316 | 5 |
| sequence_erm | 0.946 | 0.940 | 0.898 | 0.042 | 5 |

## Claim Status

Partially supported: neural ERM did not show the target OOD gap.

This summary separates neural model evidence from diagnostic feature probes.
