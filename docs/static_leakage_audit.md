# Endpoint And Static Leakage Audit

This audit asks whether the spurious-arrow result can be explained by a static
final-frame artifact.

## Active Main Variant

The active main benchmark uses:

```text
benchmark_variant: endpoint_matched
observation_layout: two_channel
```

The nuisance process keeps a directed trail over the sequence, but its final
endpoint frame is label-independent. This is the main defense against the claim
that the shortcut is only final-frame residue.

## Smoke Diagnostics

Current smoke diagnostics pass:

```text
final_nuisance_frame_iid_accuracy <= 0.65
final_nuisance_frame_ood_gap <= 0.15
nuisance_forward_reverse_arrow_accuracy >= 0.80
corr_y_realized_nuisance_motion reverses OOD
```

This means the nuisance final frame is not enough, while the sequence still has
a detectable directed arrow.

This audit controls final-frame nuisance endpoint leakage. It does not claim
that every final-frame core cue is absent. In the clean no-nuisance diagnostic
scenario, final-frame core evidence can be easier because the nuisance channel
and observation noise are removed. That scenario should be read as an upper
bound, not as the main static-leakage test.

## Main Neural Audit

From `results/main_experiments/full/`:

```text
main_spurious_arrow, endpoint-matched:
  final_frame_mlp: IID 0.501, OOD 0.500, gap 0.000
  sequence_erm:    IID 0.971, OOD 0.124, gap 0.848

residue_visible_control:
  final_frame_mlp: IID 0.969, OOD 0.031, gap 0.938
  sequence_erm:    IID 0.969, OOD 0.031, gap 0.938
```

Interpretation:

```text
The active endpoint-matched main result cannot be reduced to a final-frame MLP
shortcut through the nuisance endpoint.

The residue-visible control shows what final-frame leakage would look like:
the final-frame MLP collapses there.
```

## Claim Boundary

Allowed:

```text
The main failure is a sequence-level spurious-arrow shortcut with final-frame
nuisance endpoint leakage controlled.
```

Not allowed:

```text
The benchmark removes every possible static cue.
The result proves a universal property of time-series models.
```
