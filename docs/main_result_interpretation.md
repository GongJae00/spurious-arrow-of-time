# Main Result Interpretation

This document interprets the completed non-runtime-limited ten-seed `full`
profile. The older five-seed `main` profile is a smaller confirmation run, not
the primary evidence package.

## Active Design

The main scenario is endpoint-matched:

```text
main_spurious_arrow:
  benchmark_variant: endpoint_matched
  observation_layout: two_channel
  OOD: nuisance arrow reversed
```

The endpoint-matched variant keeps the final nuisance endpoint
label-independent while preserving a directed trail through the sequence. This
separates temporal spurious-arrow shortcutting from final-frame nuisance
residue.

## Main Results

```text
profile: full
seeds: 10
runtime_limited: false
source: results/main_experiments/full/
phenomenon gate audit: passed
counterfactual mitigation gate: failed seed-stability
```

Key neural results:

```text
main_spurious_arrow:
  final_frame_mlp:          IID 0.501, OOD 0.500, gap 0.000
  sequence_erm:             IID 0.971, OOD 0.124, gap 0.848
  nuisance_only_oracle:     IID 0.969, OOD 0.031, gap 0.938
  core_only_oracle:         IID 1.000, OOD 1.000, gap 0.000
  counterfactual_invariance: IID 0.943, OOD 0.756, gap 0.187
  counterfactual seed success: 7/10

no_spurious_correlation:
  sequence_erm:             IID 0.898, OOD 0.899, gap -0.001
  seed-level success:       8/10

residue_visible_control:
  final_frame_mlp:          IID 0.969, OOD 0.031, gap 0.938
  sequence_erm:             IID 0.969, OOD 0.031, gap 0.938
```

## Interpretation

The main run supports the intended claim structure:

```text
1. When the nuisance arrow is label-independent, the mixed sequence model learns
   the causal core.
2. When the endpoint-matched nuisance arrow is label-correlated, the same model
   follows the nuisance and collapses under OOD reversal.
3. The final-frame model is near chance in the endpoint-matched main scenario,
   so the main failure is not explained by final-frame endpoint leakage.
4. The residue-visible control shows the opposite regime: when residue is
   visible, final-frame MLP collapses too.
5. Counterfactual invariance improves mean OOD but reaches the OOD success
   threshold in only 7/10 full seeds, so it is a mitigation diagnostic rather
   than a solved method.
```

Important static-leakage nuance:

```text
The endpoint-matched final-frame result rules out final nuisance endpoint
leakage as the explanation for the main OOD collapse. It does not prove that
all possible final-frame core evidence is absent. The clean core-only/no-
nuisance scenario is an upper-bound diagnostic and should not be used as the
main static leakage comparison.
```

## Claim Boundary

Allowed:

```text
In a controlled irreversible inverse benchmark, standard raw sequence models can
prefer a non-causal nuisance arrow over a recoverable task-causal core, even when
final-frame nuisance leakage is controlled.
```

Allowed with qualification:

```text
Counterfactual nuisance replacement can rescue the model in this synthetic
setting where valid counterfactual pairs are available, but the current simple
implementation is not seed-stable enough for a primary method-success claim.
```

Not allowed:

```text
The result is a universal statement about all sequence models.
The counterfactual method is directly deployable on real data without a valid
intervention, simulator, augmentation, or learned counterfactual generator.
The project measures physical entropy production.
```
