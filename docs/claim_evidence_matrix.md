# Claim-Evidence Matrix

This document is the claim gate for the Neurocomputing manuscript. A statement
may enter the paper only if it is represented here or is a direct factual
description of the benchmark/protocol.

Primary evidence source:

```text
results/main_experiments/full/summary.md
results/main_experiments/full/final_gate_audit.md
results/main_experiments/full/claim_audit.md
results/main_experiments/full/method_table.md
results/main_experiments/full/scenario_table.md
```

Current full run:

```text
profile: full
seeds: 10
runtime_limited: False
primary scenario: main_spurious_arrow
phenomenon gates passed: True
counterfactual mitigation gates passed: False
```

## Claim C1: Spurious Arrow Failure Exists

Plain meaning:

```text
A standard neural sequence predictor can learn the training task well while
failing when the non-causal nuisance arrow reverses OOD.
```

Technical statement:

```text
In the endpoint-matched main_spurious_arrow scenario, sequence_erm reaches high
IID accuracy but has a large OOD gap under nuisance-arrow reversal.
```

Evidence:

```text
sequence_erm, main_spurious_arrow, n=10:
  IID test accuracy: 0.971 +- 0.009
  OOD test accuracy: 0.124 +- 0.292
  OOD gap:           0.848 +- 0.283
```

Required visual:

```text
Figure 4: main model results
Table 3: main results
```

Required control:

```text
Claim C2: core learnability
Claim C3: nuisance-only shortcut
Claim C4: final-frame endpoint audit
```

Allowed wording:

```text
Neural sequence ERM follows a non-causal irreversible nuisance under the
training correlation and fails when that nuisance arrow reverses OOD.
```

Forbidden wording:

```text
ERM proves neural networks always learn the wrong arrow.
The failure is universal.
The model learns thermodynamic entropy production.
```

Boundary:

```text
This is a controlled synthetic benchmark result, not a universal statement
about all sequence models or all physical irreversible systems.
```

Reviewer attack answered:

```text
Is there a real neural-model failure, or only a diagnostic probe?
```

## Claim C2: The Core Signal Is Learnable

Plain meaning:

```text
The benchmark is not merely hiding the true signal under nuisance energy.
```

Technical statement:

```text
The causal core remains learnable both through an oracle input and through
mixed inputs when the nuisance is label-independent.
```

Evidence:

```text
core_only_oracle, main_spurious_arrow, n=10:
  IID test accuracy: 1.000
  OOD test accuracy: 1.000
  OOD gap:           0.000

sequence_erm, no_spurious_correlation, n=10:
  IID test accuracy: 0.898
  OOD test accuracy: 0.899
  OOD gap:          -0.001
  OOD threshold seed success: 8/10
```

Required visual:

```text
Figure 3: evidence gates and oracle controls
Table 2: benchmark/evaluation protocol
Table 3: main results
```

Allowed wording:

```text
The mixed sequence contains a learnable core signal when the nuisance is not
label-predictive.
```

Forbidden wording:

```text
The core is equally easy in every seed.
The benchmark is trivial.
The oracle result alone proves mixed-input learnability.
```

Boundary:

```text
The no_spurious_correlation seed success rate is exactly 8/10. State this when
making strong learnability claims.
```

Reviewer attack answered:

```text
Is the observed collapse just nuisance occlusion rather than spurious selection?
```

## Claim C3: The Nuisance Arrow Is A Tempting Wrong Shortcut

Plain meaning:

```text
The nuisance process is strongly predictive in the training/IID environment but
wrong under the OOD shift.
```

Technical statement:

```text
The nuisance-only oracle reaches high IID accuracy and severe OOD collapse under
reversed nuisance-arrow evaluation.
```

Evidence:

```text
nuisance_only_oracle, main_spurious_arrow, n=10:
  IID test accuracy: 0.969 +- 0.003
  OOD test accuracy: 0.031 +- 0.002
  OOD gap:           0.938 +- 0.004
```

Required visual:

```text
Figure 1: conceptual problem diagram
Figure 3: evidence gates and oracle controls
Table 3: main results
```

Allowed wording:

```text
The nuisance arrow is a high-accuracy IID shortcut and becomes invalid when its
direction-label relation is reversed.
```

Forbidden wording:

```text
The nuisance is causal.
The nuisance is merely static label noise.
The nuisance proves all temporal features are harmful.
```

Boundary:

```text
The claim is about a configured irreversible nuisance process in this benchmark.
```

Reviewer attack answered:

```text
Is the spurious factor actually tempting enough to explain ERM behavior?
```

## Claim C4: Endpoint Leakage Is Controlled In The Main Scenario

Plain meaning:

```text
The main collapse is not explained by a final-frame-only classifier.
```

Technical statement:

```text
In the endpoint-matched main_spurious_arrow scenario, final_frame_mlp is near
chance and has near-zero OOD gap, while sequence_erm has a large OOD gap.
```

Evidence:

```text
final_frame_mlp, main_spurious_arrow, n=10:
  IID test accuracy: 0.501 +- 0.010
  OOD test accuracy: 0.500 +- 0.009
  OOD gap:           0.000 +- 0.016

sequence_minus_final_frame_gap:
  approximately 0.847

residue_visible_control:
  final_frame_mlp IID 0.969, OOD 0.031, OOD gap 0.938
```

Required visual:

```text
Figure 2: benchmark construction and visual example
Figure 3: endpoint/residue audit
```

Allowed wording:

```text
Endpoint matching prevents a final-frame-only classifier from explaining the
main sequence-model collapse.
```

Forbidden wording:

```text
The failure is purely temporal in every benchmark variant.
There is no residue issue anywhere.
```

Boundary:

```text
The residue_visible_control shows that without endpoint control, final-frame
residue can fully explain collapse. The main claim must therefore be tied to
the endpoint-matched scenario.
```

Reviewer attack answered:

```text
Could a static final image solve the shortcut?
```

## Claim C5: The Central Pattern Is Selection, Not Just Difficulty

Plain meaning:

```text
The same model can learn the core when the nuisance is not predictive, but
switches to the nuisance when the nuisance becomes predictive during training.
```

Technical statement:

```text
No-spurious mixed-input ERM remains robust, while main-spurious mixed-input ERM
shows high IID and low OOD.
```

Evidence:

```text
sequence_erm, no_spurious_correlation:
  IID 0.898, OOD 0.899, gap -0.001

sequence_erm, main_spurious_arrow:
  IID 0.971, OOD 0.124, gap 0.848
```

Required visual:

```text
Figure 3: evidence gates and scenario comparison
Figure 5: scenario/robustness audit
```

Allowed wording:

```text
The failure appears when the nuisance arrow becomes label-predictive during
training, not merely when the nuisance exists.
```

Forbidden wording:

```text
All seeds learn the core perfectly in the no-spurious setting.
The benchmark completely eliminates all nuisance dominance.
```

Boundary:

```text
No-spurious robustness is strong but not perfect: 8/10 seeds pass the OOD
threshold.
```

Reviewer attack answered:

```text
Is the model forced to fail because the core is always too hard?
```

## Claim C6: Counterfactual Replacement Mitigates But Does Not Solve

Plain meaning:

```text
Generator-provided nuisance replacement helps on average, but is not stable
enough to be the main method claim.
```

Technical statement:

```text
Counterfactual_invariance reduces the mean OOD gap relative to sequence_erm,
but fails the seed-stability gate.
```

Evidence:

```text
counterfactual_invariance, main_spurious_arrow, n=10:
  IID test accuracy: 0.943 +- 0.156
  OOD test accuracy: 0.756 +- 0.411
  OOD gap:           0.187 +- 0.394
  OOD threshold seed success: 7/10

sequence_erm OOD gap: 0.848
counterfactual OOD gap: 0.187
mean ERM-minus-counterfactual gap reduction: 0.661
consistent positive across common seeds: False
```

Required visual:

```text
Figure 4: main model results with seed visibility
Table 3: include seed success note
```

Allowed wording:

```text
Counterfactual nuisance replacement substantially improves mean OOD accuracy
but does not meet the seed-stability gate.
```

Forbidden wording:

```text
The method solves spurious arrows.
The method is robust across seeds.
Counterfactual invariance is deployable without interventions or a generator.
```

Boundary:

```text
The method assumes generator-provided nuisance counterfactuals and fails the
primary mitigation gate.
```

Reviewer attack answered:

```text
Are method claims stronger than the actual robustness evidence?
```

## Claim C7: The Paper Fits Neurocomputing As A Learning-Systems Study

Plain meaning:

```text
The contribution must be framed as a neural learning-systems failure mode and
benchmark protocol, not as a physics claim.
```

Evidence:

```text
Neurocomputing scope includes fundamental contributions to neural networks and
learning systems, including architectures, learning methods, and analysis of
network dynamics.
```

Required visual/table:

```text
Table 1: relation to prior work and scope
```

Allowed wording:

```text
The work studies a failure mode of neural sequence learning under competing
irreversible mechanisms.
```

Forbidden wording:

```text
The paper explains the nature of time.
The paper is primarily a thermodynamics paper.
```

Boundary:

```text
The physics intuition motivates the benchmark. The contribution must remain a
learning-systems contribution.
```

Reviewer attack answered:

```text
Why is this in Neurocomputing rather than a physics or application journal?
```

## Claim C8: Learned Scores Are Not Physical Entropy Production

Plain meaning:

```text
The manuscript must not overstate learned trajectory evidence as physical heat
or exact thermodynamic entropy production.
```

Technical statement:

```text
The benchmark uses irreversible dynamics and arrow-like signals, but the neural
results are predictive and diagnostic, not physical thermodynamic measurements.
```

Required location:

```text
Problem Formulation or Limitations
```

Allowed wording:

```text
irreversible mechanism
arrow-like nuisance
directed nuisance process
trajectory-level asymmetric evidence
```

Forbidden wording:

```text
physical heat dissipation measured by the network
exact entropy production learned by the classifier
thermodynamic causality recovered by ERM
```

Boundary:

```text
Only analytic or explicitly defined physical systems may use physical entropy
production terminology.
```

Reviewer attack answered:

```text
Is the paper making invalid physics claims?
```

## Main Claim Wording

Safe one-sentence version:

```text
In a controlled irreversible inverse-inference benchmark, neural sequence ERM
can rely on a non-causal irreversible nuisance arrow when that arrow is
correlated with the label during training, despite the task-causal core signal
being learnable.
```

Unsafe stronger version:

```text
Neural networks generally maximize the wrong arrow of time.
```
