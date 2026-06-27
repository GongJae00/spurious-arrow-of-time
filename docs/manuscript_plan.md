# Neurocomputing Manuscript Plan

This plan prepares the full manuscript without writing final prose. The next
writing goal should follow this structure rather than improvise a new paper.

## Working Identity

Target venue:

```text
Neurocomputing
```

Manuscript type:

```text
Research paper / learning-systems benchmark and failure-mode study
```

One-sentence thesis:

```text
In irreversible inverse inference, a neural sequence predictor can rely on a
non-causal irreversible nuisance arrow when it is predictive during training,
even though the task-causal core signal remains learnable.
```

Claim boundary:

```text
Strong on the phenomenon.
Careful on the synthetic benchmark.
Bounded on counterfactual mitigation.
No physical entropy-production claim.
```

## Title Candidates

Preferred:

```text
Spurious Arrows of Time in Irreversible Inverse Inference
```

Alternatives:

```text
When Neural Sequence Models Trust the Wrong Arrow of Time
Spurious Irreversibility in Neural Sequence Learning
Irreversible Inference Under Spurious Temporal Arrows
```

Avoid:

```text
Macroscopic Time, Neural Networks, and the Nature of Irreversibility
Selective Irreversibility Solves Spurious Temporal Learning
```

Reason:

```text
The preferred title names the problem without overclaiming a solution or
turning the paper into a physics manifesto.
```

## Contribution Bullets

The final paper should contain 3-4 contribution bullets:

```text
1. We formulate irreversible inverse inference with a competing non-causal
   irreversible nuisance mechanism.

2. We build an endpoint-matched benchmark that separates core learnability,
   nuisance shortcut strength, and final-frame leakage.

3. We show that sequence ERM can learn IID while collapsing under OOD nuisance
   arrow reversal, even though the causal core remains learnable.

4. We evaluate counterfactual nuisance replacement as a mitigation diagnostic
   and report its seed-stability limitation.
```

Do not claim:

```text
We solve selective irreversibility.
We provide a deployable counterfactual method for real data.
```

## Section Flow

### 1. Introduction

Target length:

```text
900-1,100 words
```

Role:

```text
Turn the ink/broken-glass intuition into a precise neural learning problem.
```

Paragraph plan:

```text
P1: Irreversible observations contain temporal asymmetry, but inverse tasks
    require recovering causes rather than detecting any arrow.

P2: When a label-relevant irreversible process and a non-causal irreversible
    nuisance coexist, the strongest arrow may be the wrong evidence.

P3: Existing temporal-order, OOD, shortcut-learning, and inverse-diffusion work
    do not isolate this exact competition.

P4: Define the controlled question: in mixed observations, does a neural
    sequence predictor trust the causal or spurious irreversible mechanism?

P5: Preview the benchmark controls: no-spurious learnability, core oracle,
    nuisance oracle, endpoint final-frame audit, OOD reversal.

P6: State contributions.
```

Do not say:

```text
Time is not real.
This paper reveals the nature of time.
The problem is solved.
```

### 2. Related Work

Target length:

```text
900-1,200 words
```

Role:

```text
Differentiate the contribution instead of listing papers.
```

Subsections:

```text
2.1 Arrow-of-time and temporal-order representation learning
2.2 Spurious correlations and OOD generalization
2.3 Irreversibility, entropy-production estimation, and dynamical learning
2.4 Inverse diffusion and source localization
```

Required contrast:

```text
Temporal-order work learns or exploits time direction.
Spurious-correlation work studies non-causal predictors.
Inverse-diffusion work recovers hidden causes.
This paper combines them in a controlled setting where both causal and
non-causal mechanisms are irreversible.
```

Do not say:

```text
No prior work has studied time.
No prior work has studied spurious correlation.
We are first, unless a verified literature search supports that exact claim.
```

### 3. Problem Formulation

Target length:

```text
700-900 words
```

Role:

```text
Define the learning problem precisely with minimal equations.
```

Must define:

```text
core trajectory c
nuisance trajectory s
mixed observation x
label y
train/IID correlation
OOD reversal/removal
OOD gap
endpoint leakage
irreversible inverse inference
spurious arrow
```

Suggested equations:

```text
x_t = g(c_t, s_t, epsilon_t)
y = h(c_0:L-1)
OOD gap = Acc_iid_test - Acc_ood_test
```

Do not include:

```text
unnecessary thermodynamics equations
latent entropy-production proxies
method-specific losses unless directly used in final experiments
```

### 4. Benchmark And Protocol

Target length:

```text
1,000-1,300 words
```

Role:

```text
Prove the setup is not a trick.
```

Required subsections:

```text
4.1 Core irreversible source process
4.2 Non-causal nuisance arrow
4.3 Endpoint-matched main scenario
4.4 Splits and OOD shifts
4.5 Acceptance gates and controls
```

Key controls:

```text
core_only_oracle
nuisance_only_oracle
final_frame_mlp
no_spurious_correlation
residue_visible_control
```

Do not bury:

```text
final-frame audit
no-spurious learnability
counterfactual availability assumption
```

### 5. Models And Evaluation

Target length:

```text
700-900 words
```

Role:

```text
Explain why each model answers a specific scientific question.
```

Model roles:

```text
sequence_erm:
  standard neural sequence predictor on mixed observations

final_frame_mlp:
  endpoint/static leakage audit

core_only_oracle:
  upper-bound causal core learnability

nuisance_only_oracle:
  shortcut strength and OOD invalidity

counterfactual_invariance:
  generator-assisted mitigation diagnostic

time_reversed_sequence and group_invariance_light:
  optional diagnostics, not central claims
```

Metrics:

```text
IID test accuracy
OOD test accuracy
OOD gap
seed success rate where used for gates
```

### 6. Results

Target length:

```text
1,200-1,500 words
```

Role:

```text
Answer one claim per subsection.
```

Subsection order:

```text
6.1 The core remains learnable when the nuisance is label-independent.
6.2 A correlated nuisance arrow induces ERM collapse under OOD reversal.
6.3 Endpoint matching rules out final-frame leakage in the main setting.
6.4 Counterfactual replacement mitigates mean collapse but is not seed-stable.
```

Required numbers:

```text
sequence_erm main: IID 0.971, OOD 0.124, gap 0.848
no_spurious sequence_erm: IID 0.898, OOD 0.899, gap -0.001, 8/10 seeds
core_only_oracle: IID/OOD 1.000
nuisance_only_oracle: IID 0.969, OOD 0.031, gap 0.938
final_frame_mlp main: IID 0.501, OOD 0.500, gap 0.000
counterfactual: IID 0.943, OOD 0.756, gap 0.187, 7/10 OOD-threshold seeds
```

Do not narrate:

```text
every smoke run
every pilot run
every scenario unless it changes the claim
```

### 7. Discussion And Limitations

Target length:

```text
900-1,100 words
```

Role:

```text
Explain what the evidence changes and what it does not prove.
```

Required points:

```text
The result sharpens shortcut learning for irreversible sequence tasks.
The benchmark shows selection rather than mere occlusion.
Endpoint control matters.
Counterfactual replacement needs stronger stability before method claims.
Synthetic counterfactuals require interventions/generators in real data.
The work does not measure physical entropy production.
```

Tone:

```text
Honest, not apologetic.
```

### 8. Conclusion

Target length:

```text
150-250 words
```

Role:

```text
Close the paper with the phenomenon and its learning-systems implication.
```

No new claims.

## Main Figures And Tables

Use `docs/figure_table_plan.md` as the authority, but the manuscript structure
expects:

```text
Figure 1 in Introduction or Problem Formulation
Figure 2 in Benchmark And Protocol
Figure 3 in Benchmark/Results transition
Figure 4 in Results
Figure 5 only if scenario audit is needed

Table 1 in Related Work or end of Introduction
Table 2 in Benchmark And Protocol
Table 3 in Results
```

## Writing Order

Do not write linearly from Abstract to Conclusion.

Write in this order:

```text
1. Claim-evidence table
2. Figures and captions
3. Results section
4. Benchmark and protocol
5. Problem formulation
6. Related work
7. Introduction
8. Discussion and limitations
9. Abstract
10. Highlights
11. Conclusion
```

Reason:

```text
The paper should be constrained by evidence before motivation and framing are
polished.
```
