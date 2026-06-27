# Submission Risk Audit

This audit sharpens the Neurocomputing manuscript before drafting. It should
not make the paper sound defensive; it should prevent weak claims from entering
the paper.

## Venue Fit

Neurocomputing publishes fundamental work on neural networks and learning
systems, including learning methods and analysis of network dynamics. The paper
fits only if it is framed as a neural sequence-learning failure mode with a
controlled benchmark and evidence protocol.

Source note:

```text
Elsevier's Neurocomputing scope page describes the journal as covering
fundamental contributions to neurocomputing, including neural networks and
learning systems, architectures, learning methods, analysis of network dynamics,
and related interdisciplinary learning topics.
```

Submission-preparation note:

```text
Elsevier LaTeX instructions state that Editorial Manager cannot process LaTeX
submissions containing subfolders. Final source files should be kept flat:
main .tex, .bib/.bbl, .bst, .cls, and figure files at one folder level.
```

Safe framing:

```text
learning-systems failure mode
neural sequence predictors
OOD shortcut under irreversible nuisance dynamics
benchmark controls for endpoint leakage and core learnability
```

Unsafe framing:

```text
physics theory of time
generic synthetic dataset paper
standard model applied to a new toy task
method paper claiming solved robustness
```

## Risk 1: "This Is Just Ordinary Spurious Correlation"

Reviewer concern:

```text
The nuisance is just another shortcut feature.
```

Response built into the manuscript:

```text
The nuisance is not a static color or texture. It is an irreversible,
direction-sensitive trajectory process. The task-causal process is also
irreversible, so the controlled question is which arrow the neural sequence
predictor follows.
```

Required evidence:

```text
Figure 1
Figure 2
Table 1
```

Risk if mishandled:

```text
If the introduction only says "spurious correlation", novelty becomes weak.
```

## Risk 2: "This Is Just Temporal OOD Shift"

Reviewer concern:

```text
OOD performance drops because temporal dynamics changed.
```

Response:

```text
The OOD shift specifically reverses or removes the label relation of the
non-causal nuisance arrow while preserving a learnable causal core. The
no_spurious_correlation condition and core oracle distinguish this from generic
temporal covariate shift.
```

Required evidence:

```text
no_spurious_correlation sequence_erm: IID 0.898, OOD 0.899
core_only_oracle: IID/OOD 1.000
```

## Risk 3: "The Core Is Hidden, So Failure Is Trivial"

Reviewer concern:

```text
The model fails because the core is not learnable in mixed observations.
```

Response:

```text
When the nuisance is label-independent, sequence_erm reaches high IID and OOD
accuracy on the mixed input. This supports selection rather than occlusion.
```

Required evidence:

```text
Gate A passed:
  IID 0.898
  OOD 0.899
  OOD gap -0.001
  seed success 8/10
```

Boundary:

```text
State the 8/10 seed success rate. Do not imply perfect learnability in every
seed.
```

## Risk 4: "Final Frame Leakage Explains Everything"

Reviewer concern:

```text
The model may use static endpoint residue rather than trajectory information.
```

Response:

```text
In the endpoint-matched main scenario, final_frame_mlp is near chance with
near-zero OOD gap. A separate residue_visible_control shows that the audit can
detect endpoint leakage when it exists.
```

Required evidence:

```text
main final_frame_mlp: IID 0.501, OOD 0.500, gap 0.000
residue_visible_control final_frame_mlp: IID 0.969, OOD 0.031, gap 0.938
```

Boundary:

```text
Claim endpoint control for the main scenario. Do not claim all variants are free
of residue.
```

## Risk 5: "Counterfactual Method Is Overclaimed"

Reviewer concern:

```text
The paper presents a weak or unstable method as a solution.
```

Response:

```text
Report counterfactual replacement as a mitigation diagnostic. It improves mean
OOD accuracy but fails the seed-stability gate.
```

Required evidence:

```text
counterfactual_invariance:
  IID 0.943
  OOD 0.756
  OOD gap 0.187
  seed success 7/10

mitigation gate passed: False
```

Boundary:

```text
No solved-method claim.
No deployable real-world claim without counterfactual interventions.
```

## Risk 6: "Benchmark Is Too Synthetic"

Reviewer concern:

```text
The experiment is controlled but artificial.
```

Response:

```text
The benchmark is synthetic by design because it isolates task-causal and
spurious irreversible mechanisms. The paper should sell control and diagnostic
clarity, not real-world coverage.
```

What would strengthen the paper:

```text
clear visual examples
strong relation to inverse diffusion/source localization
release-ready code
explicit limitations
supplementary configs and seeds
```

## Risk 7: "Not Enough Neural-Computing Contribution"

Reviewer concern:

```text
The paper is only a dataset or a physics analogy.
```

Response:

```text
Frame the contribution as an analysis of neural sequence predictors under a
specific failure mode. Every baseline answers a learning-systems question.
```

Required manuscript moves:

```text
Explain why sequence_erm, final_frame_mlp, oracles, and counterfactual
replacement are included.
Use neural-model results, not only probes.
Tie scope to learning systems and network dynamics.
```

## Risk 8: "The Literature Claim Is Too Broad"

Reviewer concern:

```text
Prior work already studies arrow of time, spurious correlation, and inverse
diffusion.
```

Response:

```text
Do not claim broad firstness. Claim the controlled intersection: irreversible
inverse inference with a competing non-causal irreversible nuisance, OOD
nuisance-arrow reversal, and endpoint/residue audits.
```

Required table:

```text
Table 1: relation to prior work
```

## Risk 9: "Language Sounds Generated Or Overdramatic"

Reviewer concern:

```text
The paper reads like a manifesto or generic AI-written prose.
```

Response:

```text
Use the writing style guide. Keep prose compact. Let figures and controlled
evidence carry the claim.
```

## Remaining Optional Strengtheners

If time permits before full manuscript writing:

```text
1. Improve final figures to publication quality.
2. Build a concise graphical abstract if required.
3. Add a supplement skeleton for seed tables and configs.
4. Verify references from primary sources.
5. Add a manuscript-readiness checklist for final PDF upload.
```

## Administrative Items To Verify Before Submission

These are not scientific claims, but they must be checked before upload:

```text
current Neurocomputing Guide for Authors
article type
highlight requirements
graphical abstract requirements
declaration of competing interest
CRediT statement
data/code availability statement
ORCID and author metadata
LaTeX source upload structure
```

AI-assisted writing disclosure:

```text
The author will handle this manually and honestly before submission. Do not
auto-generate the disclosure text in manuscript drafting unless explicitly
requested by the author.
```

## Final Safe Scope Statement

Use this as the internal boundary:

```text
This is a controlled neural sequence-learning study showing that a
train-correlated non-causal irreversible nuisance can dominate ERM under OOD
arrow reversal, despite a learnable causal core. The paper introduces the
problem, benchmark, controls, and diagnostic evidence; it does not claim a
fully solved mitigation method or a physical measurement of entropy production.
```
