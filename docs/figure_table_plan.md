# Figure And Table Plan

This document defines the visual evidence package. Figures and tables are not
decorations; each must remove a specific ambiguity in the manuscript.

## Global Visual Standards

Output formats:

```text
preferred: PDF/SVG vector
acceptable raster: PNG at 300-600 dpi when the content is image-like
```

Design rules:

```text
consistent font family across all figures
consistent color mapping for methods and scenarios
color-blind-safe palette
no rainbow colormap
no screenshot artifacts
no default matplotlib titles
no decorative gradients
no tiny legends
no unexplained acronyms inside figure panels
panel labels A, B, C... with consistent placement
caption states the scientific takeaway
```

Submission packaging:

```text
For final Elsevier/Editorial Manager upload, keep figure files at the same
folder level as the TeX source. Do not rely on nested figure directories.
```

Sizing targets:

```text
single column width: 85 mm
double column width: 170-180 mm
minimum readable text after scaling: 7-8 pt
preferred in-panel text: 8-10 pt
```

PDF checks:

```text
Inspect each figure at 100% zoom.
Inspect each figure at 150% zoom.
Inspect single-column scaled versions.
Check color/grayscale distinguishability.
Reject any visual that requires private verbal explanation.
```

## Figure 1: Conceptual Problem Diagram

Reviewer question:

```text
What is a spurious arrow, and how is it different from ordinary spurious
correlation?
```

Message:

```text
The task-causal process and the nuisance process are both irreversible, but only
the core mechanism determines the label outside the training correlation.
```

Panels:

```text
A. Hidden cause -> irreversible core sequence -> label
B. Independent nuisance arrow -> train correlation with label
C. OOD reversal/removal of nuisance direction
D. Model choice: causal core arrow vs spurious nuisance arrow
```

Data source:

```text
schematic; no result data
```

Caption intent:

```text
Define the paper's central mechanism in one glance.
```

Main or supplement:

```text
main
```

Implementation notes:

```text
Use vector drawing or code-generated schematic.
Avoid philosophical language.
Use minimal labels.
Keep the visual reusable for graphical abstract if required.
```

## Figure 2: Benchmark Construction And Visual Example

Reviewer question:

```text
Is the dataset understandable, and are core/nuisance components genuinely
separated?
```

Message:

```text
The benchmark contains a diffusive core source task and an independent directed
nuisance arrow; the main setting controls final-frame nuisance leakage.
```

Panels:

```text
A. Core-only sequence example
B. Nuisance-only sequence example
C. Mixed observation example
D. Endpoint-matching or final-frame audit illustration
```

Data source:

```text
Generated benchmark examples, preferably from a fixed visualization seed.
```

Caption intent:

```text
Show what the model sees and why endpoint controls matter.
```

Main or supplement:

```text
main
```

Implementation notes:

```text
Do not use raw noisy debug frames that are visually unreadable.
Use a small number of time points.
Annotate direction with clean arrows.
Use consistent intensity scaling across panels.
```

## Figure 3: Evidence Gates And Controls

Reviewer question:

```text
Is the observed ERM failure selection of a shortcut rather than benchmark
occlusion or final-frame leakage?
```

Message:

```text
Core is learnable, nuisance is a strong wrong shortcut, and final-frame leakage
does not explain the endpoint-matched main failure.
```

Panels:

```text
A. no_spurious_correlation: sequence_erm IID/OOD
B. main_spurious_arrow: sequence_erm IID/OOD
C. core_only_oracle and nuisance_only_oracle
D. final_frame_mlp endpoint audit
```

Data source:

```text
results/main_experiments/full/summary.json
results/main_experiments/full/final_gate_audit.md
```

Caption intent:

```text
Make the acceptance gates visible without forcing the reader to parse logs.
```

Main or supplement:

```text
main
```

Implementation notes:

```text
Use mean with seed markers or intervals.
Show n=10.
Avoid overloading the panel with all methods.
```

## Figure 4: Main Neural Results

Reviewer question:

```text
How do the neural models and oracle controls compare on the primary task?
```

Message:

```text
Sequence ERM and nuisance-only oracle collapse OOD; core oracle remains robust;
counterfactual replacement improves mean OOD but is seed-unstable.
```

Panels:

```text
A. IID and OOD accuracy by method
B. OOD gap by method
C. Seed-level counterfactual vs ERM gap reduction
```

Data source:

```text
results/main_experiments/full/summary.json
results/main_experiments/full/method_table.md
```

Caption intent:

```text
State the main result and the method boundary in one figure.
```

Main or supplement:

```text
main
```

Implementation notes:

```text
Do not hide counterfactual seed instability behind only a mean bar.
Use jittered seed points or a small paired-seed panel.
Use consistent method order:
  core_only_oracle
  sequence_erm
  final_frame_mlp
  nuisance_only_oracle
  counterfactual_invariance
```

## Figure 5: Scenario Or Robustness Audit

Reviewer question:

```text
Does the phenomenon depend on a single arbitrary configuration?
```

Message:

```text
The main interpretation is stable across the required control scenarios, while
residue-visible settings show why endpoint controls are necessary.
```

Panels:

```text
A. main_spurious_arrow
B. no_spurious_correlation
C. residue_visible_control
D. optional OOD randomized or partial-shift comparison
```

Data source:

```text
results/main_experiments/full/scenario_table.md
summary.json scenario section
```

Caption intent:

```text
Show boundaries without turning the main paper into a scenario catalog.
```

Main or supplement:

```text
main only if compact; otherwise supplement
```

Implementation notes:

```text
Prefer a small heatmap or grouped bar plot with limited methods.
Do not include every method/scenario combination.
```

## Table 1: Related Work Positioning

Reviewer question:

```text
What is new relative to existing shortcut, OOD, arrow-of-time, and inverse
diffusion work?
```

Columns:

```text
Research line
Typical target
Uses temporal/irreversible evidence?
Has inverse hidden-cause task?
Has competing non-causal irreversible nuisance?
Has endpoint/residue controls?
Gap relative to this work
```

Data source:

```text
docs/literature_matrix.md
docs/novelty_gap.md
verified primary literature
```

Main or supplement:

```text
main
```

Design:

```text
Use short phrases. Avoid long prose in cells.
```

## Table 2: Benchmark And Evaluation Protocol

Reviewer question:

```text
What exactly changes between train, IID, and OOD?
```

Rows:

```text
train
val_iid
iid_test
ood_test
no_spurious_correlation
residue_visible_control
```

Columns:

```text
split/scenario
core-label relation
nuisance-label relation
endpoint control
model-selection use
claim answered
```

Main or supplement:

```text
main
```

## Table 3: Main Results

Reviewer question:

```text
What are the final quantitative results supporting the claim?
```

Rows:

```text
core_only_oracle
sequence_erm
final_frame_mlp
nuisance_only_oracle
counterfactual_invariance
```

Columns:

```text
IID test accuracy
OOD test accuracy
OOD gap
seed success note
interpretation
```

Main or supplement:

```text
main
```

Design:

```text
Use 3 decimals.
Report n=10.
Include seed-stability note for counterfactual.
Do not bold every best value.
```

## Supplement Visuals

Move these outside the main paper unless they become necessary:

```text
full scenario heatmaps
all seed-level tables
smoke/pilot plots
candidate benchmark trial plots
extra visual examples
hyperparameter diagnostics
```

## Figure Implementation Requirements

Future figure scripts should:

```text
read result artifacts from results/main_experiments/full/
avoid hard-coded values except labels and thresholds
export PDF and PNG
share style constants
write reproducible output paths
support a clean command for regeneration
```

Recommended script target:

```text
src/visualization/paper_figures.py
```

Do not implement this script until the figure plan is accepted.
