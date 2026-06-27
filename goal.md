# Active Goal: Neurocomputing Full Manuscript and Submission Package

This is the only active `goal.md`.

The manuscript design package is complete. The next phase is to write the full
Neurocomputing manuscript from the locked plan, evidence, and style rules. The
work must produce a formal, compact, visually polished, submission-ready paper
draft without overclaiming, weak English, generic generated prose, low-quality
figures, or stray repository artifacts.

This goal is not another benchmark redesign. It is not a loose draft-writing
exercise. It is the full manuscript construction pass.

The target quality is a corresponding-author handoff draft: the corresponding
author should be able to read the PDF and source package as a serious
submission candidate, with only author-confirmation items left outside the
manuscript body.

## /goal Objective

```text
Write and assemble the full Neurocomputing manuscript for the spurious-arrow
project from the existing claim-evidence matrix, manuscript plan, figure/table
plan, writing style guide, submission risk audit, and length budget. Produce a
complete elsarticle paper in paper/neurocomputing/main.tex with the locked
title and author metadata, formal compact prose, verified references, polished
manuscript-grade figures and tables, clearly bounded claims, highlights,
keywords, declarations, data/code availability, and a clean compiled PDF. Use
logged final results only; do not invent results or exceed the allowed claim
boundary. Generate or regenerate figures from result artifacts rather than
manual editing, keep visual quality suitable for journal review, separate main
paper content from supplement/repository detail, and leave the directory clean
enough that the next step is final human review and submission-system upload,
not another design pass.
```

Short command form:

```text
Write the complete Neurocomputing paper from the locked evidence and plan:
main.tex, figures, tables, refs, PDF, and submission checklist, with strict
claim, style, visual, and hygiene controls.
```

## Locked Title And Authors

Use this title unless the user explicitly changes it:

```text
Spurious Arrows of Time in Irreversible Inverse Inference
```

Use this author metadata exactly:

```latex
\author[ulsan-eece]{YoungJae Cho}
\ead{gongjae00@mail.ulsan.ac.kr}

\author[ulsan-ict]{DaeYeol Kim\corref{cor1}}
\ead{daeyeol@ulsan.ac.kr}

\cortext[cor1]{Corresponding author.}

\affiliation[ulsan-eece]{
  organization={Department of Electrical, Electronic and Computer Engineering, University of Ulsan},
  addressline={93 Daehak-ro, Nam-gu},
  city={Ulsan},
  postcode={44610},
  country={Republic of Korea}
}

\affiliation[ulsan-ict]{
  organization={School of ICT Convergence, University of Ulsan},
  addressline={93 Daehak-ro, Nam-gu},
  city={Ulsan},
  postcode={44610},
  country={Republic of Korea}
}
```

Do not alter author order, affiliation text, or corresponding author metadata
without explicit user instruction.

## Source Of Truth

Use these documents as binding instructions:

```text
RESEARCH.md
docs/claim_evidence_matrix.md
docs/manuscript_plan.md
docs/figure_table_plan.md
docs/writing_style_guide.md
docs/submission_risk_audit.md
docs/manuscript_length_budget.md
docs/literature_matrix.md
docs/novelty_gap.md
```

Use these result artifacts as binding evidence:

```text
results/main_experiments/full/summary.md
results/main_experiments/full/summary.json
results/main_experiments/full/final_gate_audit.md
results/main_experiments/full/claim_audit.md
results/main_experiments/full/method_table.md
results/main_experiments/full/scenario_table.md
```

Do not use memory, expected results, smoke runs, or undocumented numbers as
manuscript evidence.

## Current Claim Boundary

Allowed main claim:

```text
In a controlled irreversible inverse-inference benchmark, neural sequence ERM
can rely on a non-causal irreversible nuisance arrow when that arrow is
correlated with the label during training, despite the task-causal core signal
being learnable.
```

Allowed secondary claim:

```text
Counterfactual nuisance replacement improves mean OOD accuracy, but it is not
seed-stable enough to be claimed as a solved method.
```

Required numerical anchors:

```text
main_spurious_arrow, n=10:
  sequence_erm:
    IID test accuracy: 0.971
    OOD test accuracy: 0.124
    OOD gap: 0.848

  final_frame_mlp:
    IID test accuracy: 0.501
    OOD test accuracy: 0.500
    OOD gap: 0.000

  nuisance_only_oracle:
    IID test accuracy: 0.969
    OOD test accuracy: 0.031
    OOD gap: 0.938

  core_only_oracle:
    IID test accuracy: 1.000
    OOD test accuracy: 1.000
    OOD gap: 0.000

  counterfactual_invariance:
    IID test accuracy: 0.943
    OOD test accuracy: 0.756
    OOD gap: 0.187
    OOD-threshold seed success: 7/10

no_spurious_correlation:
  sequence_erm:
    IID test accuracy: 0.898
    OOD test accuracy: 0.899
    OOD gap: -0.001
    OOD-threshold seed success: 8/10
```

Forbidden claims:

```text
we solved selective irreversibility
the counterfactual method is seed-stable
the result proves a universal principle of time
the model measures physical entropy production or heat dissipation
the method transfers directly to real data without counterfactual interventions
the benchmark alone establishes a complete new learning paradigm
```

## End State

The goal is complete only when all deliverables below exist and pass the final
audit.

Required manuscript files:

```text
paper/neurocomputing/main.tex
paper/neurocomputing/main.pdf
paper/neurocomputing/refs.bib
paper/neurocomputing/cover_letter.md
paper/neurocomputing/submission_checklist.md
paper/neurocomputing/handoff_note.md
```

Required optional files if needed:

```text
paper/neurocomputing/supplement.tex
paper/neurocomputing/supplement.pdf
paper/neurocomputing/graphical_abstract.*
```

Required generated visual files:

```text
paper/neurocomputing/fig01_conceptual_problem.pdf
paper/neurocomputing/fig02_benchmark_construction.pdf
paper/neurocomputing/fig03_evidence_gates.pdf
paper/neurocomputing/fig04_main_results.pdf
paper/neurocomputing/fig05_scenario_audit.pdf  # only if promoted to main
```

Required figure-generation code if figures are generated during this goal:

```text
src/visualization/paper_figures.py
```

Required main manuscript content:

```text
title and author metadata
abstract
highlights
keywords
introduction
related work
problem formulation
benchmark and protocol
models and evaluation
results
discussion and limitations
conclusion
CRediT authorship contribution statement drafted without a bare placeholder
declaration of competing interest
data and code availability
verified references
```

No bare placeholders are allowed in `main.tex` at completion.

Forbidden completion-state strings in manuscript files:

```text
TODO
TBD
To be completed
placeholder
Abstract text
Research highlight
\section{}
\item
```

If an item genuinely requires author confirmation, place the issue in
`submission_checklist.md` and `handoff_note.md`, not as a visible placeholder in
the manuscript body.

For CRediT, draft a conservative statement only when supportable from the
project record. If contribution roles require author confirmation, include a
clean provisional statement in the manuscript and list the confirmation need in
the handoff note.

Do not mark complete if the PDF compiles but the writing is generic, visually
weak, overlong, under-cited, overclaimed, contains placeholders, or is missing
required declarations.

## Length Budget

Use the internal Neurocomputing manuscript budget:

```text
main text target: 6,000-7,500 words
internal ceiling before supplement: 8,500 words
abstract: 180-220 words
highlights: 3-5 bullets, each <= 85 characters unless current guide differs
keywords: 4-6
main figures: 4-5
main tables: 2-3
references: 35-60 verified and relevant
```

Section target:

```text
Introduction: 900-1,100
Related Work: 900-1,200
Problem Formulation: 700-900
Benchmark and Protocol: 1,000-1,300
Models and Evaluation: 700-900
Results: 1,200-1,500
Discussion and Limitations: 900-1,100
Conclusion: 150-250
```

If the manuscript exceeds the ceiling, cut repetition, generic related-work
summary, secondary diagnostics, and long limitation prose before cutting
evidence required for the main claim.

## Writing Style Lock

The manuscript must sound like a technical journal article.

Required style:

```text
formal
compact
specific
evidence-constrained
claim-by-claim
not apologetic
not philosophical
not corporate
not AI-generated
```

Prose rules:

```text
One paragraph, one role.
One sentence, one claim where possible.
Review every sentence longer than 25-30 words.
Avoid generic "This paper aims to..." boilerplate.
Do not use "novel" as a substitute for explaining the gap.
Do not use "significant" unless statistical significance is tested.
Use "we show" sparingly and only where gates support it.
Use "mitigates" or "improves mean OOD" for counterfactual; do not use "solves".
```

Before accepting any paragraph, check:

```text
Does it make exactly one point?
Does a figure/table/evidence item support it if it claims a result?
Can one sentence be deleted without loss?
Does it repeat an earlier section?
Does it sound like a journal article rather than a book, blog, or report?
```

## Manuscript Structure

Follow `docs/manuscript_plan.md`.

Main section order:

```text
1. Introduction
2. Related Work
3. Problem Formulation
4. Benchmark and Protocol
5. Models and Evaluation
6. Results
7. Discussion and Limitations
8. Conclusion
```

Write in evidence-first order, not document order:

```text
1. Figures and captions
2. Results
3. Benchmark and protocol
4. Problem formulation
5. Related work
6. Introduction
7. Discussion and limitations
8. Abstract
9. Highlights
10. Conclusion
```

Reason:

```text
The manuscript must be constrained by evidence before motivation is polished.
```

## Figure And Table Requirements

Figures are central evidence objects, not decoration.

Required figure plan:

```text
Figure 1: Conceptual problem diagram
Figure 2: Benchmark construction and visual example
Figure 3: Evidence gates and oracle controls
Figure 4: Main neural results with seed visibility
Figure 5: Scenario/robustness audit only if compact enough for main text
```

Required table plan:

```text
Table 1: Related-work positioning
Table 2: Benchmark/evaluation protocol
Table 3: Main results
```

Visual quality requirements:

```text
vector PDF/SVG preferred
300-600 dpi PNG only when raster is necessary
consistent fonts and method colors
color-blind-safe palette
no rainbow colormap
no default matplotlib styling
no screenshots
no cramped legends
axis labels readable at journal column width
panel labels A, B, C... where applicable
caption states the scientific takeaway
```

PDF readability checks:

```text
inspect compiled PDF at 100% zoom
inspect compiled PDF at 150% zoom
inspect single-column scaled figure width
verify axes, legends, colorbars, labels, and table numbers remain readable
reject any visual that requires private verbal explanation
```

Figure implementation rules:

```text
Read data from logged artifacts.
Do not hard-code result values except labels/thresholds.
Export both PDF and PNG if practical.
Keep final submission files flat in paper/neurocomputing.
Do not manually edit generated figures outside code.
```

## Results And Uncertainty Reporting

Use this convention:

```text
Main text: report mean over 10 seeds, with n=10 stated.
Tables: report mean +- standard deviation over 10 seeds.
Figures: show mean plus seed markers where seed instability matters.
Counterfactual panel: must show seed-level behavior, not only mean bars.
```

If using `±` in LaTeX, write it as:

```latex
$0.971 \pm 0.009$
```

Do not hide failed seed-stability gates behind mean performance.

Do not mix standard deviation, standard error, and confidence intervals unless
explicitly labeled. Default is standard deviation across seeds.

## References And Literature

Before writing related work, verify references from primary or authoritative
sources. Use current sources for venue/submission claims if needed.

Reference requirements:

```text
35-60 verified and relevant references
primary papers preferred
official venue pages for submission/scope facts
no fabricated bibliographic entries
no citations included only for padding
refs.bib entries must compile
```

Minimum literature groups:

```text
arrow-of-time and temporal-order learning
temporal self-supervision shortcuts
spurious correlation and OOD generalization
time-series OOD
entropy-production or thermodynamic-arrow learning
inverse diffusion / source localization
neural sequence models or learning-system analysis as needed
```

Do not claim broad firstness unless verified. The novelty claim is the
controlled intersection:

```text
irreversible inverse inference with competing task-causal and non-causal
irreversible mechanisms, endpoint/residue audits, and OOD nuisance-arrow
reversal.
```

## Supplement Boundary

Keep the main paper compact.

Main paper must include:

```text
main_spurious_arrow result
no_spurious_correlation core-learnability control
core-only and nuisance-only oracle controls
final-frame endpoint audit
counterfactual partial mitigation with seed-stability caveat
```

Supplement or repository documentation should carry:

```text
full seed-by-seed result tables
smoke and pilot diagnostics
extra scenario heatmaps
full hyperparameter grids
implementation details beyond reproducibility essentials
additional visual examples
negative controls beyond the main proof chain
```

Do not move a failed gate to supplement if the main text relies on the related
claim.

## Submission Metadata And Ethics

Required in manuscript:

```text
CRediT authorship contribution statement
Declaration of competing interest
Data and code availability statement
```

Required outside manuscript:

```text
cover_letter.md:
  concise Neurocomputing cover letter
  no hype
  no claims stronger than the manuscript
  clear fit to neural sequence learning / learning systems
  note that the work studies a controlled benchmark and failure mode

submission_checklist.md:
  final upload/readiness checklist

handoff_note.md:
  concise note for the corresponding author
  what is ready
  what needs author confirmation
  what claims are intentionally bounded
```

AI-assisted writing disclosure:

```text
The author will handle this manually and honestly before submission. Do not
auto-generate disclosure text unless the user explicitly requests it.
```

Before finalizing, verify current Neurocomputing/Elsevier requirements:

```text
article type
highlight count/length
graphical abstract requirements
declaration requirements
LaTeX upload requirements
```

Do not make a submission-rule claim from memory if it could have changed.

## Paper Workspace Hygiene

Rules:

```text
paper/neurocomputing should contain only active manuscript/submission files.
Keep final TeX, class, bst, bib, PDF, figures, optional supplement, and checklist.
Remove aux/log/out/spl/bbl/blg clutter after builds unless needed for upload.
Do not leave final2/v2/old/draft duplicate manuscripts.
Do not leave copied result figures that are not part of the selected figure set.
Do not put prompts or private planning notes in paper/neurocomputing.
```

Because Elsevier Editorial Manager may not process LaTeX submissions containing
subfolders, keep final source and figure files flat under `paper/neurocomputing`
unless official instructions say otherwise.

## Execution Phases

### Phase 1: Preflight Audit

Verify:

```text
paper/neurocomputing/main.tex has locked title/authors
docs/claim_evidence_matrix.md exists and matches final results
docs/manuscript_plan.md exists
docs/figure_table_plan.md exists
docs/writing_style_guide.md exists
docs/submission_risk_audit.md exists
docs/manuscript_length_budget.md exists
final results are non-runtime-limited full run
```

Patch any mismatch before writing prose.

### Phase 2: Reference Verification

Build `refs.bib` from verified sources.

Requirements:

```text
no fake entries
no incomplete required fields when avoidable
all cited works appear in refs.bib
all refs.bib entries are cited or intentionally retained for immediate use
```

### Phase 3: Figure And Table Production

Implement or update figure-generation code if needed.

Generate:

```text
fig01_conceptual_problem.pdf
fig02_benchmark_construction.pdf
fig03_evidence_gates.pdf
fig04_main_results.pdf
fig05_scenario_audit.pdf only if promoted to main
```

Create LaTeX tables or table macros:

```text
related-work positioning table
benchmark/evaluation protocol table
main results table
```

Open/inspect figures before accepting them.

### Phase 4: Evidence-First Manuscript Draft

Write in this order:

```text
Results
Benchmark and Protocol
Problem Formulation
Models and Evaluation
Related Work
Introduction
Discussion and Limitations
Conclusion
Abstract
Highlights
Keywords
Declarations
Data/code availability
```

After each section, check:

```text
claim-evidence alignment
word budget
paragraph role
style guide compliance
figure/table references
overclaim risk
```

### Phase 5: Full Compile And Reference Pass

Run:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

or update the Makefile to perform the equivalent build.

Resolve:

```text
undefined citations
undefined references
missing figures
overfull hboxes that harm readability
bibliography style errors
```

### Phase 6: Visual And PDF Review

Inspect the compiled PDF:

```text
title/authors correct
abstract compact
highlights <= 85 characters unless guide differs
figures readable at 100% and 150%
tables not cramped
captions explain scientific takeaway
claim boundary visible in Results/Discussion
counterfactual limitation visible
no stale placeholder text
```

### Phase 7: Submission Package Cleanup

Create:

```text
paper/neurocomputing/cover_letter.md
paper/neurocomputing/submission_checklist.md
paper/neurocomputing/handoff_note.md
```

The cover letter must:

```text
be concise
state title and submission target
state the central contribution without hype
state why the work fits Neurocomputing
avoid "breakthrough", "paradigm shift", or unsupported novelty language
```

The submission checklist must record:

```text
compiled PDF status
word/figure/table budget status
reference status
figure readability status
claim boundary status
supplement decision
items for author manual completion, including AI disclosure if needed
```

The handoff note must record:

```text
final claim in one sentence
why the paper is framed as a phenomenon/benchmark paper
why the counterfactual result is bounded
what the corresponding author should inspect first
manual author items:
  CRediT confirmation
  competing-interest confirmation
  data/code availability wording confirmation
  AI-assisted writing disclosure decision/text
  final Neurocomputing guide compliance check
```

Clean:

```text
aux/log/out/spl clutter
unused figure files
obsolete temporary drafts
```

## Acceptance Checklist

Do not mark complete until every item is true:

```text
[ ] main.tex contains a full manuscript, not a skeleton.
[ ] title and author metadata are correct.
[ ] abstract, highlights, and keywords are filled.
[ ] no placeholder strings remain in main.tex, cover_letter.md, or handoff_note.md.
[ ] all main sections are written in formal compact prose.
[ ] every major claim appears in docs/claim_evidence_matrix.md.
[ ] no forbidden claim appears in manuscript.
[ ] all result numbers come from logged full-run artifacts.
[ ] uncertainty reporting is consistently mean +- seed standard deviation.
[ ] main figures are generated, polished, included, and readable.
[ ] main tables are included and not overloaded.
[ ] Figure 5 is either included for a clear narrative reason or moved out of the main paper.
[ ] references are verified, cited, and compile.
[ ] declarations and data/code availability are present and not placeholders.
[ ] cover_letter.md exists and does not overclaim.
[ ] handoff_note.md exists and clearly lists remaining author-confirmation items.
[ ] counterfactual result is framed as mitigation diagnostic, not solved method.
[ ] physical entropy-production overclaim is absent.
[ ] PDF compiles cleanly enough for review.
[ ] the full PDF has been checked for title/authors, figure readability, table readability, and stale text.
[ ] paper/neurocomputing contains no build clutter or duplicate draft artifacts.
[ ] submission_checklist.md records remaining author/manual upload items.
```

## Final Response Required From Agent

At the end of the `/goal` run, report:

```text
1. Files created/updated.
2. Final title and author metadata status.
3. Manuscript word/figure/table counts.
4. Included figures and tables.
5. Reference count and compile status.
6. PDF compile status and location.
7. Any manual author items remaining before upload.
```
