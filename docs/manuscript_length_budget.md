# Manuscript Length Budget

This budget is an internal target for a compact Neurocomputing submission. If
the current official Guide for Authors states a stricter requirement at
submission time, the official requirement overrides this document.

## Global Budget

```text
main text target: 6,000-7,500 words
internal ceiling before supplement: 8,500 words
abstract: 180-220 words
highlights: 3-5 bullets, each <= 85 characters
keywords: 4-6
main figures: 4-5
main tables: 2-3
references: 35-60, verified and relevant
supplement: detailed configs, extra seeds, long ablations, extra diagnostics
```

Rationale:

```text
The paper should be read as a sharp learning-systems contribution, not as a
project report. Extra diagnostics should support the claim without bloating the
main narrative.
```

## Section Budget

| Section | Target words | Hard local ceiling | Purpose |
|---|---:|---:|---|
| Abstract | 180-220 | 250 | Problem, setup, result, boundary |
| Introduction | 900-1,100 | 1,250 | Convert intuition into a precise learning problem |
| Related Work | 900-1,200 | 1,350 | Differentiate, not list |
| Problem Formulation | 700-900 | 1,000 | Define the setting and metrics |
| Benchmark and Protocol | 1,000-1,300 | 1,500 | Prove the benchmark is controlled |
| Models and Evaluation | 700-900 | 1,050 | Explain why each baseline exists |
| Results | 1,200-1,500 | 1,700 | Answer claim-by-claim |
| Discussion and Limitations | 900-1,100 | 1,300 | Meaning, boundaries, risks |
| Conclusion | 150-250 | 300 | Close without new claims |

## Figure And Table Budget

Main paper:

```text
Figure 1: conceptual problem diagram
Figure 2: benchmark construction and visual example
Figure 3: evidence gates and oracle controls
Figure 4: main neural results
Figure 5: scenario/robustness audit, if it earns space

Table 1: relation to prior work
Table 2: benchmark/evaluation protocol
Table 3: main results
```

Rules:

```text
Do not include a figure only because it exists.
Do not include a table if one sentence can state the point cleanly.
Do not duplicate the same information in a figure and a table unless they serve
different reviewer questions.
```

## Supplement Boundary

Move to supplement or repository documentation:

```text
full seed-by-seed tables
all smoke and pilot runs
full hyperparameter grids
extra scenario heatmaps
implementation details beyond reproducibility essentials
additional visual examples
long negative-control appendix
```

Keep in main paper:

```text
main_spurious_arrow result
no_spurious_correlation core-learnability control
core-only and nuisance-only oracle controls
final-frame endpoint audit
counterfactual mitigation with seed-stability caveat
```

## Highlights Budget

Prepare 3-5 highlights. Each must be <= 85 characters including spaces.

Source note:

```text
Elsevier highlight guidance commonly specifies 3-5 highlights with a maximum of
85 characters per highlight unless a journal-specific guide states otherwise.
Confirm the current Neurocomputing Guide for Authors before submission.
```

Rules:

```text
Each highlight should state a concrete contribution or result.
Do not use vague adjectives.
Do not claim method success if seed stability fails.
Do not repeat the title.
```

Draft candidates to refine later:

```text
Irreversible nuisance arrows can mislead neural sequence predictors.
The benchmark separates causal core dynamics from nuisance arrows.
Endpoint matching controls final-frame shortcut leakage.
ERM fails OOD when the nuisance arrow reverses.
Counterfactual replacement helps but is not seed-stable.
```

## Graphical Abstract Decision

Before submission, verify whether the current Neurocomputing Guide for Authors
requires or strongly encourages a graphical abstract.

If required:

```text
Create a separate graphical abstract from the Figure 1 visual language.
Show core irreversible cause, spurious arrow, train correlation, and OOD
reversal in one glance.
Use minimal text.
Do not include result bars or equations.
Export in the required size and file type.
```

If not required:

```text
Do not create one unless it improves editor comprehension.
```

## Cut Order If Over Budget

Cut in this order:

```text
1. repeated motivation
2. generic related-work summaries
3. secondary diagnostic baselines
4. extra scenario descriptions
5. redundant table columns
6. long limitation paragraphs
```

Do not cut:

```text
core learnability evidence
nuisance shortcut evidence
endpoint final-frame audit
counterfactual seed-stability caveat
```
