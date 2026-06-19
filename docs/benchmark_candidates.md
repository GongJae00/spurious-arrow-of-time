# Benchmark Candidates

The benchmark must express the original intuition:

```text
The forward process is easy to run, but recovering its cause becomes ambiguous.
Another irreversible process is more visible and becomes a train-only shortcut.
```

## Candidate A: Grid-Graph Diffusion Source Inference

Core:

```text
source on a grid graph
isotropic diffusion over time
delayed/noisy/coarse observation
label is a source attribute
```

Nuisance:

```text
independent directed cascade or traveling pulse
direction correlated with the label in train/IID
direction reversed or randomized in OOD
```

Strengths:

```text
grounded in graph source localization and inverse diffusion
simple oracles
clear counterfactual construction
clear component visualization
fast CPU smoke
```

Risks:

```text
source side can leak through final mass center
directed nuisance can become too obviously artificial
flattened linear models may solve the core too easily if delay is small
```

Verdict:

```text
Use as the main smoke benchmark, with source leakage gates.
```

## Candidate B: Continuous 2D Diffusion Source Inference

Core:

```text
2D heat/diffusion equation from a hidden source
observed after delay and noise
label is source region or source type
```

Nuisance:

```text
separate advection-diffusion plume or moving wave
```

Strengths:

```text
closest to the user's ink-in-water intuition
strong visual explanation
```

Risks:

```text
harder to validate source leakage
more tuning needed for final-frame ambiguity
previous prototype drifted into visible-component separation
```

Verdict:

```text
Use later as a visual companion only if the grid-graph benchmark passes.
```

## Candidate C: Reaction-Diffusion Or Fragmentation Front

Core:

```text
source causes a front, break, or fragmentation process
```

Nuisance:

```text
separate front/cascade with stronger direction statistic
```

Strengths:

```text
can connect to broken glass and irreversible morphology
```

Risks:

```text
too many domain-specific design choices
easy to overfit the story
harder to keep smoke minimal and interpretable
```

Verdict:

```text
Do not use for the reset smoke benchmark.
```
