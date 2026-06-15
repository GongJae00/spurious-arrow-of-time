# Spurious Arrow of Time

This repository studies a sequence-learning failure mode:

```text
an irreversible nuisance process can become a shortcut.
```

The central question is whether a model can learn the transition mechanism that
remains reliable when a stronger but non-causal arrow of time changes under
distribution shift.

## Core Idea

Many temporal signals are direction-sensitive. A model can learn a nuisance
flow, drift, or expansion direction because it predicts the label in training.
That signal can fail when the nuisance dynamics reverse or randomize at test
time.

This project separates:

```text
task mechanism:
  the transition process that carries label-relevant information

nuisance mechanism:
  an irreversible process correlated with the label in train/IID splits
  but unreliable under OOD shift
```

## Methods

The primary method is `itm`:

```text
Invariant Transition Mechanism
```

ITM learns core and nuisance transition mechanisms and tests them with
counterfactual pairs that preserve the core trajectory and label while changing
nuisance dynamics.

Baselines and diagnostics:

```text
erm
ib
ep_min
ep_max
ocp_style
lens_like_arrow_classifier
sib
sid
itm
```

SIB and SID are diagnostic selective baselines. ITM is the primary method for
the current study.

## Benchmarks

```text
STA-Bench:
  biased-ring Markov control with analytic dynamics

Ink Advection-Diffusion:
  passive-scalar sequence task with visible core and nuisance fields
```

Both benchmarks use:

```text
train
val_iid
iid_test
ood_test
```

`val_iid` is for model selection. `iid_test` and `ood_test` are final reporting
splits.

## Run

Install dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest -q
```

Run the smoke suite:

```bash
PYTHON_BIN=python \
OUT=results/smoke_run \
DEVICE=cpu \
RUN_TESTS=1 \
CLEAN_OUT=1 \
bash experiments/smoke_suite.sh
```

Run the full suite:

```bash
PYTHON_BIN=python \
OUT=results/full_run \
DEVICE=cuda \
REQUIRE_CUDA_FOR_FULL_RUN=1 \
EPOCHS=50 \
SEEDS="0 1 2 3 4" \
MIN_SEEDS=5 \
MIN_EPOCHS=25 \
RUN_TESTS=1 \
CLEAN_OUT=1 \
bash experiments/full_suite.sh
```

The full suite runs preflight checks before cleaning or training. To reuse a
passed preflight artifact explicitly:

```bash
PREFLIGHT_OUTPUT=results/full_run/preflight.json
REQUIRE_CUDA_FOR_FULL_RUN=1
```

CPU maintenance preflight is useful for source checks when GPU hardware is not
available. Final evidence runs should keep `REQUIRE_CUDA_FOR_FULL_RUN=1`.

The full suite writes:

```text
evidence_audit.json
result_interpretation.json
itm_mechanism_audit.json
sid_factor_audit.json
aggregate.json
manifest.json
```

`result_interpretation.json` controls result wording. High OOD accuracy
alone is not enough; ITM mechanism evidence is also required.

## Repository Layout

```text
src/data/          benchmark generators
src/models/        models
src/losses/        training objectives
src/train/         training loop
src/eval/          diagnostics, audits, plots
src/experiments/   experiment launcher
configs/           experiment configurations
experiments/       shell entry points
tests/             regression tests
docs/              short method and terminology notes
```

Generated outputs are ignored:

```text
results/
paper/
figures/
*.pt
```

## Scientific Scope

This is a controlled ML study of spurious dynamic irreversibility. It does not
measure physical heat or exact thermodynamic entropy production in learned
latent space.
