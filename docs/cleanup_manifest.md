# Cleanup Manifest

This manifest records the reset from the previous prototype codebase to the
active irreversible source inference project.

## Kept Files

| Path | Reason |
|---|---|
| `README.md` | Public entry point; rewritten for the reset. |
| `RESEARCH.md` | Scientific brief; rewritten for the reset. |
| `goal.md` | Active execution contract. |
| `pyproject.toml` | Package metadata and test configuration. |
| `requirements.txt` | Simple dependency list. |
| `.gitignore` | Keeps generated outputs out of the public surface. |
| `figures/.gitkeep`, `results/.gitkeep` | Empty placeholders for ignored generated outputs. |

## Rewritten Files

These files are replaced by the reset:

```text
README.md
RESEARCH.md
pyproject.toml
requirements.txt
src/
tests/
configs/
experiments/
```

## Deleted Legacy Categories

The following categories are removed from the active public surface:

```text
previous synthetic benchmark generators, configs, diagnostics, and tests
previous method-first models, losses, trainers, audits, and tests
old EP/arrow-score training loops
old paper drafts and generated manuscript assets
old result summaries and logs
Python caches and local test caches
obsolete runbooks for the previous full suite
```

Git history remains the archive. No large archive directory is kept in the
working tree.

## Extracted Legacy Insights

The previous project contributed these active lessons:

```text
1. A benchmark must prove source ambiguity before method comparisons.
2. Core and nuisance components must be visualized separately and together.
3. Nuisance shortcuts must be dynamic, not static state leaks.
4. Counterfactuals must preserve the core source and change only nuisance.
5. Method claims must be gated by diagnostics, not by desired thesis language.
6. Old method-first prototypes should not drive the benchmark design.
```

## Cleanup Commands

The cleanup removes tracked legacy files and ignored generated artifacts, then
adds the new minimal benchmark files. Exact commands are recorded in the shell
history of this goal run and summarized in the final response.

## Remaining Generated Or Ignored Outputs

Generated outputs are allowed only under ignored paths:

```text
results/
figures/
paper/
logs/
```

The reset smoke command may recreate `results/smoke_benchmark/`.
