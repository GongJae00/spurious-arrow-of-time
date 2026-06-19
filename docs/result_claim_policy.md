# Result Claim Policy

## Allowed From Smoke

```text
The benchmark and training scripts execute.
The run is diagnostic only.
```

## Allowed From Pilot

```text
Training dynamics suggest whether the main run is worth launching.
```

Pilot results cannot support a paper-level claim.

## Allowed From Main

Only if gates and controls pass:

```text
Raw neural sequence models can be misled by a spurious irreversible nuisance
arrow in this controlled benchmark.
```

Only if counterfactual results support it:

```text
Counterfactual invariance reduces the OOD gap in this controlled benchmark.
```

## Not Allowed

```text
physical heat dissipation is measured
learned neural scores are exact entropy production
feature probes are neural ERM
a runtime-limited smoke run is a final paper result
method success is claimed when only benchmark diagnostics pass
```

Failures must be preserved and interpreted directly.
