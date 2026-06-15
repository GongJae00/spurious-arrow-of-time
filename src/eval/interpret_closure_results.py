"""Interpret the focused closure experiment without expanding the main study.

The closure experiment asks a narrow causal question:

  Are OOD failures tied to a dynamic spurious-arrow shift, rather than generic
  shortcut artifacts?

It is not a clean-factorization success gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from src.train.common import save_json


SCHEMA = "closure_result_interpretation_v1"
MAIN_BENCHMARKS = {
    "sta": "sta/aggregate.json",
    "ink_advection_diffusion": "ink_advection_diffusion/aggregate.json",
}
CONDITIONS = (
    "closure_spurious_causality/correlated_reversed_ood",
    "closure_spurious_causality/correlated_no_shift",
    "closure_spurious_causality/randomized_no_shortcut",
)
PRIMARY_METHODS = ("erm", "sib", "sid", "itm")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _method_metric(method_payload: Mapping[str, Any], metric: str) -> float | None:
    return _float_or_none(method_payload.get(f"{metric}_mean"))


def _condition_metrics(aggregate: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    by_condition = aggregate.get("by_condition", {})
    out: dict[str, dict[str, dict[str, Any]]] = {}
    if not isinstance(by_condition, Mapping):
        return out
    for condition in CONDITIONS:
        payload = by_condition.get(condition)
        if not isinstance(payload, Mapping):
            continue
        out[condition] = {
            method: dict(method_payload)
            for method, method_payload in payload.items()
            if method in PRIMARY_METHODS and isinstance(method_payload, Mapping)
        }
    return out


def _extract_table(
    benchmark: str,
    aggregate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, methods in _condition_metrics(aggregate).items():
        condition_name = condition.rsplit("/", 1)[-1]
        for method, payload in methods.items():
            iid = _method_metric(payload, "iid_test_accuracy")
            ood = _method_metric(payload, "ood_test_accuracy")
            gap = _method_metric(payload, "ood_gap")
            rows.append(
                {
                    "benchmark": benchmark,
                    "condition": condition_name,
                    "condition_key": condition,
                    "method": method,
                    "n_runs": payload.get("n_runs"),
                    "iid_test_accuracy_mean": iid,
                    "ood_test_accuracy_mean": ood,
                    "ood_gap_mean": gap,
                }
            )
    return rows


def _row_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (row["benchmark"], row["condition"], row["method"]): row
        for row in rows
    }


def _mean_values(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(mean(float(value) for value in values))


def _closure_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = _row_lookup(rows)
    per_benchmark: dict[str, dict[str, Any]] = {}
    for benchmark in MAIN_BENCHMARKS:
        trap_erm = lookup.get((benchmark, "correlated_reversed_ood", "erm"), {})
        no_shift_erm = lookup.get((benchmark, "correlated_no_shift", "erm"), {})
        randomized_erm = lookup.get((benchmark, "randomized_no_shortcut", "erm"), {})
        trap_sid = lookup.get((benchmark, "correlated_reversed_ood", "sid"), {})
        trap_sib = lookup.get((benchmark, "correlated_reversed_ood", "sib"), {})
        erm_trap_gap = trap_erm.get("ood_gap_mean")
        erm_no_shift_gap = no_shift_erm.get("ood_gap_mean")
        erm_random_gap = randomized_erm.get("ood_gap_mean")
        sid_trap_ood = trap_sid.get("ood_test_accuracy_mean")
        sib_trap_ood = trap_sib.get("ood_test_accuracy_mean")
        erm_trap_ood = trap_erm.get("ood_test_accuracy_mean")
        reasons: list[str] = []
        if erm_trap_gap is None or erm_no_shift_gap is None or erm_random_gap is None:
            reasons.append("missing ERM closure gap metrics")
        else:
            if float(erm_trap_gap) <= float(erm_no_shift_gap) + 0.10:
                reasons.append(
                    "ERM trap gap is not clearly larger than no-shift control"
                )
            if float(erm_trap_gap) <= float(erm_random_gap) + 0.10:
                reasons.append(
                    "ERM trap gap is not clearly larger than randomized control"
                )
        if sid_trap_ood is None or erm_trap_ood is None:
            reasons.append("missing SID/ERM trap OOD metrics")
        elif float(sid_trap_ood) <= float(erm_trap_ood) + 0.10:
            reasons.append("SID does not clearly improve over ERM in the trap condition")
        per_benchmark[benchmark] = {
            "passed": not reasons,
            "reasons": reasons,
            "erm_trap_ood_gap": erm_trap_gap,
            "erm_no_shift_ood_gap": erm_no_shift_gap,
            "erm_randomized_ood_gap": erm_random_gap,
            "erm_trap_ood": erm_trap_ood,
            "sib_trap_ood": sib_trap_ood,
            "sid_trap_ood": sid_trap_ood,
        }
    return {
        "passed": all(item["passed"] for item in per_benchmark.values()),
        "per_benchmark": per_benchmark,
        "interpretation": (
            "This gate supports a diagnostic closure only: a larger ERM gap in "
            "the correlated/reversed condition than in controls is evidence that "
            "the failure is tied to dynamic spurious irreversibility. It does not "
            "license a clean SID factorization claim."
        ),
    }


def _conditional_audit_summary(root: Path) -> dict[str, Any] | None:
    path = root / "sid_conditional_factor_audit_summary.json"
    if not path.exists():
        return None
    report = _load_json(path)
    out: dict[str, Any] = {
        "path": str(path),
        "schema": report.get("schema"),
        "passed": report.get("passed"),
        "interpretation_lock": report.get("interpretation_lock"),
        "selected_metrics": {},
    }
    aggregate = report.get("aggregate", {})
    if isinstance(aggregate, Mapping):
        for key in (
            "iid_test.task_rep.spurious_dynamic.residualized_orientation_free_auc",
            "ood_test.task_rep.spurious_dynamic.residualized_orientation_free_auc",
            "ood_test.z_ir_spur.spurious_dynamic.raw_orientation_free_auc",
            "ood_test.z_ir_spur.spurious_dynamic.residualized_orientation_free_auc",
        ):
            if key in aggregate:
                out["selected_metrics"][key] = aggregate[key]
    return out


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Closure Result Interpretation",
        "",
        "This report summarizes the focused closure experiment. It is intentionally",
        "limited to STA-Bench and Ink Advection-Diffusion.",
        "",
        f"- result_root: `{report.get('root')}`",
        f"- passed: `{report.get('passed')}`",
        f"- claim_mode: `{report.get('claim_mode')}`",
        "",
        "## Closure Gate",
        "",
    ]
    gate = report.get("closure_gate", {})
    lines.append(f"- passed: `{gate.get('passed')}`")
    for benchmark, payload in gate.get("per_benchmark", {}).items():
        lines.extend(
            [
                "",
                f"### {benchmark}",
                "",
                f"- ERM trap OOD gap: `{payload.get('erm_trap_ood_gap')}`",
                f"- ERM no-shift OOD gap: `{payload.get('erm_no_shift_ood_gap')}`",
                f"- ERM randomized OOD gap: `{payload.get('erm_randomized_ood_gap')}`",
                f"- ERM trap OOD: `{payload.get('erm_trap_ood')}`",
                f"- SIB trap OOD: `{payload.get('sib_trap_ood')}`",
                f"- SID trap OOD: `{payload.get('sid_trap_ood')}`",
                f"- passed: `{payload.get('passed')}`",
            ]
        )
        reasons = payload.get("reasons") or []
        if reasons:
            lines.append("- reasons:")
            lines.extend(f"  - {reason}" for reason in reasons)
        else:
            lines.append("- reasons: none")
    lines.extend(
        [
            "",
            "## Claim Lock",
            "",
            "Use this closure evidence to discuss the spurious-arrow failure mode",
            "and counterfactual robustness. Do not claim that SID learns a clean",
            "reversible/task-irreversible/spurious-irreversible factorization unless",
            "separate factor-role gates support that statement.",
            "",
        ]
    )
    return "\n".join(lines)


def interpret_closure_results(
    root: str | Path,
    *,
    output: str | Path | None = None,
    markdown_output: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for benchmark, rel_path in MAIN_BENCHMARKS.items():
        path = root / rel_path
        if not path.exists():
            missing.append(str(path))
            continue
        rows.extend(_extract_table(benchmark, _load_json(path)))
    gate = _closure_gate(rows)
    report = {
        "schema": SCHEMA,
        "root": str(root),
        "passed": bool(not missing and gate.get("passed") is True),
        "claim_mode": "diagnostic_closure_evidence",
        "main_benchmarks": list(MAIN_BENCHMARKS),
        "appendix_diagnostic_benchmarks": [],
        "missing_inputs": missing,
        "rows": rows,
        "closure_gate": gate,
        "conditional_factor_audit": _conditional_audit_summary(root),
        "mean_trap_sid_ood": _mean_values(
            [
                row
                for row in rows
                if row["condition"] == "correlated_reversed_ood" and row["method"] == "sid"
            ],
            "ood_test_accuracy_mean",
        ),
        "interpretation_lock": (
            "Closure evidence may support a spurious-arrow failure-mode claim and "
            "counterfactual robustness discussion. It must not be used to claim "
            "clean SID factorization."
        ),
    }
    output_path = Path(output) if output else root / "closure_result_interpretation.json"
    markdown_path = (
        Path(markdown_output)
        if markdown_output
        else root / "closure_result_interpretation.md"
    )
    save_json(output_path, report)
    _write_text(markdown_path, _markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpret focused closure results.")
    parser.add_argument("--root", default="results/closure_spurious_causality")
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    report = interpret_closure_results(
        args.root,
        output=args.output,
        markdown_output=args.markdown_output,
    )
    print(json.dumps({"passed": report["passed"], "claim_mode": report["claim_mode"]}))


if __name__ == "__main__":
    main()
