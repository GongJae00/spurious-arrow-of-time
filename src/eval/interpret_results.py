"""Interpret final results without overclaiming.

The evidence audit checks that the runs exist and obey protocol. This module
adds a separate claim gate: even valid runs may support only a diagnostic or
negative interpretation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.eval.audit_evidence import audit_evidence
from src.eval.audit_smoke import EXPECTED_BENCHMARKS, REQUIRED_METHODS
from src.train.common import save_json


ARROW_PRETRAINING_METHODS = {"ocp_style", "lens_like_arrow_classifier"}
PRIMARY_METHOD = "itm"
PRIMARY_LABEL = "ITM"


@dataclass(frozen=True)
class ClaimThresholds:
    min_primary_iid: float = 0.60
    min_primary_ood: float = 0.60
    min_ood_improvement: float = 0.03
    max_iid_drop_vs_best_non_primary: float = 0.10


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _is_smoke_path(root: Path) -> bool:
    return any("smoke" in part.lower() for part in root.parts)


def _method_payloads(aggregate: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    top_level = {
        method: aggregate[method]
        for method in REQUIRED_METHODS
        if isinstance(aggregate.get(method), Mapping)
    }
    if top_level:
        return top_level, "."

    by_condition = aggregate.get("by_condition", {})
    if isinstance(by_condition, Mapping):
        for condition, payload in sorted(by_condition.items()):
            if not isinstance(payload, Mapping):
                continue
            condition_methods = {
                method: payload[method]
                for method in REQUIRED_METHODS
                if isinstance(payload.get(method), Mapping)
            }
            if condition_methods:
                return condition_methods, str(condition)
    return {}, "."


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metrics(method: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if method in ARROW_PRETRAINING_METHODS:
        iid = payload.get("fine_tuned_encoder_iid_test_accuracy_mean")
        ood = payload.get("fine_tuned_encoder_ood_test_accuracy_mean")
        gap = payload.get("fine_tuned_encoder_ood_gap_mean")
        protocol = "fine_tuned_encoder"
    else:
        iid = payload.get("iid_test_accuracy_mean")
        ood = payload.get("ood_test_accuracy_mean")
        gap = payload.get("ood_gap_mean")
        protocol = "supervised"
    return {
        "method": method,
        "protocol": protocol,
        "n_runs": payload.get("n_runs"),
        "iid_test_accuracy_mean": _float_or_none(iid),
        "ood_test_accuracy_mean": _float_or_none(ood),
        "ood_gap_mean": _float_or_none(gap),
    }


def _best_non_primary(
    rows: Mapping[str, Mapping[str, Any]], metric: str
) -> tuple[str | None, float | None]:
    best_method: str | None = None
    best_value: float | None = None
    for method, metrics in rows.items():
        if method == PRIMARY_METHOD:
            continue
        value = metrics.get(metric)
        if value is None:
            continue
        if best_value is None or float(value) > best_value:
            best_method = method
            best_value = float(value)
    return best_method, best_value


def _benchmark_claim_gate(
    suite_name: str,
    rows: Mapping[str, Mapping[str, Any]],
    sid_factor_passed: bool,
    sid_role_claim_ready: bool,
    sid_role_claim_ready_runs: int,
    sid_role_total_runs: int,
    itm_mechanism_passed: bool,
    itm_mechanism_claim_ready: bool,
    itm_mechanism_claim_ready_runs: int,
    itm_mechanism_total_runs: int,
    thresholds: ClaimThresholds,
) -> dict[str, Any]:
    reasons: list[str] = []
    primary = rows.get(PRIMARY_METHOD)
    if primary is None:
        return {
            "benchmark": suite_name,
            "passed": False,
            "reasons": [f"missing {PRIMARY_LABEL} aggregate metrics"],
        }

    primary_iid = primary.get("iid_test_accuracy_mean")
    primary_ood = primary.get("ood_test_accuracy_mean")
    primary_gap = primary.get("ood_gap_mean")
    best_ood_method, best_non_primary_ood = _best_non_primary(
        rows, "ood_test_accuracy_mean"
    )
    best_iid_method, best_non_primary_iid = _best_non_primary(
        rows, "iid_test_accuracy_mean"
    )

    if not sid_factor_passed:
        reasons.append("SID diagnostic factor audit did not pass")
    if not itm_mechanism_passed:
        reasons.append("ITM mechanism audit did not pass")
    if not itm_mechanism_claim_ready:
        reasons.append(
            "ITM mechanism audit did not authorize mechanism-claim language: "
            f"{itm_mechanism_claim_ready_runs}/{itm_mechanism_total_runs} runs ready"
        )
    if primary_iid is None or primary_iid < thresholds.min_primary_iid:
        reasons.append(
            f"{PRIMARY_LABEL} IID accuracy is below threshold: "
            f"{primary_iid} < {thresholds.min_primary_iid}"
        )
    if primary_ood is None or primary_ood < thresholds.min_primary_ood:
        reasons.append(
            f"{PRIMARY_LABEL} OOD accuracy is below threshold: "
            f"{primary_ood} < {thresholds.min_primary_ood}"
        )
    if best_non_primary_ood is None:
        reasons.append(f"no non-{PRIMARY_LABEL} baseline OOD metric found")
    elif (
        primary_ood is None
        or primary_ood < best_non_primary_ood + thresholds.min_ood_improvement
    ):
        reasons.append(
            f"{PRIMARY_LABEL} OOD improvement is insufficient: "
            f"primary={primary_ood}, best_non_primary={best_non_primary_ood} "
            f"({best_ood_method}), required_margin={thresholds.min_ood_improvement}"
        )
    if best_non_primary_iid is None:
        reasons.append(f"no non-{PRIMARY_LABEL} baseline IID metric found")
    elif (
        primary_iid is None
        or best_non_primary_iid - primary_iid
        > thresholds.max_iid_drop_vs_best_non_primary
    ):
        reasons.append(
            f"{PRIMARY_LABEL} IID drop versus best non-{PRIMARY_LABEL} is too large: "
            f"primary={primary_iid}, best_non_primary={best_non_primary_iid} "
            f"({best_iid_method}), max_drop={thresholds.max_iid_drop_vs_best_non_primary}"
        )

    gate = {
        "benchmark": suite_name,
        "passed": not reasons,
        "primary_method": PRIMARY_METHOD,
        "primary_label": PRIMARY_LABEL,
        "primary_iid_test_accuracy_mean": primary_iid,
        "primary_ood_test_accuracy_mean": primary_ood,
        "primary_ood_gap_mean": primary_gap,
        "best_non_primary_ood_method": best_ood_method,
        "best_non_primary_ood_test_accuracy_mean": best_non_primary_ood,
        "best_non_primary_iid_method": best_iid_method,
        "best_non_primary_iid_test_accuracy_mean": best_non_primary_iid,
        "sid_factor_audit_passed": sid_factor_passed,
        "sid_factor_role_claim_ready": sid_role_claim_ready,
        "sid_factor_role_claim_ready_runs": sid_role_claim_ready_runs,
        "sid_factor_role_total_runs": sid_role_total_runs,
        "itm_mechanism_audit_passed": itm_mechanism_passed,
        "itm_mechanism_claim_ready": itm_mechanism_claim_ready,
        "itm_mechanism_claim_ready_runs": itm_mechanism_claim_ready_runs,
        "itm_mechanism_total_runs": itm_mechanism_total_runs,
        "reasons": reasons,
    }
    # Backward-compatible keys for older generated artifacts and tests.
    gate["best_non_sid_ood_method"] = best_ood_method
    gate["best_non_sid_ood_test_accuracy_mean"] = best_non_primary_ood
    gate["best_non_sid_iid_method"] = best_iid_method
    gate["best_non_sid_iid_test_accuracy_mean"] = best_non_primary_iid
    return gate


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Result Interpretation",
        "",
        "This report interprets logged final results after protocol audit.",
        "It is a claim gate, not a training metric.",
        "",
        f"- result_root: `{report.get('root')}`",
        f"- evidence_audit_passed: `{report.get('evidence_audit_passed')}`",
        f"- primary_method: `{report.get('primary_method')}`",
        f"- positive_primary_claim_allowed: `{report.get('positive_primary_claim_allowed')}`",
        f"- positive_primary_success_claim_ready: `{report.get('positive_primary_success_claim_ready')}`",
        f"- paper_submission_scope: `{report.get('paper_submission_scope')}`",
        f"- claim_mode: `{report.get('claim_mode')}`",
        f"- recommended_wording: {report.get('recommended_wording')}",
        "",
        "## Thresholds",
        "",
    ]
    for key, value in report.get("thresholds", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Benchmarks", ""])
    for item in report.get("benchmark_gates", []):
        lines.extend(
            [
                f"### {item.get('benchmark')}",
                "",
                f"- passed: `{item.get('passed')}`",
                f"- {item.get('primary_label', PRIMARY_LABEL)} IID: "
                f"`{item.get('primary_iid_test_accuracy_mean')}`",
                f"- {item.get('primary_label', PRIMARY_LABEL)} OOD: "
                f"`{item.get('primary_ood_test_accuracy_mean')}`",
                f"- {item.get('primary_label', PRIMARY_LABEL)} OOD gap: "
                f"`{item.get('primary_ood_gap_mean')}`",
                f"- best non-{item.get('primary_label', PRIMARY_LABEL)} OOD: "
                f"`{item.get('best_non_primary_ood_method')}` = "
                f"`{item.get('best_non_primary_ood_test_accuracy_mean')}`",
                f"- SID diagnostic factor role claim ready: "
                f"`{item.get('sid_factor_role_claim_ready')}` "
                f"({item.get('sid_factor_role_claim_ready_runs')}/"
                f"{item.get('sid_factor_role_total_runs')} runs)",
                f"- ITM mechanism claim ready: "
                f"`{item.get('itm_mechanism_claim_ready')}` "
                f"({item.get('itm_mechanism_claim_ready_runs')}/"
                f"{item.get('itm_mechanism_total_runs')} runs)",
            ]
        )
        reasons = item.get("reasons") or []
        if reasons:
            lines.append("- reasons:")
            lines.extend(f"  - {reason}" for reason in reasons)
        else:
            lines.append("- reasons: none")
        lines.append("")
    lines.extend(
        [
            "## Interpretation Lock",
            "",
            "If `positive_primary_claim_allowed` is false, do not write that the",
            "primary method solves selective irreversibility or outperforms baselines. Preserve the logged",
            "result and discuss it as diagnostic evidence.",
            "",
            "`paper_submission_scope` is a paper-wording gate. Hardware execution",
            "readiness and operational warnings are tracked separately in",
            "`paper_assets_manifest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def interpret_results(
    root: str | Path,
    *,
    output: str | Path | None = None,
    markdown_output: str | Path | None = None,
    min_seeds: int = 5,
    min_epochs: int = 10,
    allow_smoke: bool = False,
    preflight_path: str | Path | None = None,
    thresholds: ClaimThresholds | None = None,
) -> dict[str, Any]:
    root = Path(root)
    thresholds = thresholds or ClaimThresholds()
    audit = audit_evidence(
        root,
        min_seeds=min_seeds,
        min_epochs=min_epochs,
        allow_smoke=allow_smoke,
        preflight_path=preflight_path,
    )

    benchmark_rows: dict[str, dict[str, dict[str, Any]]] = {}
    benchmark_gates: list[dict[str, Any]] = []
    for suite_name in EXPECTED_BENCHMARKS:
        aggregate_path = root / suite_name / "aggregate.json"
        sid_factor_path = root / suite_name / "sid_factor_audit.json"
        itm_mechanism_path = root / suite_name / "itm_mechanism_audit.json"
        rows: dict[str, dict[str, Any]] = {}
        condition = "."
        if aggregate_path.exists():
            payloads, condition = _method_payloads(_load_json(aggregate_path))
            rows = {
                method: _extract_metrics(method, payload)
                for method, payload in payloads.items()
            }
        sid_factor_passed = False
        sid_role_claim_ready = False
        sid_role_claim_ready_runs = 0
        sid_role_total_runs = 0
        itm_mechanism_passed = False
        itm_mechanism_claim_ready = False
        itm_mechanism_claim_ready_runs = 0
        itm_mechanism_total_runs = 0
        if sid_factor_path.exists():
            sid_factor = _load_json(sid_factor_path)
            sid_factor_passed = sid_factor.get("passed") is True
            sid_runs = sid_factor.get("runs", [])
            if isinstance(sid_runs, list):
                sid_role_total_runs = len(sid_runs)
                sid_role_claim_ready_runs = sum(
                    1
                    for row in sid_runs
                    if isinstance(row, Mapping)
                    and isinstance(row.get("audit"), Mapping)
                    and isinstance(row["audit"].get("metrics"), Mapping)
                    and row["audit"]["metrics"].get("decomposition_role_claim_ready")
                    is True
                )
                sid_role_claim_ready = (
                    sid_role_total_runs > 0
                    and sid_role_claim_ready_runs == sid_role_total_runs
                )
        if itm_mechanism_path.exists():
            itm_mechanism = _load_json(itm_mechanism_path)
            itm_mechanism_passed = itm_mechanism.get("passed") is True
            itm_runs = itm_mechanism.get("runs", [])
            if isinstance(itm_runs, list):
                itm_mechanism_total_runs = len(itm_runs)
                itm_mechanism_claim_ready_runs = int(
                    itm_mechanism.get("mechanism_claim_ready_runs", 0)
                )
                itm_mechanism_claim_ready = (
                    itm_mechanism_total_runs > 0
                    and itm_mechanism_claim_ready_runs == itm_mechanism_total_runs
                )
        benchmark_rows[suite_name] = rows
        gate = _benchmark_claim_gate(
            suite_name,
            rows,
            sid_factor_passed,
            sid_role_claim_ready,
            sid_role_claim_ready_runs,
            sid_role_total_runs,
            itm_mechanism_passed,
            itm_mechanism_claim_ready,
            itm_mechanism_claim_ready_runs,
            itm_mechanism_total_runs,
            thresholds,
        )
        gate["aggregate_condition"] = condition
        benchmark_gates.append(gate)

    all_benchmarks_passed = all(item.get("passed") is True for item in benchmark_gates)
    positive_allowed = (
        audit.get("passed") is True
        and all_benchmarks_passed
        and not _is_smoke_path(root)
    )
    if positive_allowed:
        claim_mode = "qualified_positive_primary_claim"
        paper_submission_scope = "positive_primary_claim_candidate"
        wording = (
            f"Logged non-smoke results support a qualified claim that {PRIMARY_LABEL} improves "
            "OOD robustness under the benchmark suite, subject to the stated "
            "controlled-counterfactual assumptions."
        )
    else:
        claim_mode = "diagnostic_or_negative_evidence"
        paper_submission_scope = "diagnostic_or_negative_evidence_candidate"
        wording = (
            "Treat the results as diagnostic evidence. State that the hypothesis was "
            f"tested and report where {PRIMARY_LABEL} did or did not improve over baselines."
        )

    report = {
        "schema": "result_interpretation_v1",
        "root": str(root),
        "primary_method": PRIMARY_METHOD,
        "primary_label": PRIMARY_LABEL,
        "evidence_audit_passed": bool(audit.get("passed")),
        "allow_smoke": bool(allow_smoke),
        "is_smoke_root": _is_smoke_path(root),
        "positive_primary_claim_allowed": bool(positive_allowed),
        "positive_primary_success_claim_ready": bool(positive_allowed),
        "positive_sid_claim_allowed": False,
        "positive_sid_success_claim_ready": False,
        "claim_mode": claim_mode,
        "paper_submission_scope": paper_submission_scope,
        "recommended_wording": wording,
        "thresholds": asdict(thresholds),
        "required_methods": list(REQUIRED_METHODS),
        "benchmark_metrics": benchmark_rows,
        "benchmark_gates": benchmark_gates,
        "audit_path": str(root / "evidence_audit.json"),
        "preflight_path": None if preflight_path is None else str(preflight_path),
        "interpretation_lock": (
            "Protocol-valid evidence is not automatically positive evidence. "
            "Only use success language when positive_primary_claim_allowed is true."
        ),
    }

    output_path = Path(output) if output else root / "result_interpretation.json"
    markdown_path = (
        Path(markdown_output)
        if markdown_output
        else root / "result_interpretation.md"
    )
    save_json(output_path, report)
    _write_text(markdown_path, _markdown_report(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpret final logged results.")
    parser.add_argument("--root", default="results/full_run")
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--preflight-path", default=None)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument(
        "--min-primary-iid",
        type=float,
        default=ClaimThresholds.min_primary_iid,
    )
    parser.add_argument(
        "--min-primary-ood",
        type=float,
        default=ClaimThresholds.min_primary_ood,
    )
    parser.add_argument(
        "--min-ood-improvement",
        type=float,
        default=ClaimThresholds.min_ood_improvement,
    )
    parser.add_argument(
        "--max-iid-drop-vs-best-non-sid",
        "--max-iid-drop-vs-best-non-primary",
        dest="max_iid_drop_vs_best_non_primary",
        type=float,
        default=ClaimThresholds.max_iid_drop_vs_best_non_primary,
    )
    args = parser.parse_args()
    report = interpret_results(
        args.root,
        output=args.output,
        markdown_output=args.markdown_output,
        min_seeds=args.min_seeds,
        min_epochs=args.min_epochs,
        allow_smoke=args.allow_smoke,
        preflight_path=args.preflight_path,
        thresholds=ClaimThresholds(
            min_primary_iid=args.min_primary_iid,
            min_primary_ood=args.min_primary_ood,
            min_ood_improvement=args.min_ood_improvement,
            max_iid_drop_vs_best_non_primary=(
                args.max_iid_drop_vs_best_non_primary
            ),
        ),
    )
    print(
        json.dumps(
            {
                "positive_primary_claim_allowed": report[
                    "positive_primary_claim_allowed"
                ],
                "claim_mode": report["claim_mode"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
