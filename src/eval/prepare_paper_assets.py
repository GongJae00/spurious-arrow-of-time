"""Prepare final paper assets from audited logged results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from src.eval.audit_evidence import audit_evidence
from src.eval.audit_smoke import EXPECTED_BENCHMARKS, REQUIRED_METHODS
from src.eval.interpret_results import interpret_results
from src.train.common import save_json


DEFAULT_OUTPUT_DIR = "paper/generated_irreversibility_trust"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _is_smoke_path(path: Path) -> bool:
    return any("smoke" in part.lower() for part in path.parts)


def _latex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _latex_breakable_identifier(value: Any) -> str:
    return _latex_escape(value).replace("\\_", "\\_\\allowbreak{}")


def _metric(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return _latex_escape(value)


def _method_metrics(payload: Mapping[str, Any], method: str) -> dict[str, Any]:
    row = payload.get(method, {})
    if method in {"ocp_style", "lens_like_arrow_classifier"}:
        return {
            "n_runs": row.get("n_runs"),
            "iid": row.get("fine_tuned_encoder_iid_test_accuracy_mean"),
            "ood": row.get("fine_tuned_encoder_ood_test_accuracy_mean"),
            "gap": row.get("fine_tuned_encoder_ood_gap_mean"),
        }
    return {
        "n_runs": row.get("n_runs"),
        "iid": row.get("iid_test_accuracy_mean"),
        "ood": row.get("ood_test_accuracy_mean"),
        "gap": row.get("ood_gap_mean"),
    }


def _method_payloads(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    top_level = {
        method: aggregate[method]
        for method in REQUIRED_METHODS
        if isinstance(aggregate.get(method), Mapping)
    }
    if top_level:
        return top_level
    by_condition = aggregate.get("by_condition", {})
    if isinstance(by_condition, Mapping):
        for _, payload in sorted(by_condition.items()):
            if not isinstance(payload, Mapping):
                continue
            condition_methods = {
                method: payload[method]
                for method in REQUIRED_METHODS
                if isinstance(payload.get(method), Mapping)
            }
            if condition_methods:
                return condition_methods
    return {}


def _metrics_table(result_root: Path) -> str:
    lines = [
        "% Generated metrics table generated from logged aggregate.json files.",
        "% Do not edit numeric values by hand.",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Benchmark & Method & Runs & IID acc & OOD acc & OOD gap \\\\",
        "\\midrule",
    ]
    for suite_name in EXPECTED_BENCHMARKS:
        aggregate = _method_payloads(_load_json(result_root / suite_name / "aggregate.json"))
        for method in REQUIRED_METHODS:
            metrics = _method_metrics(aggregate, method)
            lines.append(
                " & ".join(
                    [
                        _latex_escape(suite_name),
                        _latex_escape(method),
                        _metric(metrics["n_runs"]),
                        _metric(metrics["iid"]),
                        _metric(metrics["ood"]),
                        _metric(metrics["gap"]),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _sid_factor_table(result_root: Path) -> str:
    lines = [
        "% SID factor audit table generated from sid_factor_audit.json files.",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Benchmark & Split & Task head lock & z\\_rev CF MSE & z\\_task CF MSE & z\\_spur CF MSE \\\\",
        "\\midrule",
    ]
    for suite_name in EXPECTED_BENCHMARKS:
        summary = _load_json(result_root / suite_name / "sid_factor_audit.json")
        if not summary.get("runs"):
            continue
        audit = summary["runs"][0]["audit"]
        metrics = audit.get("metrics", {})
        for split in ("iid_test", "ood_test"):
            lines.append(
                " & ".join(
                    [
                        _latex_escape(suite_name),
                        _latex_escape(split),
                        _latex_escape(metrics.get("task_head_excludes_z_ir_spur")),
                        _metric(metrics.get(f"{split}_z_rev_cf_mse")),
                        _metric(metrics.get(f"{split}_z_ir_task_cf_mse")),
                        _metric(metrics.get(f"{split}_z_ir_spur_cf_mse")),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _itm_mechanism_table(result_root: Path) -> str:
    lines = [
        "% ITM mechanism audit table generated from itm_mechanism_audit.json files.",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Benchmark & Split & Task head lock & Core CF MSE & Spur CF MSE & Task-spur AUC \\\\",
        "\\midrule",
    ]
    for suite_name in EXPECTED_BENCHMARKS:
        summary = _load_json(result_root / suite_name / "itm_mechanism_audit.json")
        if not summary.get("runs"):
            continue
        audit = summary["runs"][0]["audit"]
        metrics = audit.get("metrics", {})
        for split in ("iid_test", "ood_test"):
            lines.append(
                " & ".join(
                    [
                        _latex_escape(suite_name),
                        _latex_escape(split),
                        _latex_escape(metrics.get("task_head_excludes_spur_mechanism")),
                        _metric(metrics.get(f"{split}_core_delta_cf_mse")),
                        _metric(metrics.get(f"{split}_spur_delta_cf_mse")),
                        _metric(
                            metrics.get(
                                f"{split}_task_rep_spurious_dynamic_residualized_auc"
                            )
                        ),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _claim_gate_interpretation(interpretation: Mapping[str, Any]) -> list[str]:
    gates = interpretation.get("benchmark_gates", [])
    lines = [
        "\\paragraph{Generated Claim-Gate Interpretation}",
        "The following benchmark-level interpretation is generated from",
        "\\texttt{result\\_interpretation.json}. It is included so",
        "benchmark-specific outcome language is tied to logged artifacts rather",
        "than hand-written manuscript text.",
        "\\begin{itemize}",
    ]
    if not isinstance(gates, list) or not gates:
        lines.append(
            "\\item No benchmark-level claim gates were available in the "
            "interpretation artifact."
        )
    else:
        for gate in gates:
            if not isinstance(gate, Mapping):
                continue
            benchmark = _latex_breakable_identifier(gate.get("benchmark", "unknown"))
            status = "passed" if gate.get("passed") is True else "not passed"
            primary_label = _latex_escape(gate.get("primary_label", "primary"))
            primary_iid = _metric(gate.get("primary_iid_test_accuracy_mean"))
            primary_ood = _metric(gate.get("primary_ood_test_accuracy_mean"))
            primary_gap = _metric(gate.get("primary_ood_gap_mean"))
            best_method = _latex_breakable_identifier(
                gate.get(
                    "best_non_primary_ood_method",
                    gate.get("best_non_sid_ood_method", "--"),
                )
            )
            best_ood = _metric(
                gate.get(
                    "best_non_primary_ood_test_accuracy_mean",
                    gate.get("best_non_sid_ood_test_accuracy_mean"),
                )
            )
            role_ready = _latex_escape(gate.get("sid_factor_role_claim_ready"))
            role_ready_runs = _latex_escape(gate.get("sid_factor_role_claim_ready_runs"))
            role_total_runs = _latex_escape(gate.get("sid_factor_role_total_runs"))
            itm_ready = _latex_escape(gate.get("itm_mechanism_claim_ready"))
            itm_ready_runs = _latex_escape(gate.get("itm_mechanism_claim_ready_runs"))
            itm_total_runs = _latex_escape(gate.get("itm_mechanism_total_runs"))
            lines.append(
                "\\item "
                f"\\texttt{{{benchmark}}}: claim gate {status}. "
                f"{primary_label} IID={primary_iid}, "
                f"{primary_label} OOD={primary_ood}, "
                f"{primary_label} OOD gap={primary_gap}; "
                f"best OOD method={best_method} ({best_ood}); "
                f"SID diagnostic factor role claim ready={role_ready} "
                f"({role_ready_runs}/{role_total_runs} runs); "
                f"ITM mechanism claim ready={itm_ready} "
                f"({itm_ready_runs}/{itm_total_runs} runs)."
            )
            reasons = gate.get("reasons", [])
            if isinstance(reasons, list) and reasons:
                reason_text = "; ".join(_latex_escape(reason) for reason in reasons)
                lines.append(f"Gate reasons: {reason_text}.")
            else:
                lines.append("Gate reasons: none.")
    lines.extend(["\\end{itemize}", ""])
    return lines


def _readme_claim_gate_summary(interpretation: Mapping[str, Any]) -> list[str]:
    gates = interpretation.get("benchmark_gates", [])
    lines = [
        "",
        "Benchmark claim gate summary:",
        "",
        "```text",
    ]
    if not isinstance(gates, list) or not gates:
        lines.append("no benchmark-level claim gates available")
    else:
        for gate in gates:
            if not isinstance(gate, Mapping):
                continue
            benchmark = gate.get("benchmark", "unknown")
            passed = str(gate.get("passed")).lower()
            primary_label = gate.get("primary_label", "primary")
            role_ready = str(gate.get("sid_factor_role_claim_ready")).lower()
            role_ready_runs = gate.get("sid_factor_role_claim_ready_runs")
            role_total_runs = gate.get("sid_factor_role_total_runs")
            itm_ready = str(gate.get("itm_mechanism_claim_ready")).lower()
            itm_ready_runs = gate.get("itm_mechanism_claim_ready_runs")
            itm_total_runs = gate.get("itm_mechanism_total_runs")
            lines.append(
                f"{benchmark}: claim_gate_passed={passed}, "
                f"primary_method={primary_label}, "
                f"SID diagnostic factor role claim ready={role_ready} "
                f"({role_ready_runs}/{role_total_runs} runs), "
                f"ITM mechanism claim ready={itm_ready} "
                f"({itm_ready_runs}/{itm_total_runs} runs)"
            )
    lines.extend(
        [
            "```",
            "",
            "SID factor-role diagnostics are auxiliary. Do not describe SID as",
            "having learned the intended reversible/task-irreversible/",
            "spurious-irreversible factor roles unless its diagnostic gate is ready.",
        ]
    )
    return lines


def _results_summary(
    result_root: Path,
    output_dir: Path,
    audit: Mapping[str, Any],
    interpretation: Mapping[str, Any],
) -> str:
    rel_table = output_dir / "tables" / "main_metrics_table.tex"
    rel_sid = output_dir / "tables" / "sid_factor_audit_table.tex"
    rel_itm = output_dir / "tables" / "itm_mechanism_audit_table.tex"
    positive_primary_success_claim_ready = bool(
        interpretation.get(
            "positive_primary_success_claim_ready",
            interpretation.get("positive_primary_claim_allowed"),
        )
    )
    paper_submission_scope = str(
        interpretation.get(
            "paper_submission_scope",
            "positive_primary_claim_candidate"
            if positive_primary_success_claim_ready
            else "diagnostic_or_negative_evidence_candidate",
        )
    )
    primary_label = str(interpretation.get("primary_label", "primary method"))
    return "\n".join(
        [
            "% Generated from final logged artifacts only.",
            f"% Result root: {result_root}",
            "",
            "\\section{Logged Results}",
            "These assets were generated from the Spurious Arrow of Time",
            "pipeline for \\emph{Learning Which Irreversibility to Trust}.",
            "They must be interpreted through the accompanying",
            "\\texttt{evidence\\_audit.json}.",
            "",
            "This generated section is not a success claim. It reports logged",
            "IID/OOD metrics and SID factor-audit diagnostics. Any claim that",
            f"{primary_label} solves selective irreversibility or broadly dominates the",
            "baselines requires the non-smoke audit and logged results to",
            "support it.",
            "",
            f"Audit pass flag: \\texttt{{{str(audit.get('passed')).lower()}}}.",
            f"Run count: \\texttt{{{audit.get('run_count')}}}.",
            f"Minimum seeds: \\texttt{{{audit.get('min_seeds')}}}.",
            f"Minimum epochs: \\texttt{{{audit.get('min_epochs')}}}.",
            "",
            "\\begin{quote}",
            f"Positive primary claim allowed: \\texttt{{{str(interpretation.get('positive_primary_claim_allowed')).lower()}}}.\\\\",
            f"Positive primary success claim ready: \\texttt{{{str(positive_primary_success_claim_ready).lower()}}}.\\\\",
            f"Paper submission scope: \\texttt{{{_latex_escape(paper_submission_scope)}}}.\\\\",
            f"Claim mode: \\texttt{{{_latex_escape(interpretation.get('claim_mode'))}}}.",
            "\\end{quote}",
            "",
            "The generated paper must follow the result interpretation",
            "artifact. If the positive claim flag is false, report the result as",
            "diagnostic or negative evidence rather than as a method success claim.",
            "Warning-free hardware readiness and operational warning names are",
            "recorded in \\texttt{paper\\_assets\\_manifest.json};",
            "\\texttt{final\\_submission\\_ready} alone is not a positive method",
            "claim and is not a warning-free hardware-execution claim.",
            "",
            "\\paragraph{Main Metrics}",
            "\\begin{center}",
            f"\\resizebox{{\\textwidth}}{{!}}{{\\input{{{rel_table.as_posix()}}}}}",
            "\\end{center}",
            "",
            "\\paragraph{ITM Mechanism Audit}",
            "\\begin{center}",
            f"\\resizebox{{\\textwidth}}{{!}}{{\\input{{{rel_itm.as_posix()}}}}}",
            "\\end{center}",
            "",
            "\\paragraph{SID Factor Audit}",
            "\\begin{center}",
            f"\\resizebox{{\\textwidth}}{{!}}{{\\input{{{rel_sid.as_posix()}}}}}",
            "\\end{center}",
            "",
            *_claim_gate_interpretation(interpretation),
        ]
    )


def prepare_paper_assets(
    result_root: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    min_seeds: int = 5,
    min_epochs: int = 10,
    allow_smoke: bool = False,
    preflight_path: str | Path | None = None,
) -> dict[str, Any]:
    result_root = Path(result_root)
    output_dir = Path(output_dir)
    resolved_preflight_path = (
        None
        if allow_smoke
        else Path(preflight_path)
        if preflight_path is not None
        else result_root / "preflight.json"
    )
    audit_path = result_root / "evidence_audit.json"
    audit = audit_evidence(
        result_root,
        min_seeds=min_seeds,
        min_epochs=min_epochs,
        allow_smoke=allow_smoke,
        preflight_path=resolved_preflight_path,
    )
    save_json(audit_path, audit)

    if not audit.get("passed"):
        raise ValueError(f"evidence audit did not pass: {audit_path}")
    if _is_smoke_path(result_root) and not allow_smoke:
        raise ValueError("refusing to generate final paper assets from a smoke result root")
    interpretation = interpret_results(
        result_root,
        min_seeds=min_seeds,
        min_epochs=min_epochs,
        allow_smoke=allow_smoke,
        preflight_path=resolved_preflight_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    _write_text(tables_dir / "main_metrics_table.tex", _metrics_table(result_root))
    _write_text(
        tables_dir / "itm_mechanism_audit_table.tex",
        _itm_mechanism_table(result_root),
    )
    _write_text(tables_dir / "sid_factor_audit_table.tex", _sid_factor_table(result_root))
    _write_text(
        output_dir / "results_summary.tex",
        _results_summary(result_root, output_dir, audit, interpretation),
    )
    interpretation_src = result_root / "result_interpretation.md"
    if interpretation_src.exists():
        _write_text(
            output_dir / "result_interpretation.md",
            interpretation_src.read_text(encoding="utf-8"),
        )
    final_ready = bool(audit.get("passed")) and not _is_smoke_path(result_root)
    positive_primary_success_claim_ready = bool(
        interpretation.get(
            "positive_primary_success_claim_ready",
            interpretation.get("positive_primary_claim_allowed"),
        )
    )
    paper_submission_scope = str(
        interpretation.get(
            "paper_submission_scope",
            "positive_primary_claim_candidate"
            if positive_primary_success_claim_ready
            else "diagnostic_or_negative_evidence_candidate",
        )
    )
    preflight_environment_summary = audit.get("preflight_environment_summary")
    if not isinstance(preflight_environment_summary, dict):
        preflight_environment_summary = {}
    _write_text(
        output_dir / "README.md",
        "\n".join(
            [
                "# Paper Assets",
                "",
                "Generated from final logged artifacts only.",
                "",
                f"Result root: `{result_root}`",
                f"Audit path: `{audit_path}`",
                "",
                "These assets are final-paper candidates only if the result root is",
                "non-smoke and `evidence_audit.json` passed with the required",
                "seed/epoch thresholds.",
                "",
                "Readiness fields are intentionally separated:",
                "",
                "```text",
                f"final_manuscript_ready: {str(final_ready).lower()}",
                f"positive_primary_success_claim_ready: {str(positive_primary_success_claim_ready).lower()}",
                f"paper_submission_scope: {paper_submission_scope}",
                "final_manuscript_ready_without_warnings: see paper_assets_manifest.json",
                "hardware_execution_context_ready: see paper_assets_manifest.json",
                "operational_warning_names: see paper_assets_manifest.json",
                "```",
                "",
                "The final paper policy audit must be run in final mode",
                "before these assets are treated as final-paper candidates:",
                "",
                "```text",
                "require_final_manuscript_ready=true",
                "generated_manifest_final_manuscript_ready=true",
                "```",
                "",
                "`final_manuscript_ready=true` does not by itself permit method success",
                "language and does not mean the current hardware execution context is",
                "warning-free. Follow `result_interpretation.json` and",
                "`paper_assets_manifest.json`.",
                "",
                "Preflight authorization provenance:",
                "",
                "```text",
                "device: "
                f"{preflight_environment_summary.get('device')}",
                "require_cuda_for_full_run: "
                f"{preflight_environment_summary.get('require_cuda_for_full_run')}",
                "launch_authorization: "
                f"{preflight_environment_summary.get('launch_authorization')}",
                "```",
                *_readme_claim_gate_summary(interpretation),
                "",
            ]
        ),
    )
    manifest = {
        "schema": "paper_assets_v1",
        "title": "The Spurious Arrow of Time: Learning Which Irreversibility to Trust",
        "result_root": str(result_root),
        "output_dir": str(output_dir),
        "audit_path": str(audit_path),
        "preflight_path": None if resolved_preflight_path is None else str(resolved_preflight_path),
        "evidence_audit_passed": bool(audit.get("passed")),
        "evidence_audit_run_count": int(audit.get("run_count", 0)),
        "evidence_audit_n_checks": int(audit.get("n_checks", 0)),
        "evidence_audit_n_failed": int(audit.get("n_failed", 0)),
        "evidence_audit_failed_checks": list(audit.get("failed_checks", [])),
        "preflight_device": preflight_environment_summary.get("device"),
        "preflight_require_cuda_for_full_run": (
            preflight_environment_summary.get("require_cuda_for_full_run")
        ),
        "preflight_launch_authorization": (
            preflight_environment_summary.get("launch_authorization")
        ),
        "preflight_environment_summary": preflight_environment_summary,
        "positive_primary_claim_allowed": bool(
            interpretation.get("positive_primary_claim_allowed")
        ),
        "positive_primary_success_claim_ready": positive_primary_success_claim_ready,
        "positive_sid_claim_allowed": bool(
            interpretation.get("positive_sid_claim_allowed", False)
        ),
        "positive_sid_success_claim_ready": bool(
            interpretation.get("positive_sid_success_claim_ready", False)
        ),
        "primary_method": interpretation.get("primary_method"),
        "primary_label": interpretation.get("primary_label"),
        "claim_mode": interpretation.get("claim_mode"),
        "result_claim_mode": interpretation.get("claim_mode"),
        "paper_submission_scope": paper_submission_scope,
        "result_recommended_wording": interpretation.get("recommended_wording"),
        "final_manuscript_ready": final_ready,
        "final_manuscript_ready_without_warnings": None,
        "hardware_execution_context_ready": None,
        "operational_warning_names": [],
        "allow_smoke": bool(allow_smoke),
        "generated_files": {
            "results_summary": str(output_dir / "results_summary.tex"),
            "main_metrics_table": str(tables_dir / "main_metrics_table.tex"),
            "itm_mechanism_audit_table": str(
                tables_dir / "itm_mechanism_audit_table.tex"
            ),
            "sid_factor_audit_table": str(tables_dir / "sid_factor_audit_table.tex"),
            "result_interpretation": str(
                output_dir / "result_interpretation.md"
            ),
            "readme": str(output_dir / "README.md"),
        },
        "interpretation_lock": (
            "Generated values are logged evidence, not hand-edited claims. "
            "Do not use success language unless evidence supports it."
        ),
    }
    save_json(output_dir / "paper_assets_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare final paper assets.")
    parser.add_argument("--result-root", default="results/full_run")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--preflight-path", default=None)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    manifest = prepare_paper_assets(
        args.result_root,
        output_dir=args.output_dir,
        min_seeds=args.min_seeds,
        min_epochs=args.min_epochs,
        allow_smoke=args.allow_smoke,
        preflight_path=args.preflight_path,
    )
    print(
        json.dumps(
            {
                "output_dir": manifest["output_dir"],
                "final_manuscript_ready": manifest["final_manuscript_ready"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
