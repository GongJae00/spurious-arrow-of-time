"""Export paper-ready tables from logged result files only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = (
    "val_iid_accuracy",
    "iid_test_accuracy",
    "ood_test_accuracy",
    "ood_gap",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _metric_mean_std(summary: dict[str, Any], metric: str) -> str:
    mean = summary.get(f"{metric}_mean")
    std = summary.get(f"{metric}_std")
    if mean is None:
        fine_tuned = f"fine_tuned_encoder_{metric}"
        mean = summary.get(f"{fine_tuned}_mean")
        std = summary.get(f"{fine_tuned}_std")
    if mean is None:
        return ""
    if std is None:
        return _fmt(mean)
    return f"{_fmt(mean)} +/- {_fmt(std)}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(str(cell)) for cell in [header] + [row[idx] for row in rows])
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def _latex_table(headers: list[str], rows: list[list[str]]) -> str:
    cols = "l" * len(headers)
    lines = [
        "\\begin{center}",
        "\\resizebox{0.98\\linewidth}{!}{%",
        "\\begin{tabular}{" + cols + "}",
        "\\toprule",
        " & ".join(_latex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(_latex_escape(cell) for cell in row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{center}"])
    return "\n".join(lines)


def _latex_escape(cell: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(cell))


def _table_title(text: str, fmt: str) -> str:
    if fmt == "latex":
        text = text.replace(" - actual logged results only.", ".")
        text = text.replace(" - val_iid selector, no OOD tuning.", ".")
    return _latex_escape(text) if fmt == "latex" else text


def _render(headers: list[str], rows: list[list[str]], fmt: str) -> str:
    if not rows:
        raise ValueError("no logged rows available for table export")
    if fmt == "markdown":
        return _markdown_table(headers, rows)
    if fmt == "latex":
        return _latex_table(headers, rows)
    raise ValueError(f"unknown table format {fmt!r}")


def export_main_metrics_table(
    aggregate_json: str | Path,
    output: str | Path | None = None,
    *,
    fmt: str = "markdown",
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> str:
    data = _load_json(aggregate_json)
    rows = []
    for method in sorted(k for k in data if k != "by_condition" and isinstance(data[k], dict)):
        summary = data[method]
        if "n_runs" not in summary:
            continue
        rows.append(
            [
                method,
                str(int(summary["n_runs"])),
                *[_metric_mean_std(summary, metric) for metric in metrics],
            ]
        )
    headers = ["method", "n"] + list(metrics)
    table = _render(headers, rows, fmt)
    text = (
        _table_title("Logged metrics table - actual logged results only.", fmt)
        + "\n\n"
        + table
        + "\n"
    )
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text)
    return text


def export_condition_metrics_table(
    aggregate_json: str | Path,
    output: str | Path | None = None,
    *,
    fmt: str = "markdown",
    metric: str = "ood_test_accuracy",
) -> str:
    data = _load_json(aggregate_json)
    by_condition = data.get("by_condition", {})
    rows = []
    for condition, methods in sorted(by_condition.items()):
        for method, summary in sorted(methods.items()):
            rows.append(
                [
                    condition,
                    method,
                    str(int(summary.get("n_runs", 0))),
                    _metric_mean_std(summary, metric),
                ]
            )
    headers = ["condition", "method", "n", metric]
    table = _render(headers, rows, fmt)
    text = (
        _table_title("Logged condition table - actual logged results only.", fmt)
        + "\n\n"
        + table
        + "\n"
    )
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text)
    return text


def export_setpoint_selection_table(
    selection_json: str | Path,
    output: str | Path | None = None,
    *,
    fmt: str = "markdown",
) -> str:
    data = _load_json(selection_json)
    rows = []
    for method, row in sorted(data.get("methods", {}).items()):
        rows.append(
            [
                method,
                _fmt(row.get("selected_target")),
                str(row.get("selected_condition", "")),
                _fmt(row.get("selection_metric_value")),
                str(row.get("n_candidates", "")),
            ]
        )
    headers = ["method", "selected_target", "selected_condition", "val_iid_metric", "n_candidates"]
    table = _render(headers, rows, fmt)
    text = (
        _table_title(
            "Logged setpoint selection table - val_iid selector, no OOD tuning.",
            fmt,
        )
        + "\n\n"
        + table
        + "\n"
    )
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tables from logged result files.")
    parser.add_argument("input_json")
    parser.add_argument("--kind", choices=["main", "condition", "setpoint_selection"], default="main")
    parser.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    parser.add_argument("--metric", default="ood_test_accuracy")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.kind == "main":
        text = export_main_metrics_table(args.input_json, args.output, fmt=args.format)
    elif args.kind == "condition":
        text = export_condition_metrics_table(
            args.input_json,
            args.output,
            fmt=args.format,
            metric=args.metric,
        )
    else:
        text = export_setpoint_selection_table(args.input_json, args.output, fmt=args.format)
    print(text)


if __name__ == "__main__":
    main()
