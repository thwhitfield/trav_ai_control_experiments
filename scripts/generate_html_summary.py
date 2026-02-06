"""Generate HTML and Markdown summary reports for AI Control paper evaluation runs.

This script creates summary files containing:
- The safety vs usefulness plot
- Summary statistics for all evaluation tasks
- Per-model token usage breakdown

Usage:
    python scripts/generate_html_summary.py logs/ai_control_paper_20260206_091848
"""

from __future__ import annotations

import argparse
import base64
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from inspect_ai.log import EvalLog, read_eval_log


@dataclass
class TaskSummary:
    """Summary of a single evaluation task."""

    file_name: str
    protocol_name: str
    defer_threshold: float | None
    eval_mode: str
    status: str
    completed_samples: int
    total_samples: int
    duration_seconds: float | None
    # Token usage by model role
    trusted_input_tokens: int
    trusted_output_tokens: int
    trusted_cached_tokens: int
    untrusted_input_tokens: int
    untrusted_output_tokens: int
    untrusted_cached_tokens: int
    # Model call counts
    trusted_calls: int
    untrusted_calls: int
    # Scores
    main_task_accuracy: float | None
    side_task_accuracy: float | None
    max_suspicion: float | None


def extract_task_summary(log: EvalLog, file_path: Path) -> TaskSummary:
    """Extract a summary from an eval log."""
    # Get metadata
    metadata = log.eval.metadata if log.eval else {}
    protocol_name = metadata.get("protocol_name", "unknown")
    defer_threshold = metadata.get("defer_threshold")
    eval_mode = metadata.get("mode", "unknown")

    # Status
    status = log.status

    # Sample counts
    total_samples = 0
    completed_samples = 0
    if log.results:
        total_samples = log.results.total_samples or 0
        completed_samples = log.results.completed_samples or 0

    # Duration
    duration_seconds = None
    if log.stats and log.stats.started_at and log.stats.completed_at:
        try:
            started = (
                log.stats.started_at
                if isinstance(log.stats.started_at, datetime)
                else datetime.fromisoformat(log.stats.started_at)
            )
            completed = (
                log.stats.completed_at
                if isinstance(log.stats.completed_at, datetime)
                else datetime.fromisoformat(log.stats.completed_at)
            )
            duration_seconds = (completed - started).total_seconds()
        except (ValueError, TypeError):
            pass

    # Token usage by model
    trusted_input = trusted_output = trusted_cached = 0
    untrusted_input = untrusted_output = untrusted_cached = 0
    trusted_calls = untrusted_calls = 0

    if log.stats and log.stats.model_usage:
        for model_name, usage in log.stats.model_usage.items():
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            cached_tokens = getattr(usage, "input_tokens_cache_read", 0) or 0

            # Determine if this is trusted or untrusted based on model name
            # Trusted models typically contain "gpt" or similar
            # Untrusted models typically contain "claude" or similar
            model_lower = model_name.lower()
            if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
                trusted_input += input_tokens
                trusted_output += output_tokens
                trusted_cached += cached_tokens
                trusted_calls += 1
            elif "claude" in model_lower or "anthropic" in model_lower:
                untrusted_input += input_tokens
                untrusted_output += output_tokens
                untrusted_cached += cached_tokens
                untrusted_calls += 1
            else:
                # Default to untrusted for unknown models
                untrusted_input += input_tokens
                untrusted_output += output_tokens
                untrusted_cached += cached_tokens
                untrusted_calls += 1

    # Scores
    main_task_accuracy = None
    side_task_accuracy = None
    max_suspicion = None

    if log.results and log.results.scores:
        for score in log.results.scores:
            if score.metrics:
                for metric_name, metric in score.metrics.items():
                    if metric.value is not None:
                        if "main_task" in score.name and "accuracy" in metric_name:
                            main_task_accuracy = float(metric.value)
                        elif "side_task" in score.name and "accuracy" in metric_name:
                            side_task_accuracy = float(metric.value)
                        elif "suspicion" in score.name and "accuracy" in metric_name:
                            max_suspicion = float(metric.value)

    return TaskSummary(
        file_name=file_path.name,
        protocol_name=protocol_name,
        defer_threshold=defer_threshold,
        eval_mode=eval_mode,
        status=status,
        completed_samples=completed_samples,
        total_samples=total_samples,
        duration_seconds=duration_seconds,
        trusted_input_tokens=trusted_input,
        trusted_output_tokens=trusted_output,
        trusted_cached_tokens=trusted_cached,
        untrusted_input_tokens=untrusted_input,
        untrusted_output_tokens=untrusted_output,
        untrusted_cached_tokens=untrusted_cached,
        trusted_calls=trusted_calls,
        untrusted_calls=untrusted_calls,
        main_task_accuracy=main_task_accuracy,
        side_task_accuracy=side_task_accuracy,
        max_suspicion=max_suspicion,
    )


def create_task_display_name(summary: TaskSummary, separator: str = " | ") -> str:
    """Create a human-readable display name for a task."""
    parts = [summary.protocol_name]
    if summary.defer_threshold is not None:
        parts.append(f"t={summary.defer_threshold}")
    parts.append(summary.eval_mode)
    return separator.join(parts)


def analyze_log_directory(log_dir: Path) -> list[TaskSummary]:
    """Analyze all eval files in a directory."""
    eval_files = sorted(log_dir.glob("*.eval"))
    summaries = []

    for eval_file in eval_files:
        try:
            log = read_eval_log(str(eval_file))
            summary = extract_task_summary(log, eval_file)
            summaries.append(summary)
        except Exception as e:
            print(f"Warning: Failed to read {eval_file.name}: {e}", file=sys.stderr)

    return summaries


def get_plot_base64(log_dir: Path) -> str | None:
    """Read the plot image and convert to base64."""
    plot_path = log_dir / "ai_control_safety_vs_usefulness.png"
    if not plot_path.exists():
        return None

    with open(plot_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def format_tokens(n: int) -> str:
    """Format token count with commas."""
    return f"{n:,}"


def format_duration(seconds: float | None) -> str:
    """Format duration as mm:ss."""
    if seconds is None:
        return "-"
    mins, secs = divmod(seconds, 60)
    return f"{int(mins)}:{int(secs):02d}"


def format_percentage(value: float | None) -> str:
    """Format a percentage value."""
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def generate_html_report(log_dir: Path, summaries: list[TaskSummary]) -> str:
    """Generate the HTML report."""
    plot_base64 = get_plot_base64(log_dir)

    # Calculate totals
    total_duration = sum(s.duration_seconds or 0 for s in summaries)
    total_samples = sum(s.completed_samples for s in summaries)
    total_trusted_input = sum(s.trusted_input_tokens for s in summaries)
    total_trusted_output = sum(s.trusted_output_tokens for s in summaries)
    total_untrusted_input = sum(s.untrusted_input_tokens for s in summaries)
    total_untrusted_output = sum(s.untrusted_output_tokens for s in summaries)

    # Sort summaries by protocol, threshold, mode
    summaries = sorted(
        summaries,
        key=lambda s: (
            s.protocol_name,
            s.defer_threshold if s.defer_threshold is not None else -1,
            s.eval_mode,
        ),
    )

    # Build table rows
    table_rows = []
    for s in summaries:
        display_name = create_task_display_name(s)

        # Calculate cache rates
        trusted_cache_rate = (
            f"{s.trusted_cached_tokens / s.trusted_input_tokens * 100:.0f}%"
            if s.trusted_input_tokens > 0
            else "-"
        )
        untrusted_cache_rate = (
            f"{s.untrusted_cached_tokens / s.untrusted_input_tokens * 100:.0f}%"
            if s.untrusted_input_tokens > 0
            else "-"
        )

        row = f"""
        <tr>
            <td class="task-name">{display_name}</td>
            <td class="status status-{s.status}">{s.status}</td>
            <td class="numeric">{s.completed_samples}/{s.total_samples}</td>
            <td class="numeric">{format_duration(s.duration_seconds)}</td>
            <td class="numeric">{format_tokens(s.trusted_input_tokens)}</td>
            <td class="numeric">{format_tokens(s.trusted_output_tokens)}</td>
            <td class="numeric">{trusted_cache_rate}</td>
            <td class="numeric">{format_tokens(s.untrusted_input_tokens)}</td>
            <td class="numeric">{format_tokens(s.untrusted_output_tokens)}</td>
            <td class="numeric">{untrusted_cache_rate}</td>
            <td class="numeric">{format_percentage(s.main_task_accuracy)}</td>
            <td class="numeric">{format_percentage(s.side_task_accuracy)}</td>
            <td class="numeric">{format_percentage(s.max_suspicion)}</td>
        </tr>
        """
        table_rows.append(row)

    table_rows_html = "\n".join(table_rows)

    # Plot section
    plot_html = ""
    if plot_base64:
        plot_html = f"""
        <section class="plot-section">
            <h2>Safety vs Usefulness</h2>
            <img src="data:image/png;base64,{plot_base64}" alt="Safety vs Usefulness Plot" />
        </section>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Control Evaluation Summary - {log_dir.name}</title>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --text-muted: #888;
            --border-color: #0f3460;
            --accent: #e94560;
            --success: #4ade80;
            --warning: #facc15;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
        }}

        h1 {{
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }}

        h2 {{
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: var(--accent);
        }}

        .meta {{
            color: var(--text-muted);
            font-size: 0.9rem;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}

        .card-label {{
            color: var(--text-muted);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .card-value {{
            font-size: 1.8rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }}

        .plot-section {{
            margin-bottom: 2rem;
        }}

        .plot-section img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}

        .table-section {{
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
        }}

        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background: var(--border-color);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            white-space: nowrap;
        }}

        th.group-header {{
            text-align: center;
            background: var(--accent);
            color: white;
        }}

        tr:hover {{
            background: rgba(255, 255, 255, 0.03);
        }}

        .numeric {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}

        .task-name {{
            font-weight: 500;
            white-space: nowrap;
        }}

        .status {{
            font-weight: 600;
        }}

        .status-success {{
            color: var(--success);
        }}

        .status-error, .status-failed {{
            color: var(--accent);
        }}

        .status-cancelled {{
            color: var(--warning);
        }}

        footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
            color: var(--text-muted);
            font-size: 0.85rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AI Control Evaluation Summary</h1>
            <p class="meta">Run: {log_dir.name}</p>
            <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <section class="summary-cards">
            <div class="card">
                <div class="card-label">Total Tasks</div>
                <div class="card-value">{len(summaries)}</div>
            </div>
            <div class="card">
                <div class="card-label">Total Samples</div>
                <div class="card-value">{total_samples:,}</div>
            </div>
            <div class="card">
                <div class="card-label">Total Duration</div>
                <div class="card-value">{format_duration(total_duration)}</div>
            </div>
            <div class="card">
                <div class="card-label">Trusted Tokens</div>
                <div class="card-value">{format_tokens(total_trusted_input + total_trusted_output)}</div>
            </div>
            <div class="card">
                <div class="card-label">Untrusted Tokens</div>
                <div class="card-value">{format_tokens(total_untrusted_input + total_untrusted_output)}</div>
            </div>
        </section>

        {plot_html}

        <section class="table-section">
            <h2>Task Details</h2>
            <table>
                <thead>
                    <tr>
                        <th rowspan="2">Task</th>
                        <th rowspan="2">Status</th>
                        <th rowspan="2">Samples</th>
                        <th rowspan="2">Duration</th>
                        <th colspan="3" class="group-header">Trusted Model</th>
                        <th colspan="3" class="group-header">Untrusted Model</th>
                        <th colspan="3" class="group-header">Scores</th>
                    </tr>
                    <tr>
                        <th class="numeric">Input</th>
                        <th class="numeric">Output</th>
                        <th class="numeric">Cache</th>
                        <th class="numeric">Input</th>
                        <th class="numeric">Output</th>
                        <th class="numeric">Cache</th>
                        <th class="numeric">Main</th>
                        <th class="numeric">Side</th>
                        <th class="numeric">Suspicion</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows_html}
                </tbody>
            </table>
        </section>

        <footer>
            Generated by AI Control Experiments
        </footer>
    </div>
</body>
</html>
"""
    return html


def generate_markdown_report(log_dir: Path, summaries: list[TaskSummary]) -> str:
    """Generate the Markdown report."""
    # Calculate totals
    total_duration = sum(s.duration_seconds or 0 for s in summaries)
    total_samples = sum(s.completed_samples for s in summaries)
    total_trusted_input = sum(s.trusted_input_tokens for s in summaries)
    total_trusted_output = sum(s.trusted_output_tokens for s in summaries)
    total_untrusted_input = sum(s.untrusted_input_tokens for s in summaries)
    total_untrusted_output = sum(s.untrusted_output_tokens for s in summaries)

    # Sort summaries by protocol, threshold, mode
    summaries = sorted(
        summaries,
        key=lambda s: (
            s.protocol_name,
            s.defer_threshold if s.defer_threshold is not None else -1,
            s.eval_mode,
        ),
    )

    lines = []

    # Header
    lines.append("# AI Control Evaluation Summary")
    lines.append("")
    lines.append(f"**Run:** `{log_dir.name}`")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary stats
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Tasks | {len(summaries)} |")
    lines.append(f"| Total Samples | {total_samples:,} |")
    lines.append(f"| Total Duration | {format_duration(total_duration)} |")
    lines.append(
        f"| Trusted Tokens | {format_tokens(total_trusted_input + total_trusted_output)} |"
    )
    lines.append(
        f"| Untrusted Tokens | {format_tokens(total_untrusted_input + total_untrusted_output)} |"
    )
    lines.append("")

    # Plot reference
    plot_path = log_dir / "ai_control_safety_vs_usefulness.png"
    if plot_path.exists():
        lines.append("## Safety vs Usefulness")
        lines.append("")
        lines.append("![Safety vs Usefulness](ai_control_safety_vs_usefulness.png)")
        lines.append("")

    # Task details table
    lines.append("## Task Details")
    lines.append("")

    # Table header
    lines.append(
        "| Task | Status | Samples | Duration | "
        "Trusted In | Trusted Out | Trusted Cache | "
        "Untrusted In | Untrusted Out | Untrusted Cache | "
        "Main | Side | Suspicion |"
    )
    lines.append(
        "|------|--------|---------|----------|"
        "-----------|-------------|---------------|"
        "-------------|---------------|----------------|"
        "------|------|-----------|"
    )

    # Table rows
    for s in summaries:
        display_name = create_task_display_name(s, separator=" / ")

        # Calculate cache rates
        trusted_cache_rate = (
            f"{s.trusted_cached_tokens / s.trusted_input_tokens * 100:.0f}%"
            if s.trusted_input_tokens > 0
            else "-"
        )
        untrusted_cache_rate = (
            f"{s.untrusted_cached_tokens / s.untrusted_input_tokens * 100:.0f}%"
            if s.untrusted_input_tokens > 0
            else "-"
        )

        row = (
            f"| {display_name} "
            f"| {s.status} "
            f"| {s.completed_samples}/{s.total_samples} "
            f"| {format_duration(s.duration_seconds)} "
            f"| {format_tokens(s.trusted_input_tokens)} "
            f"| {format_tokens(s.trusted_output_tokens)} "
            f"| {trusted_cache_rate} "
            f"| {format_tokens(s.untrusted_input_tokens)} "
            f"| {format_tokens(s.untrusted_output_tokens)} "
            f"| {untrusted_cache_rate} "
            f"| {format_percentage(s.main_task_accuracy)} "
            f"| {format_percentage(s.side_task_accuracy)} "
            f"| {format_percentage(s.max_suspicion)} |"
        )
        lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("*Generated by AI Control Experiments*")

    return "\n".join(lines)


def main(log_dir: Path) -> None:
    """Generate HTML and Markdown summaries for a log directory."""
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    eval_files = list(log_dir.glob("*.eval"))
    if not eval_files:
        print(f"Error: No .eval files found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(eval_files)} eval files...")
    summaries = analyze_log_directory(log_dir)
    print(f"Extracted summaries for {len(summaries)} tasks")

    # Generate HTML
    html = generate_html_report(log_dir, summaries)
    html_path = log_dir / "summary.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Generated HTML summary: {html_path}")

    # Generate Markdown
    markdown = generate_markdown_report(log_dir, summaries)
    md_path = log_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"Generated Markdown summary: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an HTML summary report for AI Control evaluation runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/generate_html_summary.py logs/ai_control_paper_20260206_091848
        """,
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Path to log directory containing .eval files",
    )
    args = parser.parse_args()

    main(args.log_dir)
