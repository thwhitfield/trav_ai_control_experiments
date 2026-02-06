"""Analysis utilities for extracting metrics from evaluation logs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from inspect_ai.log import EvalLog, read_eval_log

logger = logging.getLogger(__name__)


def parse_iso_datetime(value: str | datetime | None) -> datetime | None:
    """Parse an ISO format datetime string, or return datetime as-is."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


@dataclass
class TokenUsage:
    """Token usage statistics for a model."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass
class EvalTaskMetrics:
    """Metrics extracted from a single eval task."""

    file_path: Path
    task_name: str
    model: str
    status: str

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    # Sample counts
    total_samples: int = 0
    completed_samples: int = 0

    # Token usage per model
    token_usage: dict[str, TokenUsage] = field(default_factory=dict)

    # Aggregate token usage
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)

    # Scores
    scores: dict[str, float] = field(default_factory=dict)

    # Additional metadata
    dataset: str | None = None
    sandbox_type: str | None = None


def extract_token_usage(model_usage: dict[str, Any]) -> dict[str, TokenUsage]:
    """Extract token usage from model_usage dict."""
    result = {}
    for model_name, usage in model_usage.items():
        result[model_name] = TokenUsage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            cached_input_tokens=getattr(usage, "input_tokens_cache_read", 0) or 0,
            reasoning_tokens=getattr(usage, "reasoning_tokens", 0) or 0,
        )
    return result


def extract_metrics_from_log(log: EvalLog, file_path: Path) -> EvalTaskMetrics:
    """Extract metrics from an EvalLog object."""
    # Basic info
    task_name = log.eval.task if log.eval else "unknown"
    model = log.eval.model if log.eval else "unknown"
    status = log.status

    # Timing - parse ISO strings to datetime objects
    started_at = None
    completed_at = None
    duration_seconds = None

    if log.stats:
        started_at = parse_iso_datetime(log.stats.started_at)
        completed_at = parse_iso_datetime(log.stats.completed_at)
        if started_at and completed_at:
            duration_seconds = (completed_at - started_at).total_seconds()

    # Sample counts
    total_samples = 0
    completed_samples = 0
    if log.results:
        total_samples = log.results.total_samples or 0
        completed_samples = log.results.completed_samples or 0

    # Token usage
    token_usage: dict[str, TokenUsage] = {}
    total_token_usage = TokenUsage()
    if log.stats and log.stats.model_usage:
        token_usage = extract_token_usage(log.stats.model_usage)
        for usage in token_usage.values():
            total_token_usage = total_token_usage + usage

    # Scores
    scores: dict[str, float] = {}
    if log.results and log.results.scores:
        for score in log.results.scores:
            if score.metrics:
                for metric_name, metric in score.metrics.items():
                    key = f"{score.name}.{metric_name}" if score.name else metric_name
                    if metric.value is not None:
                        scores[key] = float(metric.value)

    # Dataset info
    dataset = None
    if log.eval and log.eval.dataset:
        dataset = log.eval.dataset.name

    # Sandbox type
    sandbox_type = None
    if log.eval and log.eval.sandbox:
        sandbox_type = log.eval.sandbox.type if hasattr(log.eval.sandbox, "type") else str(log.eval.sandbox)

    return EvalTaskMetrics(
        file_path=file_path,
        task_name=task_name,
        model=model,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        total_samples=total_samples,
        completed_samples=completed_samples,
        token_usage=token_usage,
        total_token_usage=total_token_usage,
        scores=scores,
        dataset=dataset,
        sandbox_type=sandbox_type,
    )


def analyze_eval_file(file_path: Path) -> EvalTaskMetrics | None:
    """Analyze a single eval file and extract metrics."""
    try:
        log = read_eval_log(str(file_path))
        return extract_metrics_from_log(log, file_path)
    except Exception as e:
        logger.warning(f"Failed to read eval file {file_path}: {e}")
        return None


def analyze_log_directory(log_dir: Path) -> list[EvalTaskMetrics]:
    """Analyze all eval files in a directory."""
    eval_files = list(log_dir.glob("*.eval"))
    if not eval_files:
        logger.warning(f"No .eval files found in {log_dir}")
        return []

    metrics = []
    for eval_file in sorted(eval_files):
        result = analyze_eval_file(eval_file)
        if result:
            metrics.append(result)

    return metrics


def metrics_to_dataframe(metrics: list[EvalTaskMetrics]) -> pd.DataFrame:
    """Convert list of metrics to a DataFrame for display/analysis."""
    rows = []
    for m in metrics:
        # Calculate cache rate
        cache_rate = None
        if m.total_token_usage.input_tokens > 0:
            cache_rate = round(
                m.total_token_usage.cached_input_tokens / m.total_token_usage.input_tokens * 100, 1
            )

        row = {
            "file": m.file_path.name,
            "task": m.task_name,
            "model": m.model,
            "status": m.status,
            "dataset": m.dataset,
            "completed_samples": m.completed_samples,
            "total_samples": m.total_samples,
            "started_at": m.started_at.isoformat() if m.started_at else None,
            "completed_at": m.completed_at.isoformat() if m.completed_at else None,
            "duration_sec": round(m.duration_seconds, 1) if m.duration_seconds else None,
            "input_tokens": m.total_token_usage.input_tokens,
            "output_tokens": m.total_token_usage.output_tokens,
            "cached_tokens": m.total_token_usage.cached_input_tokens,
            "reasoning_tokens": m.total_token_usage.reasoning_tokens,
            "total_tokens": m.total_token_usage.total_tokens,
            "cache_rate_pct": cache_rate,
        }

        # Add scores as separate columns
        for score_name, value in m.scores.items():
            row[f"score_{score_name}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


def generate_task_report(m: EvalTaskMetrics) -> str:
    """Generate a detailed report for a single eval task."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"EVAL TASK: {m.task_name}")
    lines.append("=" * 80)

    # File info
    lines.append(f"\nFile: {m.file_path.name}")

    # Basic info
    lines.append(f"\nTask Information:")
    lines.append(f"  Task Name:  {m.task_name}")
    lines.append(f"  Model:      {m.model}")
    lines.append(f"  Status:     {m.status}")
    if m.dataset:
        lines.append(f"  Dataset:    {m.dataset}")
    if m.sandbox_type:
        lines.append(f"  Sandbox:    {m.sandbox_type}")

    # Timing
    lines.append(f"\nTiming:")
    if m.started_at:
        lines.append(f"  Started:    {m.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if m.completed_at:
        lines.append(f"  Completed:  {m.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if m.duration_seconds:
        mins, secs = divmod(m.duration_seconds, 60)
        lines.append(f"  Duration:   {int(mins)}m {secs:.1f}s ({m.duration_seconds:.1f}s total)")

    # Samples
    lines.append(f"\nSamples:")
    lines.append(f"  Completed:  {m.completed_samples}/{m.total_samples}")
    if m.total_samples > 0:
        completion_rate = (m.completed_samples / m.total_samples) * 100
        lines.append(f"  Completion: {completion_rate:.1f}%")

    # Token usage
    lines.append(f"\nToken Usage (Aggregate):")
    lines.append(f"  Input tokens:     {m.total_token_usage.input_tokens:,}")
    lines.append(f"  Output tokens:    {m.total_token_usage.output_tokens:,}")
    lines.append(f"  Cached tokens:    {m.total_token_usage.cached_input_tokens:,}")
    lines.append(f"  Reasoning tokens: {m.total_token_usage.reasoning_tokens:,}")
    lines.append(f"  Total tokens:     {m.total_token_usage.total_tokens:,}")

    if m.total_token_usage.input_tokens > 0:
        cache_rate = (m.total_token_usage.cached_input_tokens / m.total_token_usage.input_tokens) * 100
        lines.append(f"  Cache hit rate:   {cache_rate:.1f}%")

    # Per-model breakdown
    if m.token_usage:
        lines.append(f"\nToken Usage by Model:")
        for model_name, usage in m.token_usage.items():
            lines.append(f"\n  {model_name}:")
            lines.append(f"    Input:     {usage.input_tokens:,}")
            lines.append(f"    Output:    {usage.output_tokens:,}")
            lines.append(f"    Cached:    {usage.cached_input_tokens:,}")
            lines.append(f"    Reasoning: {usage.reasoning_tokens:,}")
            lines.append(f"    Total:     {usage.total_tokens:,}")

    # Scores
    if m.scores:
        lines.append(f"\nScores:")
        for score_name, value in sorted(m.scores.items()):
            lines.append(f"  {score_name}: {value:.4f}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def generate_summary_report(metrics: list[EvalTaskMetrics], detailed: bool = False) -> str:
    """Generate a human-readable summary report of eval metrics.

    Args:
        metrics: List of EvalTaskMetrics to summarize
        detailed: If True, include full per-task reports. If False, show compact summary.
    """
    if not metrics:
        return "No eval files found to analyze."

    lines = []
    lines.append("=" * 80)
    lines.append("EVALUATION RUN SUMMARY")
    lines.append("=" * 80)

    # Overall stats
    total_duration = sum(m.duration_seconds or 0 for m in metrics)
    total_samples = sum(m.completed_samples for m in metrics)
    total_input = sum(m.total_token_usage.input_tokens for m in metrics)
    total_output = sum(m.total_token_usage.output_tokens for m in metrics)
    total_cached = sum(m.total_token_usage.cached_input_tokens for m in metrics)
    total_tokens = sum(m.total_token_usage.total_tokens for m in metrics)

    lines.append(f"\nTotal eval files: {len(metrics)}")
    lines.append(f"Total samples completed: {total_samples}")
    mins, secs = divmod(total_duration, 60)
    lines.append(f"Total duration: {int(mins)}m {secs:.1f}s ({total_duration:.1f}s)")
    lines.append("\nAggregate Token Usage:")
    lines.append(f"  Input tokens:  {total_input:,}")
    lines.append(f"  Output tokens: {total_output:,}")
    lines.append(f"  Cached tokens: {total_cached:,}")
    lines.append(f"  Total tokens:  {total_tokens:,}")

    if total_input > 0:
        cache_rate = (total_cached / total_input) * 100
        lines.append(f"  Cache hit rate: {cache_rate:.1f}%")

    lines.append("\n" + "=" * 80)

    if detailed:
        # Full detailed report for each task
        for m in metrics:
            lines.append("\n")
            lines.append(generate_task_report(m))
    else:
        # Compact per-task breakdown
        lines.append("\nPER-TASK BREAKDOWN")
        lines.append("-" * 80)

        for m in metrics:
            lines.append(f"\n{m.task_name}")
            lines.append(f"  Model: {m.model}")
            lines.append(f"  Status: {m.status}")
            lines.append(f"  Samples: {m.completed_samples}/{m.total_samples}")

            if m.duration_seconds:
                lines.append(f"  Duration: {m.duration_seconds:.1f}s")

            lines.append(f"  Tokens: {m.total_token_usage.total_tokens:,} total "
                         f"({m.total_token_usage.input_tokens:,} in, "
                         f"{m.total_token_usage.output_tokens:,} out, "
                         f"{m.total_token_usage.cached_input_tokens:,} cached)")

            # Per-model breakdown if multiple models
            if len(m.token_usage) > 1:
                lines.append("  Per-model usage:")
                for model_name, usage in m.token_usage.items():
                    short_name = model_name.split("/")[-1] if "/" in model_name else model_name
                    lines.append(f"    {short_name}: {usage.total_tokens:,} tokens")

            # Scores
            if m.scores:
                lines.append("  Scores:")
                for score_name, value in m.scores.items():
                    lines.append(f"    {score_name}: {value:.4f}")

        lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def analyze_and_report(
    log_dir: Path, save_csv: bool = True, detailed: bool = False
) -> tuple[list[EvalTaskMetrics], str]:
    """Analyze a log directory and generate a report.

    Args:
        log_dir: Directory containing .eval files
        save_csv: If True, save metrics to a CSV file
        detailed: If True, generate detailed per-task reports

    Returns:
        Tuple of (metrics list, report string)
    """
    metrics = analyze_log_directory(log_dir)
    report = generate_summary_report(metrics, detailed=detailed)

    if save_csv and metrics:
        df = metrics_to_dataframe(metrics)
        csv_path = log_dir / "eval_metrics.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved metrics to {csv_path}")

    return metrics, report
