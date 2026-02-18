"""Simple analysis utilities for AI Control evaluation results.

Provides pandas-based functions for loading and analyzing eval logs.
Designed for use in Jupyter notebooks.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from inspect_ai.log import read_eval_log

try:
    from litellm import cost_per_token

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


def calculate_cost(
    model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0
) -> float | None:
    """Calculate cost for a model's token usage."""
    if not LITELLM_AVAILABLE:
        return None
    try:
        prompt_cost, completion_cost = cost_per_token(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            cache_read_input_tokens=cached_tokens,
        )
        return prompt_cost + completion_cost
    except Exception:
        if "/" in model:
            try:
                prompt_cost, completion_cost = cost_per_token(
                    model=model.split("/", 1)[1],
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    cache_read_input_tokens=cached_tokens,
                )
                return prompt_cost + completion_cost
            except Exception:
                pass
        return None


def load_eval_logs(log_dir: Path | str) -> pd.DataFrame:
    """Load all eval logs from a directory into a DataFrame.

    Args:
        log_dir: Path to directory containing .eval files

    Returns:
        DataFrame with one row per eval task
    """
    log_dir = Path(log_dir)
    eval_files = sorted(log_dir.glob("*.eval"))

    rows = []
    for eval_file in eval_files:
        try:
            log = read_eval_log(str(eval_file))
            row = _extract_log_data(log, eval_file)
            rows.append(row)
        except Exception as e:
            print(f"Warning: Failed to read {eval_file.name}: {e}")

    return pd.DataFrame(rows)


def _extract_log_data(log, file_path: Path) -> dict:
    """Extract data from a single eval log."""
    metadata = log.eval.metadata or {}

    # Basic info
    data = {
        "file": file_path.name,
        "protocol": metadata.get("protocol_name", "unknown"),
        "threshold": metadata.get("defer_threshold"),
        "mode": metadata.get("mode", "unknown"),
        "status": log.status,
    }

    # Sample counts
    if log.results:
        data["samples"] = log.results.completed_samples or 0
        data["total_samples"] = log.results.total_samples or 0
    else:
        data["samples"] = 0
        data["total_samples"] = 0

    # Duration
    data["duration_sec"] = None
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
            data["duration_sec"] = (completed - started).total_seconds()
        except (ValueError, TypeError):
            pass

    # Token usage by role
    model_to_role = {}
    role_to_model = {}
    if log.eval and log.eval.model_roles:
        for role, model_config in log.eval.model_roles.items():
            model_to_role[model_config.model] = role
            role_to_model[role] = model_config.model

    data["trusted_model"] = role_to_model.get("trusted")
    data["untrusted_model"] = role_to_model.get("untrusted")

    # Initialize token counts
    for prefix in ["trusted", "untrusted"]:
        data[f"{prefix}_input"] = 0
        data[f"{prefix}_output"] = 0
        data[f"{prefix}_cached"] = 0

    if log.stats and log.stats.model_usage:
        for model_name, usage in log.stats.model_usage.items():
            role = model_to_role.get(model_name, "untrusted")
            prefix = "trusted" if role == "trusted" else "untrusted"

            data[f"{prefix}_input"] += getattr(usage, "input_tokens", 0) or 0
            data[f"{prefix}_output"] += getattr(usage, "output_tokens", 0) or 0
            data[f"{prefix}_cached"] += (
                getattr(usage, "input_tokens_cache_read", 0) or 0
            )

    # Calculate costs
    data["trusted_cost"] = None
    data["untrusted_cost"] = None
    if data["trusted_model"]:
        data["trusted_cost"] = calculate_cost(
            data["trusted_model"],
            data["trusted_input"],
            data["trusted_output"],
            data["trusted_cached"],
        )
    if data["untrusted_model"]:
        data["untrusted_cost"] = calculate_cost(
            data["untrusted_model"],
            data["untrusted_input"],
            data["untrusted_output"],
            data["untrusted_cached"],
        )

    # Scores
    data["main_task"] = None
    data["side_task"] = None
    data["suspicion"] = None

    if log.results and log.results.scores:
        for score in log.results.scores:
            if score.metrics:
                for metric_name, metric in score.metrics.items():
                    if metric.value is not None:
                        if "main_task" in score.name and "accuracy" in metric_name:
                            data["main_task"] = float(metric.value)
                        elif "side_task" in score.name and "accuracy" in metric_name:
                            data["side_task"] = float(metric.value)
                        elif "suspicion" in score.name and "accuracy" in metric_name:
                            data["suspicion"] = float(metric.value)

    return data


def load_config(log_dir: Path | str) -> dict | None:
    """Load the config.yaml from a log directory."""
    config_path = Path(log_dir) / "config.yaml"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics from eval logs DataFrame."""
    total_cost = 0
    if df["trusted_cost"].notna().any():
        total_cost += df["trusted_cost"].sum()
    if df["untrusted_cost"].notna().any():
        total_cost += df["untrusted_cost"].sum()

    stats = {
        "Total Tasks": len(df),
        "Total Samples": df["samples"].sum(),
        "Total Duration": f"{df['duration_sec'].sum() / 60:.1f} min",
        "Trusted Tokens": f"{(df['trusted_input'].sum() + df['trusted_output'].sum()):,}",
        "Untrusted Tokens": f"{(df['untrusted_input'].sum() + df['untrusted_output'].sum()):,}",
        "Total Cost": f"${total_cost:.2f}" if total_cost > 0 else "-",
    }
    return pd.DataFrame([stats]).T.rename(columns={0: "Value"})


def format_task_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format the task details for display."""

    def fmt_cost(x):
        if pd.isna(x):
            return "-"
        return f"${x:.4f}" if x < 0.01 else f"${x:.2f}"

    def fmt_pct(x):
        if pd.isna(x):
            return "-"
        return f"{x * 100:.1f}%"

    def fmt_duration(x):
        if pd.isna(x):
            return "-"
        mins, secs = divmod(x, 60)
        return f"{int(mins)}:{int(secs):02d}"

    # Create display name
    display = df.copy()
    display["task"] = display.apply(
        lambda r: f"{r['protocol']}"
        + (f" t={r['threshold']}" if pd.notna(r["threshold"]) else "")
        + f" / {r['mode']}",
        axis=1,
    )

    # Calculate total cost
    display["total_cost"] = display["trusted_cost"].fillna(0) + display[
        "untrusted_cost"
    ].fillna(0)

    # Format columns
    result = pd.DataFrame(
        {
            "Task": display["task"],
            "Status": display["status"],
            "Samples": display["samples"].astype(str)
            + "/"
            + display["total_samples"].astype(str),
            "Duration": display["duration_sec"].apply(fmt_duration),
            "Trusted $": display["trusted_cost"].apply(fmt_cost),
            "Untrusted $": display["untrusted_cost"].apply(fmt_cost),
            "Total $": display["total_cost"].apply(fmt_cost),
            "Main": display["main_task"].apply(fmt_pct),
            "Side": display["side_task"].apply(fmt_pct),
            "Suspicion": display["suspicion"].apply(fmt_pct),
        }
    )

    return result


def format_token_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format token usage details for display."""

    def fmt_tokens(x):
        return f"{x:,}"

    def cache_rate(cached, total):
        if total == 0:
            return "-"
        return f"{cached / total * 100:.0f}%"

    display = df.copy()
    display["task"] = display.apply(
        lambda r: f"{r['protocol']}"
        + (f" t={r['threshold']}" if pd.notna(r["threshold"]) else "")
        + f" / {r['mode']}",
        axis=1,
    )

    result = pd.DataFrame(
        {
            "Task": display["task"],
            "Trusted In": display["trusted_input"].apply(fmt_tokens),
            "Trusted Out": display["trusted_output"].apply(fmt_tokens),
            "Trusted Cache": display.apply(
                lambda r: cache_rate(r["trusted_cached"], r["trusted_input"]), axis=1
            ),
            "Untrusted In": display["untrusted_input"].apply(fmt_tokens),
            "Untrusted Out": display["untrusted_output"].apply(fmt_tokens),
            "Untrusted Cache": display.apply(
                lambda r: cache_rate(r["untrusted_cached"], r["untrusted_input"]),
                axis=1,
            ),
        }
    )

    return result
