"""MLflow tracking integration with graceful fallback."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .config import MLflowConfig

logger = logging.getLogger(__name__)

# Check if MLflow is available
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore


class ExperimentTracker:
    """Wrapper for MLflow tracking with graceful fallback."""

    def __init__(self, config: MLflowConfig):
        self.config = config
        self._active = False
        self._run: Any = None

        if config.enabled and not MLFLOW_AVAILABLE:
            logger.warning(
                "MLflow tracking enabled but mlflow package not installed. "
                "Install with: uv sync --extra tracking"
            )

    @property
    def is_active(self) -> bool:
        """Check if tracking is currently active."""
        return self._active and MLFLOW_AVAILABLE and self.config.enabled

    def start_run(self) -> None:
        """Start an MLflow run if tracking is enabled."""
        if not self.config.enabled or not MLFLOW_AVAILABLE:
            return

        if self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)

        mlflow.set_experiment(self.config.experiment_name)
        self._run = mlflow.start_run(run_name=self.config.run_name)
        self._active = True
        logger.info(f"Started MLflow run: {self._run.info.run_id}")

    def end_run(self) -> None:
        """End the current MLflow run."""
        if self.is_active:
            mlflow.end_run()
            self._active = False
            logger.info("Ended MLflow run")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if not self.is_active:
            return
        # MLflow has 500 char limit for param values
        for key, value in params.items():
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:497] + "..."
            mlflow.log_param(key, str_value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to MLflow."""
        if not self.is_active:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(
        self, local_path: str | Path, artifact_path: str | None = None
    ) -> None:
        """Log an artifact file to MLflow."""
        if not self.is_active or not self.config.log_artifacts:
            return
        mlflow.log_artifact(str(local_path), artifact_path)

    def log_config_file(self, config_path: Path | None) -> None:
        """Log the config file used for this run."""
        if config_path and config_path.exists():
            self.log_artifact(config_path, "config")

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on the current run."""
        if not self.is_active:
            return
        mlflow.set_tags(tags)


@contextmanager
def tracking_context(
    config: MLflowConfig,
) -> Generator[ExperimentTracker]:
    """Context manager for MLflow tracking.

    Usage:
        with tracking_context(config) as tracker:
            tracker.log_params(params)
            # run experiment
            tracker.log_metrics(metrics)
            tracker.log_artifact(plot_path)
    """
    tracker = ExperimentTracker(config)
    try:
        tracker.start_run()
        yield tracker
    finally:
        tracker.end_run()


def log_protocol_metrics(
    tracker: ExperimentTracker,
    results_df: pd.DataFrame,
) -> None:
    """Log per-protocol metrics to MLflow.

    Creates metrics like:
    - trusted_monitor_defer.threshold_0.5.safety
    - trusted_monitor_defer.threshold_0.5.usefulness
    """
    if not tracker.is_active:
        return

    for _, row in results_df.iterrows():
        protocol = row["protocol_name"]
        threshold = row.get("defer_threshold")

        # Build metric name prefix
        if threshold is not None and not pd.isna(threshold):
            prefix = f"{protocol}.threshold_{threshold}"
        else:
            prefix = protocol

        tracker.log_metrics(
            {
                f"{prefix}.safety": float(row["safety"]),
                f"{prefix}.usefulness": float(row["usefulness"]),
            }
        )
