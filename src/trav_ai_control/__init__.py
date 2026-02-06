"""AI Control experiments package."""

from trav_ai_control.config import (
    ExperimentConfig,
    config_to_flat_dict,
    load_config,
    merge_cli_overrides,
)
from trav_ai_control.metrics import (
    AuditResult,
    audit_runs,
    create_protocol_results_dataframe,
)
from trav_ai_control.plotting import plot_safety_vs_usefulness_by_protocol
from trav_ai_control.tracking import (
    ExperimentTracker,
    log_protocol_metrics,
    tracking_context,
)

__all__ = [
    # Config
    "ExperimentConfig",
    "load_config",
    "merge_cli_overrides",
    "config_to_flat_dict",
    # Metrics
    "AuditResult",
    "audit_runs",
    "create_protocol_results_dataframe",
    # Plotting
    "plot_safety_vs_usefulness_by_protocol",
    # Tracking
    "ExperimentTracker",
    "tracking_context",
    "log_protocol_metrics",
]
