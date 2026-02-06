"""AI Control experiments package."""

from trav_ai_control.analysis import (
    EvalTaskMetrics,
    analyze_and_report,
    analyze_log_directory,
    generate_summary_report,
    generate_task_report,
    metrics_to_dataframe,
)
from trav_ai_control.config import (
    ExperimentConfig,
    load_config,
)
from trav_ai_control.metrics import (
    AuditResult,
    audit_runs,
    create_protocol_results_dataframe,
)
from trav_ai_control.plotting import plot_safety_vs_usefulness_by_protocol
from trav_ai_control.summary import (
    RunConfig,
    TaskSummary,
    generate_html_report,
    generate_markdown_report,
    load_run_config,
)
from trav_ai_control.summary import (
    analyze_log_directory as analyze_for_html,
)

__all__ = [
    # Analysis
    "EvalTaskMetrics",
    "analyze_log_directory",
    "analyze_and_report",
    "generate_summary_report",
    "generate_task_report",
    "metrics_to_dataframe",
    # Config
    "ExperimentConfig",
    "load_config",
    # Metrics
    "AuditResult",
    "audit_runs",
    "create_protocol_results_dataframe",
    # Plotting
    "plot_safety_vs_usefulness_by_protocol",
    # Summary
    "RunConfig",
    "TaskSummary",
    "analyze_for_html",
    "generate_html_report",
    "generate_markdown_report",
    "load_run_config",
]
