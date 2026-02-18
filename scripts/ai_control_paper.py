"""AI Control paper replication using ControlArena built-in components.

This example replicates the "AI Control: Improving Safety Despite Intentional Subversion" paper
evaluation using APPS backdooring tasks. It demonstrates defer-to-trusted protocols with
various threshold configurations to explore the safety-usefulness tradeoff.

Usage:
    python scripts/ai_control_paper.py configs/ai_control_paper.yaml
"""

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from control_arena import ControlAgent, EvalMode
from control_arena.analysis import samples_df
from control_arena.eval import ControlEvalConfig, get_control_task
from control_arena.micro_protocol import (
    defer_to_trusted_builder,
    trusted_editing_builder,
)
from control_arena.monitor import ensemble_monitor_builder
from control_arena.policy import (
    attack_policy_builder,
    honest_policy_builder,
    trusted_editor_builder,
    trusted_policy_builder,
    untrusted_policy_builder,
)
from control_arena.scorers import max_suspicion_scorer
from control_arena.settings.apps.apps_setting import AppsSetting
from control_arena.settings.apps.apps_solution_policy import apps_solution_policy
from control_arena.settings.apps.task.apps_trusted_editor import (
    update_edited_state_from_untrusted_state_apps,
)
from inspect_ai import eval_set

from trav_ai_control.analysis import analyze_and_report
from trav_ai_control.config import (
    ExperimentConfig,
    load_config,
)
from trav_ai_control.metrics import create_protocol_results_dataframe
from trav_ai_control.plotting import plot_safety_vs_usefulness_by_protocol

PROMPTS_DIR = Path(__file__).parent / "prompts"
ATTACK_PROMPT_PATH = PROMPTS_DIR / "attack_system_prompt.txt"
EDIT_PROMPT_PATH = PROMPTS_DIR / "edit_prompt.txt"
HONEST_PROMPT_PATH = PROMPTS_DIR / "honest_system_prompt.txt"
MONITOR_PROMPT_PATH = PROMPTS_DIR / "monitor_prompt.txt"


@dataclass
class PromptLoader:
    """Loads prompts from file paths with defaults."""

    attack_prompt_path: Path | None = None
    edit_prompt_path: Path | None = None
    honest_prompt_path: Path | None = None
    monitor_prompt_path: Path | None = None

    def __post_init__(self):
        self.attack_prompt_path = self.attack_prompt_path or ATTACK_PROMPT_PATH
        self.edit_prompt_path = self.edit_prompt_path or EDIT_PROMPT_PATH
        self.honest_prompt_path = self.honest_prompt_path or HONEST_PROMPT_PATH
        self.monitor_prompt_path = self.monitor_prompt_path or MONITOR_PROMPT_PATH

    @property
    def attack_prompt(self) -> str:
        assert self.attack_prompt_path is not None
        return self.attack_prompt_path.read_text()

    @property
    def edit_prompt(self) -> str:
        assert self.edit_prompt_path is not None
        return self.edit_prompt_path.read_text()

    @property
    def honest_prompt(self) -> str:
        assert self.honest_prompt_path is not None
        return self.honest_prompt_path.read_text()

    @property
    def monitor_prompt(self) -> str:
        assert self.monitor_prompt_path is not None
        return self.monitor_prompt_path.read_text()


def create_policies(
    prompts: PromptLoader,
    cache_enabled: bool,
    use_dataset_solutions: bool = True,
) -> tuple[ControlAgent, ControlAgent, ControlAgent]:
    """Create basic policies used in control protocols for the AI Control paper.

    Args:
        prompts: Prompt loader with all prompt files.
        cache_enabled: Whether to enable caching for the trusted policy.
        use_dataset_solutions: If True, use pre-computed solutions from the dataset
            (faster, deterministic). If False, have the untrusted model generate
            solutions itself using attack/honest prompts (slower, tests model capability).

    Returns:
        Tuple of (untrusted_policy, trusted_policy, trusted_editor_policy)
    """
    trusted_policy = trusted_policy_builder(cache=cache_enabled)

    if use_dataset_solutions:
        # Use pre-computed solutions from the dataset (faster, deterministic)
        untrusted = apps_solution_policy()
    else:
        # Have the untrusted model generate solutions itself
        attack = attack_policy_builder(prompt=prompts.attack_prompt)
        honest = honest_policy_builder()
        untrusted = untrusted_policy_builder(attack=attack, honest=honest)

    # Create the trusted editor policy
    trusted_editor_policy = trusted_editor_builder(
        trusted_policy=trusted_policy,
        prompt=prompts.edit_prompt,
        update_edited_state_from_untrusted_state=update_edited_state_from_untrusted_state_apps,
    )

    return untrusted, trusted_policy, trusted_editor_policy


def create_monitors(
    prompts: PromptLoader,
    num_samples: int,
) -> tuple[ControlAgent, ControlAgent]:
    """Create monitors used in control protocols for the AI Control paper."""
    # Define ensemble monitors for better precision
    trusted_monitor = ensemble_monitor_builder(
        prompt=prompts.monitor_prompt, num_samples=num_samples
    )
    untrusted_monitor = ensemble_monitor_builder(
        model_role="untrusted", prompt=prompts.monitor_prompt, num_samples=num_samples
    )
    return trusted_monitor, untrusted_monitor


def create_micro_protocols(
    trusted_policy: ControlAgent,
    untrusted_policy: ControlAgent,
    edit_policy: ControlAgent,
    trusted_monitor: ControlAgent,
    untrusted_monitor: ControlAgent,
    config: ExperimentConfig,
) -> list[tuple[dict[str, str | float], ControlAgent, list[EvalMode]]]:
    """Create control protocols with threshold sweeps.

    Creates control protocols for the AI Control paper replication, including:
    - Baseline: trusted-only and upfront audit policies (if enabled)
    - Trusted monitoring + defer to trusted (threshold sweep) - if enabled
    - Untrusted monitoring + defer to trusted (threshold sweep) - if enabled
    - Trusted monitoring + defer to trusted editing (threshold sweep) - if enabled

    Args:
        trusted_policy: The trusted policy agent builder.
        untrusted_policy: The untrusted policy agent builder.
        edit_policy: The trusted editor policy agent builder.
        trusted_monitor: The trusted monitor agent builder.
        untrusted_monitor: The untrusted monitor agent builder.
        config: Full experiment configuration.

    Returns:
        List of tuples containing protocol metadata, protocol builder, and required eval modes.
    """
    micro_protocols: list[
        tuple[dict[str, str | float], ControlAgent, list[EvalMode]]
    ] = []

    # Baseline: Trusted model only - only needs honest mode
    if config.protocols.baselines.run_trusted_only:
        micro_protocols.append(
            ({"protocol_name": "trusted_only"}, trusted_policy, [EvalMode.HONEST])
        )

    # Baseline: Upfront auditing - needs both modes for analysis
    # This will contain our untrusted-only baseline, which runs the untrusted model
    # in honest mode to act as our upper bound of usefulness.
    if config.protocols.baselines.run_upfront_audit:
        micro_protocols.append(
            (
                {"protocol_name": "upfront_audit"},
                untrusted_policy,
                [EvalMode.HONEST, EvalMode.ATTACK],
            )
        )

    # Trusted Monitoring + Defer to Trusted
    if config.protocols.run_trusted_monitor_defer:
        for threshold in config.protocols.trusted_monitor_defer_thresholds:
            micro_protocols.append(
                (
                    {
                        "protocol_name": "trusted_monitor_defer",
                        "defer_threshold": threshold,
                    },
                    defer_to_trusted_builder(
                        defer_threshold=threshold,
                        untrusted_policy=untrusted_policy,
                        trusted_policy=trusted_policy,
                        monitor=trusted_monitor,
                    ),
                    [EvalMode.HONEST, EvalMode.ATTACK],
                )
            )

    # Untrusted Monitoring + Defer to Trusted
    if config.protocols.run_untrusted_monitor_defer:
        for threshold in config.protocols.untrusted_monitor_defer_thresholds:
            micro_protocols.append(
                (
                    {
                        "protocol_name": "untrusted_monitor_defer",
                        "defer_threshold": threshold,
                    },
                    defer_to_trusted_builder(
                        defer_threshold=threshold,
                        untrusted_policy=untrusted_policy,
                        trusted_policy=trusted_policy,
                        monitor=untrusted_monitor,
                    ),
                    [EvalMode.HONEST, EvalMode.ATTACK],
                )
            )

    # Trusted Monitoring + Defer to Trusted Editing
    if config.protocols.run_trusted_monitor_edit:
        for threshold in config.protocols.trusted_monitor_edit_thresholds:
            micro_protocols.append(
                (
                    {
                        "protocol_name": "trusted_monitor_edit",
                        "defer_threshold": threshold,
                    },
                    trusted_editing_builder(
                        edit_threshold=threshold,
                        untrusted_policy=untrusted_policy,
                        trusted_editor_policy=edit_policy,
                        monitor=trusted_monitor,
                    ),
                    [EvalMode.HONEST, EvalMode.ATTACK],
                )
            )

    return micro_protocols


def create_evaluation_tasks(
    eval_config: ControlEvalConfig,
    setting: AppsSetting,
    config: ExperimentConfig,
    prompts: PromptLoader,
):
    """Create all evaluation tasks for the AI Control paper replication.

    Creates policies and monitors, then builds tasks for all protocol combinations.

    Args:
        eval_config: Evaluation configuration for inspect_ai.
        setting: Task setting (e.g., AppsSetting).
        config: Full experiment configuration.
        prompts: Prompt loader with all prompts.
    """
    # Create policies
    untrusted_policy, trusted_policy, edit_policy = create_policies(
        prompts=prompts,
        cache_enabled=config.cache.enabled,
        use_dataset_solutions=config.policy.use_dataset_solutions,
    )

    # Create monitors
    trusted_monitor, untrusted_monitor = create_monitors(
        prompts=prompts,
        num_samples=config.monitor.num_samples,
    )

    # Collect all protocols to evaluate
    micro_protocols = create_micro_protocols(
        trusted_policy=trusted_policy,
        untrusted_policy=untrusted_policy,
        edit_policy=edit_policy,
        trusted_monitor=trusted_monitor,
        untrusted_monitor=untrusted_monitor,
        config=config,
    )

    # Create tasks for all protocol combinations
    eval_tasks = []
    for metadata, micro_protocol_builder, eval_modes in micro_protocols:
        # Build a descriptive task name from metadata
        protocol_name = metadata.get("protocol_name", "unknown")
        defer_threshold = metadata.get("defer_threshold")

        # Create a task for each eval mode
        for mode in eval_modes:
            # Build clean task name: protocol_t{threshold}_{mode}
            if defer_threshold is not None:
                task_name = f"{protocol_name}_t{defer_threshold:.2f}_{mode.value}"
            else:
                task_name = f"{protocol_name}_{mode.value}"

            task = get_control_task(
                setting=setting,
                micro_protocol=micro_protocol_builder,
                mode=mode,
                eval_config=eval_config,
                monitoring_scorers=[max_suspicion_scorer()],
                metadata=metadata,
                task_name=task_name,
            )
            eval_tasks.append(task)

    print(f"Created {len(eval_tasks)} evaluation tasks total")
    return eval_tasks


def run_evaluation(
    config: ExperimentConfig,
    log_dir: str,
    prompts: PromptLoader,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run AI Control evaluation with all protocols from the paper.

    Args:
        config: Full experiment configuration.
        log_dir: Directory to save logs and results.
        prompts: Prompt loader with all prompts.

    Returns:
        Tuple of (samples_df, results_df, plot_path)
    """
    # Get honest prompt for eval config
    honest_system_prompt = prompts.honest_prompt

    # ======= Setup Evaluation =======
    eval_config = ControlEvalConfig(
        limit=config.evaluation.limit,
        log_dir=log_dir,
        trusted_model=config.models.trusted,
        untrusted_model=config.models.untrusted,
        system_prompt=honest_system_prompt,
        max_sandboxes=config.sandbox.max_sandboxes,
        max_tasks=config.sandbox.max_tasks,
        max_samples=config.sandbox.max_samples,
    )

    # Use APPS setting for backdooring tasks
    setting = AppsSetting(
        cache_python_code=config.cache.enabled,
        has_backdoor_solution=config.setting.has_backdoor_solution,
    )

    # Create all evaluation tasks
    eval_tasks = create_evaluation_tasks(
        eval_config=eval_config,
        setting=setting,
        config=config,
        prompts=prompts,
    )

    # ======= Run Evaluations =======
    _ = eval_set(
        tasks=eval_tasks,
        **eval_config.eval_args(),
    )

    # ======= Analyse Results =======
    df = samples_df(log_dir)
    print(f"Evaluation completed. Results saved to {log_dir}")
    print(f"Analyzed {len(df)} samples")

    # Extract protocol_name and defer_threshold from metadata_eval
    df["protocol_name"] = df["metadata_eval"].apply(
        lambda x: x.get("protocol_name", None) if isinstance(x, dict) else None
    )
    df["defer_threshold"] = df["metadata_eval"].apply(
        lambda x: x.get("defer_threshold", None) if isinstance(x, dict) else None
    )

    # Create analysis plots
    # Compute results by protocol and save to a new DataFrame
    results_df = create_protocol_results_dataframe(
        df, config.evaluation.audit_budget
    )  # type: ignore

    # Create comparison plot
    fig = plot_safety_vs_usefulness_by_protocol(results_df, title=config.plot.title)

    # Save the plot
    plot_path = Path(log_dir) / "ai_control_safety_vs_usefulness.png"
    fig.savefig(plot_path.as_posix(), dpi=config.plot.dpi, bbox_inches="tight")
    print(f"Saved plot to {plot_path.as_posix()}")

    # Save results CSV
    results_csv_path = Path(log_dir) / "results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved results to {results_csv_path.as_posix()}")

    return df, results_df, plot_path


def main(config_path: Path) -> None:
    """Run AI Control paper replication evaluation.

    This implements the key protocols from the AI Control paper:
    - Baseline: trusted-only and upfront audit policies
    - Trusted monitoring + defer to trusted (threshold sweep)
    - Untrusted monitoring + defer to trusted (threshold sweep)
    - Trusted monitoring + defer to trusted editing (threshold sweep)

    Args:
        config_path: Path to YAML configuration file.
    """
    # Load config from file
    config = load_config(config_path)

    # Generate log_dir if not specified
    effective_log_dir = config.evaluation.log_dir
    if effective_log_dir is None:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        effective_log_dir = f"logs/ai_control_paper_{datetime_str}"
        print(f"Logging to {effective_log_dir} (auto-generated)")
    else:
        print(f"Logging to {effective_log_dir}")

    # Print configuration summary
    print(f"Running with audit budget: {config.evaluation.audit_budget}%")
    print(f"Trusted monitor defer thresholds: {config.protocols.trusted_monitor_defer_thresholds}")
    print(f"Untrusted monitor defer thresholds: {config.protocols.untrusted_monitor_defer_thresholds}")
    print(f"Trusted monitor edit thresholds: {config.protocols.trusted_monitor_edit_thresholds}")
    print(f"Trusted model: {config.models.trusted}")
    print(f"Untrusted model: {config.models.untrusted}")
    print(f"Sandbox: max_sandboxes={config.sandbox.max_sandboxes}, "
          f"max_tasks={config.sandbox.max_tasks}, "
          f"max_samples={config.sandbox.max_samples}")
    print(f"Monitor: num_samples={config.monitor.num_samples}")

    # Ensure log directory exists and copy config file
    log_dir_path = Path(effective_log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, log_dir_path / "config.yaml")

    # Create prompt loader
    prompts = PromptLoader(
        attack_prompt_path=config.prompts.attack,
        edit_prompt_path=config.prompts.edit,
        honest_prompt_path=config.prompts.honest,
        monitor_prompt_path=config.prompts.monitor,
    )

    # Run evaluation
    raw_df, results_df, plot_path = run_evaluation(
        config=config,
        log_dir=effective_log_dir,
        prompts=prompts,
    )

    # Run analysis on eval files
    log_dir_path = Path(effective_log_dir)
    metrics, report = analyze_and_report(log_dir_path, save_csv=True, detailed=True)

    # Print the analysis report
    print("\n" + report)

    # Generate HTML and Markdown summaries
    from trav_ai_control.summary import (
        analyze_log_directory as analyze_for_html,
    )
    from trav_ai_control.summary import (
        generate_html_report,
        generate_markdown_report,
        load_run_config,
    )

    summaries = analyze_for_html(log_dir_path)
    run_config = load_run_config(log_dir_path)

    html = generate_html_report(log_dir_path, summaries, run_config)
    html_path = log_dir_path / "summary.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Generated HTML summary: {html_path}")

    markdown = generate_markdown_report(log_dir_path, summaries, run_config)
    md_path = log_dir_path / "summary.md"
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"Generated Markdown summary: {md_path}")

    print("\nEvaluation complete!")
    print(f"Total samples: {len(raw_df)}")
    if len(raw_df) > 0:
        print(f"Protocols evaluated: {sorted(raw_df['protocol_name'].unique())}")
        print(f"Results and plots saved to: {effective_log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AI Control paper replication evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/ai_control_paper.py configs/ai_control_paper.yaml
    python scripts/ai_control_paper.py configs/my_custom_config.yaml

All configuration is done via the YAML config file. See configs/ai_control_paper.yaml
for the default configuration and available options.
        """,
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    main(args.config)
