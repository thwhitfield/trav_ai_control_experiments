"""Decomposed AI Control evaluation for post-hoc analysis.

Generates component outputs independently across four phases:
  Phase 1: Untrusted model responses (honest + attack)
  Phase 2: Post-hoc monitoring scores (trusted + untrusted monitors)
  Phase 3: Trusted model responses (honest baseline)
  Phase 4: Trusted editing responses (always edits, for post-hoc threshold analysis)

This approach enables post-hoc analysis with arbitrary defer thresholds
and attack strategies without re-running evaluations.

Usage:
    python scripts/ai_control_paper2.py configs/ai_control_paper2.yaml
"""

import argparse
import asyncio
import shutil
import sys
from datetime import datetime
from pathlib import Path

from control_arena import EvalMode
from control_arena.eval import (
    ControlEvalConfig,
    evaluate_monitors_post_hoc,
    get_control_task,
)
from control_arena.micro_protocol import trusted_editing_builder
from control_arena.scorers import max_suspicion_scorer
from control_arena.settings.apps.apps_setting import AppsSetting
from inspect_ai import eval_set

from ai_control_paper import (
    PromptLoader,
    create_monitors,
    create_policies,
)
from trav_ai_control.config import (
    DecomposedExperimentConfig,
    load_decomposed_config,
)


def make_eval_config(
    config: DecomposedExperimentConfig,
    prompts: PromptLoader,
    phase_log_dir: str,
) -> ControlEvalConfig:
    """Create a ControlEvalConfig for a given phase log directory."""
    return ControlEvalConfig(
        limit=config.evaluation.limit,
        log_dir=phase_log_dir,
        trusted_model=config.models.trusted,
        untrusted_model=config.models.untrusted,
        system_prompt=prompts.honest_prompt,
        max_sandboxes=config.sandbox.max_sandboxes,
        max_tasks=config.sandbox.max_tasks,
        max_samples=config.sandbox.max_samples,
    )


def run_phase1(
    config: DecomposedExperimentConfig,
    log_dir: Path,
    prompts: PromptLoader,
    untrusted_policy,
    setting: AppsSetting,
) -> None:
    """Phase 1: Generate untrusted model responses in honest and attack modes."""
    print("\n===== Phase 1: Untrusted Responses =====")

    phase_log_dir = str(log_dir / "phase1_untrusted")
    eval_config = make_eval_config(config, prompts, phase_log_dir)

    tasks = []
    for mode in [EvalMode.HONEST, EvalMode.ATTACK]:
        task_name = f"untrusted_{mode.value}"
        tasks.append(
            get_control_task(
                setting=setting,
                micro_protocol=untrusted_policy,
                mode=mode,
                eval_config=eval_config,
                monitoring_scorers=[],
                metadata={
                    "phase": "untrusted_responses",
                    "protocol_name": "untrusted",
                },
                task_name=task_name,
            )
        )

    print(f"Running {len(tasks)} Phase 1 tasks...")
    eval_set(tasks=tasks, **eval_config.eval_args())
    print("Phase 1 complete.")


def run_phase2(
    config: DecomposedExperimentConfig,
    log_dir: Path,
    trusted_monitor,
    untrusted_monitor,
) -> None:
    """Phase 2: Run post-hoc monitoring on Phase 1 eval logs."""
    print("\n===== Phase 2: Post-hoc Monitoring =====")

    phase1_dir = log_dir / "phase1_untrusted"
    phase2_dir = log_dir / "phase2_monitoring"

    eval_files = sorted(phase1_dir.glob("*.eval"))
    if not eval_files:
        print("WARNING: No eval files found in Phase 1 directory. Skipping.")
        return

    monitors = []
    if config.monitor.run_trusted_monitor:
        monitors.append(trusted_monitor)
    if config.monitor.run_untrusted_monitor:
        monitors.append(untrusted_monitor)

    if not monitors:
        print("No monitors configured. Skipping Phase 2.")
        return

    log_paths = [str(f) for f in eval_files]

    print(
        f"Running {len(monitors)} monitor(s) on {len(eval_files)} eval file(s)..."
    )
    asyncio.run(
        evaluate_monitors_post_hoc(
            log_paths=log_paths,
            monitors=monitors,
            output_dir=str(phase2_dir),
            verbose=True,
        )
    )
    print("Phase 2 complete.")


def run_phase3(
    config: DecomposedExperimentConfig,
    log_dir: Path,
    prompts: PromptLoader,
    trusted_policy,
    setting: AppsSetting,
) -> None:
    """Phase 3: Generate trusted model responses in honest mode."""
    print("\n===== Phase 3: Trusted Responses =====")

    phase_log_dir = str(log_dir / "phase3_trusted")
    eval_config = make_eval_config(config, prompts, phase_log_dir)

    task = get_control_task(
        setting=setting,
        micro_protocol=trusted_policy,
        mode=EvalMode.HONEST,
        eval_config=eval_config,
        monitoring_scorers=[],
        metadata={
            "phase": "trusted_responses",
            "protocol_name": "trusted_only",
        },
        task_name="trusted_honest",
    )

    print("Running 1 Phase 3 task...")
    eval_set(tasks=[task], **eval_config.eval_args())
    print("Phase 3 complete.")


def run_phase4(
    config: DecomposedExperimentConfig,
    log_dir: Path,
    prompts: PromptLoader,
    untrusted_policy,
    edit_policy,
    trusted_monitor,
    setting: AppsSetting,
) -> None:
    """Phase 4: Generate trusted editing responses (always edits every response)."""
    print("\n===== Phase 4: Trusted Editing =====")

    phase_log_dir = str(log_dir / "phase4_editing")
    eval_config = make_eval_config(config, prompts, phase_log_dir)

    # edit_threshold=-1.0 guarantees all responses get edited
    # (protocol checks suspicion_score > edit_threshold, so -1.0 always triggers)
    te_protocol = trusted_editing_builder(
        edit_threshold=-1.0,
        untrusted_policy=untrusted_policy,
        trusted_editor_policy=edit_policy,
        monitor=trusted_monitor,
    )

    tasks = []
    for mode in [EvalMode.HONEST, EvalMode.ATTACK]:
        task_name = f"trusted_edit_{mode.value}"
        tasks.append(
            get_control_task(
                setting=setting,
                micro_protocol=te_protocol,
                mode=mode,
                eval_config=eval_config,
                monitoring_scorers=[max_suspicion_scorer()],
                metadata={
                    "phase": "trusted_editing",
                    "protocol_name": "trusted_editing",
                    "edit_threshold": -1.0,
                },
                task_name=task_name,
            )
        )

    print(f"Running {len(tasks)} Phase 4 tasks...")
    eval_set(tasks=tasks, **eval_config.eval_args())
    print("Phase 4 complete.")


def main(config_path: Path) -> None:
    """Run decomposed AI Control evaluation.

    Args:
        config_path: Path to YAML configuration file.
    """
    config = load_decomposed_config(config_path)

    # Generate log directory
    effective_log_dir = config.evaluation.log_dir
    if effective_log_dir is None:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        effective_log_dir = f"logs/ai_control_decomposed_{datetime_str}"
        print(f"Logging to {effective_log_dir} (auto-generated)")
    else:
        print(f"Logging to {effective_log_dir}")

    log_dir = Path(effective_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, log_dir / "config.yaml")

    # Create shared components
    prompts = PromptLoader(
        attack_prompt_path=config.prompts.attack,
        edit_prompt_path=config.prompts.edit,
        honest_prompt_path=config.prompts.honest,
        monitor_prompt_path=config.prompts.monitor,
    )

    untrusted_policy, trusted_policy, edit_policy = create_policies(
        prompts=prompts,
        cache_enabled=config.cache.enabled,
        use_dataset_solutions=config.policy.use_dataset_solutions,
    )
    trusted_monitor, untrusted_monitor = create_monitors(
        prompts=prompts,
        num_samples=config.monitor.num_samples,
    )

    setting = AppsSetting(
        cache_python_code=config.cache.enabled,
        has_backdoor_solution=config.setting.has_backdoor_solution,
    )

    # Print configuration summary
    print(f"\nEvaluation limit: {config.evaluation.limit}")
    print(f"Trusted model: {config.models.trusted}")
    print(f"Untrusted model: {config.models.untrusted}")
    print(f"Phases: ", end="")
    phases = []
    if config.phases.run_untrusted_responses:
        phases.append("1:untrusted")
    if config.phases.run_post_hoc_monitoring:
        phases.append("2:monitoring")
    if config.phases.run_trusted_responses:
        phases.append("3:trusted")
    if config.phases.run_trusted_editing:
        phases.append("4:editing")
    print(", ".join(phases) if phases else "none")

    # Run enabled phases
    if config.phases.run_untrusted_responses:
        run_phase1(config, log_dir, prompts, untrusted_policy, setting)

    if config.phases.run_post_hoc_monitoring:
        run_phase2(config, log_dir, trusted_monitor, untrusted_monitor)

    if config.phases.run_trusted_responses:
        run_phase3(config, log_dir, prompts, trusted_policy, setting)

    if config.phases.run_trusted_editing:
        run_phase4(
            config,
            log_dir,
            prompts,
            untrusted_policy,
            edit_policy,
            trusted_monitor,
            setting,
        )

    print(f"\nDecomposed evaluation complete!")
    print(f"Results saved to: {effective_log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run decomposed AI Control evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/ai_control_paper2.py configs/ai_control_paper2.yaml
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
