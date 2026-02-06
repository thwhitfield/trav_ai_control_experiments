"""Configuration loading and management for AI Control experiments."""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentMetadata:
    """Metadata for experiment tracking."""

    name: str = "ai_control_experiment"
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class EvaluationConfig:
    """Evaluation parameters."""

    limit: int = 50
    audit_budget: float = 1.0
    log_dir: str | None = None


@dataclass
class ModelConfig:
    """Model configuration."""

    trusted: str = "openai/gpt-4.1-mini-2025-04-14"
    untrusted: str = "anthropic/claude-sonnet-4-20250514"


@dataclass
class ProtocolSettings:
    """Protocol configuration."""

    defer_thresholds: list[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.5, 0.8, 1.0]
    )
    run_trusted_monitor_defer: bool = True
    run_untrusted_monitor_defer: bool = True
    run_trusted_monitor_edit: bool = True


@dataclass
class PromptPaths:
    """Paths to prompt files."""

    attack: Path | None = None
    honest: Path | None = None
    monitor: Path | None = None
    edit: Path | None = None


@dataclass
class CacheConfig:
    """Caching configuration."""

    enabled: bool = True


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""

    enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str = "ai_control_experiments"
    run_name: str | None = None
    log_artifacts: bool = True


@dataclass
class ExperimentConfig:
    """Root configuration object combining all settings."""

    experiment: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    protocols: ProtocolSettings = field(default_factory=ProtocolSettings)
    prompts: PromptPaths = field(default_factory=PromptPaths)
    cache: CacheConfig = field(default_factory=CacheConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)


def load_config(config_path: Path | str | None) -> ExperimentConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, returns defaults.

    Returns:
        ExperimentConfig with loaded values or defaults.
    """
    if config_path is None:
        return ExperimentConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return _parse_config(data)


def _parse_config(data: dict[str, Any]) -> ExperimentConfig:
    """Parse raw YAML data into ExperimentConfig."""
    return ExperimentConfig(
        experiment=_parse_dataclass(
            ExperimentMetadata, data.get("experiment", {})
        ),
        evaluation=_parse_dataclass(
            EvaluationConfig, data.get("evaluation", {})
        ),
        models=_parse_dataclass(ModelConfig, data.get("models", {})),
        protocols=_parse_dataclass(ProtocolSettings, data.get("protocols", {})),
        prompts=_parse_prompt_paths(data.get("prompts", {})),
        cache=_parse_dataclass(CacheConfig, data.get("cache", {})),
        mlflow=_parse_dataclass(MLflowConfig, data.get("mlflow", {})),
    )


def _parse_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Parse a dict into a dataclass, ignoring unknown fields."""
    if not data:
        return cls()

    field_names = {f.name for f in fields(cls)}
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered_data)


def _parse_prompt_paths(data: dict[str, Any]) -> PromptPaths:
    """Parse prompt paths, converting strings to Path objects."""
    if not data:
        return PromptPaths()

    return PromptPaths(
        attack=Path(data["attack"]) if data.get("attack") else None,
        honest=Path(data["honest"]) if data.get("honest") else None,
        monitor=Path(data["monitor"]) if data.get("monitor") else None,
        edit=Path(data["edit"]) if data.get("edit") else None,
    )


def merge_cli_overrides(config: ExperimentConfig, **cli_args: Any) -> ExperimentConfig:
    """Merge CLI arguments into config, with CLI taking precedence.

    Args:
        config: Base configuration from YAML
        **cli_args: CLI argument values (None values are ignored)

    Returns:
        New ExperimentConfig with CLI overrides applied.
    """
    # Filter out None values - these mean "use config default"
    overrides = {k: v for k, v in cli_args.items() if v is not None}

    if not overrides:
        return config

    # Create copies of nested configs to avoid mutation
    evaluation = EvaluationConfig(
        limit=overrides.get("limit", config.evaluation.limit),
        audit_budget=overrides.get("audit_budget", config.evaluation.audit_budget),
        log_dir=overrides.get("log_dir", config.evaluation.log_dir),
    )

    models = ModelConfig(
        trusted=overrides.get("trusted_model", config.models.trusted),
        untrusted=overrides.get("untrusted_model", config.models.untrusted),
    )

    protocols = ProtocolSettings(
        defer_thresholds=overrides.get(
            "defer_thresholds", config.protocols.defer_thresholds
        ),
        run_trusted_monitor_defer=overrides.get(
            "run_trusted_monitor_defer", config.protocols.run_trusted_monitor_defer
        ),
        run_untrusted_monitor_defer=overrides.get(
            "run_untrusted_monitor_defer", config.protocols.run_untrusted_monitor_defer
        ),
        run_trusted_monitor_edit=overrides.get(
            "run_trusted_monitor_edit", config.protocols.run_trusted_monitor_edit
        ),
    )

    prompts = PromptPaths(
        attack=overrides.get("attack_prompt_path", config.prompts.attack),
        honest=overrides.get("honest_prompt_path", config.prompts.honest),
        monitor=overrides.get("monitor_prompt_path", config.prompts.monitor),
        edit=overrides.get("edit_prompt_path", config.prompts.edit),
    )

    cache = CacheConfig(
        enabled=overrides.get("cache", config.cache.enabled),
    )

    mlflow = MLflowConfig(
        enabled=overrides.get("mlflow_enabled", config.mlflow.enabled),
        tracking_uri=config.mlflow.tracking_uri,
        experiment_name=config.mlflow.experiment_name,
        run_name=config.mlflow.run_name,
        log_artifacts=config.mlflow.log_artifacts,
    )

    return ExperimentConfig(
        experiment=config.experiment,
        evaluation=evaluation,
        models=models,
        protocols=protocols,
        prompts=prompts,
        cache=cache,
        mlflow=mlflow,
    )


def config_to_flat_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Flatten config to dict for MLflow parameter logging.

    Returns dict with dotted keys like 'models.trusted', 'protocols.defer_thresholds'
    """
    result: dict[str, Any] = {}

    # Experiment metadata
    result["experiment.name"] = config.experiment.name
    result["experiment.description"] = config.experiment.description
    result["experiment.tags"] = ",".join(config.experiment.tags)

    # Evaluation config
    result["evaluation.limit"] = config.evaluation.limit
    result["evaluation.audit_budget"] = config.evaluation.audit_budget
    result["evaluation.log_dir"] = config.evaluation.log_dir or "auto"

    # Models
    result["models.trusted"] = config.models.trusted
    result["models.untrusted"] = config.models.untrusted

    # Protocols
    result["protocols.defer_thresholds"] = ",".join(
        str(t) for t in config.protocols.defer_thresholds
    )
    result["protocols.run_trusted_monitor_defer"] = (
        config.protocols.run_trusted_monitor_defer
    )
    result["protocols.run_untrusted_monitor_defer"] = (
        config.protocols.run_untrusted_monitor_defer
    )
    result["protocols.run_trusted_monitor_edit"] = (
        config.protocols.run_trusted_monitor_edit
    )

    # Prompts (paths as strings)
    result["prompts.attack"] = str(config.prompts.attack) if config.prompts.attack else "default"
    result["prompts.honest"] = str(config.prompts.honest) if config.prompts.honest else "default"
    result["prompts.monitor"] = str(config.prompts.monitor) if config.prompts.monitor else "default"
    result["prompts.edit"] = str(config.prompts.edit) if config.prompts.edit else "default"

    # Cache
    result["cache.enabled"] = config.cache.enabled

    # MLflow (don't log mlflow config to mlflow - that's meta)
    result["mlflow.enabled"] = config.mlflow.enabled

    return result
