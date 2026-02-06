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
class ExperimentConfig:
    """Root configuration object combining all settings."""

    experiment: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    protocols: ProtocolSettings = field(default_factory=ProtocolSettings)
    prompts: PromptPaths = field(default_factory=PromptPaths)
    cache: CacheConfig = field(default_factory=CacheConfig)


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
