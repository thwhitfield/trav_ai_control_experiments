"""Plotting utilities for AI Control paper replication."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_safety_vs_usefulness_by_protocol(
    results_df: pd.DataFrame,
    title: str = "AI Control: Safety vs Usefulness",
) -> plt.Figure:
    """Plot safety vs usefulness for different protocols with error bars.

    Args:
        results_df: DataFrame with columns protocol_name, safety_mean, safety_error,
                   usefulness_mean, usefulness_error, audit_budget
        title: Plot title

    Returns:
        The matplotlib figure object
    """
    fig = plt.figure(figsize=(12, 8))

    # Colors for different protocols
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]

    for i, (protocol_name, protocol_df) in enumerate(
        results_df.groupby("protocol_name")
    ):
        # Sort by audit_budget for consistent line drawing
        if "defer_threshold" in protocol_df.columns:
            protocol_df = protocol_df.sort_values("defer_threshold")

        color = colors[i % len(colors)]

        # Plot with error bars
        assert isinstance(protocol_name, str)
        plt.errorbar(
            protocol_df["safety"],
            protocol_df["usefulness"],
            label=protocol_name.replace("_", " ").title(),
            marker="o",
            markersize=6,
            color=color,
            alpha=0.8,
            capsize=3,
            capthick=1,
        )

        # Add defer percentage labels to points (fallback to threshold if not available)
        for _, row in protocol_df.iterrows():
            # Skip if no defer info
            if row.get("defer_threshold") is None or pd.isna(
                row.get("defer_threshold")
            ):
                continue

            # Prefer defer_percentage, fallback to threshold
            if row.get("defer_percentage") is not None and not pd.isna(
                row.get("defer_percentage")
            ):
                label = f"{row['defer_percentage']:.0f}%"
            else:
                label = f"t={row['defer_threshold']:.1f}"

            plt.annotate(
                label,
                (row["safety"], row["usefulness"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
                color=color,
            )

    plt.xlabel("Safety")
    plt.ylabel("Usefulness (scaled)")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_monitoring_score_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plot monitoring score distributions for honest vs attack samples."""
    monitor_configs = [
        ("trusted_monitor_defer", "Trusted Monitoring"),
        ("untrusted_monitor_defer", "Untrusted Monitoring"),
    ]

    fig, axes = plt.subplots(
        1, len(monitor_configs), figsize=(6 * len(monitor_configs), 5)
    )

    for ax, (protocol_name, title) in zip(axes, monitor_configs):
        # Subset to protocol and minimum threshold
        proto_df = df[df["protocol_name"] == protocol_name]
        min_threshold = proto_df["defer_threshold"].min()
        proto_df = proto_df[proto_df["defer_threshold"] == min_threshold]

        # Get the honest and attack score pandas series
        honest = proto_df.loc[
            proto_df["eval_mode"] == "honest", "score_max_suspicion_scorer"
        ].dropna()
        attack = proto_df.loc[
            proto_df["eval_mode"] == "attack", "score_max_suspicion_scorer"
        ].dropna()

        # Plot them in comparison
        ax.hist(
            [honest, attack], bins=20, density=True, label=["Innocent", "Backdoored"]
        )
        ax.legend()
        ax.set_xlabel("Monitoring score")
        ax.set_ylabel("Relative Frequency")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y}%"))

    plt.tight_layout()
    return fig
