from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .runtime import METHOD_COLORS, METHOD_LABELS, PERTURBATION_METHODS, ordered_methods


def display_figures(figures: dict[str, plt.Figure] | list[tuple[str, plt.Figure]]) -> None:
    try:
        from IPython.display import display
    except Exception:
        display = None
    items = figures.items() if isinstance(figures, dict) else figures
    for _, fig in items:
        if display is not None:
            display(fig)
        else:
            fig.show()


def add_curve_legend(ax, methods: list[str]) -> None:
    handles = [Line2D([0], [0], color=METHOD_COLORS[method], lw=1.5, label=METHOD_LABELS[method]) for method in methods]
    handles.extend(
        [
            Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="Training"),
            Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="Test"),
        ]
    )
    ax.legend(handles=handles, ncol=2, frameon=True, framealpha=0.95)


def cosine_axis_limits(values: np.ndarray, padding_fraction: float = 0.15) -> tuple[float, float]:
    finite_values = np.asarray([float(value) for value in values if pd.notna(value) and np.isfinite(float(value))])
    if finite_values.size == 0:
        return 0.0, 1.0
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    if min_value >= 0.0:
        upper = max_value * (1.0 + padding_fraction) if max_value > 0.0 else 0.05
        return 0.0, min(1.0, upper)
    value_range = max_value - min_value
    padding = value_range * padding_fraction if value_range > 0.0 else 0.05
    return max(-1.0, min_value - padding), min(1.0, max_value + padding)


def plot_loss_curves(config: dict, history_df: pd.DataFrame) -> plt.Figure:
    grouped = history_df.groupby(["method", "epoch"], as_index=False).agg(train_loss=("train_loss", "mean"), test_loss=("test_loss", "mean"))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    methods = ordered_methods(config["methods"])
    for method in methods:
        method_df = grouped[grouped["method"] == method].sort_values("epoch")
        ax.plot(method_df["epoch"], method_df["train_loss"], color=METHOD_COLORS[method], lw=1.15, linestyle="-")
        ax.plot(method_df["epoch"], method_df["test_loss"], color=METHOD_COLORS[method], lw=1.15, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    add_curve_legend(ax, methods)
    fig.tight_layout()
    return fig


def plot_score_curves(config: dict, history_df: pd.DataFrame) -> plt.Figure | None:
    if config["task_type"] != "classification":
        return None
    grouped = history_df.groupby(["method", "epoch"], as_index=False).agg(train_score=("train_score", "mean"), test_score=("test_score", "mean"))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    methods = ordered_methods(config["methods"])
    for method in methods:
        method_df = grouped[grouped["method"] == method].sort_values("epoch")
        ax.plot(method_df["epoch"], method_df["train_score"], color=METHOD_COLORS[method], lw=1.15, linestyle="-")
        ax.plot(method_df["epoch"], method_df["test_score"], color=METHOD_COLORS[method], lw=1.15, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    add_curve_legend(ax, methods)
    fig.tight_layout()
    return fig


def plot_diagnostic_bar(summary_df: pd.DataFrame, value_mean_col: str, ylabel: str, log_scale: bool = False) -> plt.Figure:
    plot_df = summary_df.set_index("method").loc[[method for method in PERTURBATION_METHODS if method in set(summary_df["method"])]].reset_index()
    values = plot_df[value_mean_col].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(plot_df))
    for index, row in plot_df.iterrows():
        method = row["method"]
        ax.bar(x[index], row[value_mean_col], color=METHOD_COLORS[method], width=0.72, label=METHOD_LABELS[method])
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[method] for method in plot_df["method"]], rotation=20, ha="right")
    ax.set_ylabel(f"{ylabel} (log scale)" if log_scale else ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.95)
    if log_scale:
        positive = values[values > 0]
        ax.set_yscale("log")
        ax.set_ylim(float(positive.min()) / 3.0, float(positive.max()) * 3.0)
        ax.grid(True, which="both", axis="y", alpha=0.25)
    else:
        ax.set_ylim(*cosine_axis_limits(values))
    fig.tight_layout()
    return fig


def plot_final_result_figures(config: dict, history_df: pd.DataFrame, diagnostic_summary_df: pd.DataFrame, checkpoint_summary_df: pd.DataFrame) -> dict[str, plt.Figure]:
    figures = {"loss": plot_loss_curves(config, history_df)}
    score_fig = plot_score_curves(config, history_df)
    if score_fig is not None:
        figures["accuracy"] = score_fig
    figures["cosine"] = plot_diagnostic_bar(diagnostic_summary_df, "cosine_mean", "Cosine similarity")
    figures["variance"] = plot_diagnostic_bar(diagnostic_summary_df, "variance_mean", "Estimator variance", log_scale=True)
    for checkpoint_epoch in sorted(checkpoint_summary_df["checkpoint_epoch"].unique()):
        checkpoint_df = checkpoint_summary_df[checkpoint_summary_df["checkpoint_epoch"] == checkpoint_epoch]
        figures[f"cosine_checkpoint_{int(checkpoint_epoch):03d}"] = plot_diagnostic_bar(checkpoint_df, "cosine_mean", "Cosine similarity")
        figures[f"variance_checkpoint_{int(checkpoint_epoch):03d}"] = plot_diagnostic_bar(checkpoint_df, "variance_mean", "Estimator variance", log_scale=True)
    return figures


def plot_sigma_search_bars(summary_df: pd.DataFrame) -> dict[str, plt.Figure]:
    return {
        "sigma_search_cosine": plot_sigma_metric(summary_df, "cosine_mean", "Cosine similarity", log_scale=False),
        "sigma_search_variance": plot_sigma_metric(summary_df, "variance_mean", "Estimator variance", log_scale=True),
    }


def plot_sigma_metric(summary_df: pd.DataFrame, value_col: str, ylabel: str, log_scale: bool) -> plt.Figure:
    methods = [method for method in PERTURBATION_METHODS if method in set(summary_df["method"])]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.0 * len(methods), 3.4), squeeze=False)
    for axis, method in zip(axes.ravel(), methods):
        method_df = summary_df[summary_df["method"] == method].sort_values("sigma")
        axis.bar(
            method_df["sigma"].astype(str),
            method_df[value_col],
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.5,
        )
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlabel(r"$\sigma$")
        axis.tick_params(axis="x", labelrotation=35)
        if log_scale:
            positive = method_df[value_col].to_numpy(dtype=float)
            positive = positive[positive > 0]
            axis.set_yscale("log")
            if positive.size:
                axis.set_ylim(float(positive.min()) / 3.0, float(positive.max()) * 3.0)
        else:
            axis.set_ylim(*cosine_axis_limits(method_df[value_col].to_numpy(dtype=float)))
        axis.set_ylabel(f"{ylabel} (log scale)" if log_scale else ylabel)
    fig.tight_layout()
    return fig


def plot_scaled_input_comparison(summary_df: pd.DataFrame, metric: str, ylabel: str, checkpoint_epoch=None) -> plt.Figure:
    if checkpoint_epoch is not None:
        plot_df = summary_df[summary_df["checkpoint_epoch"] == checkpoint_epoch].copy()
    else:
        plot_df = summary_df.copy()
    base_df = plot_df[plot_df["condition"] == "base"].set_index("method")
    scaled_df = plot_df[plot_df["condition"] == "scaled"].set_index("method")
    methods = [method for method in PERTURBATION_METHODS if method in base_df.index and method in scaled_df.index]
    base_values = np.asarray([float(base_df.loc[method, metric]) for method in methods])
    scaled_values = np.asarray([float(scaled_df.loc[method, metric]) for method in methods])
    all_values = np.concatenate([base_values, scaled_values])
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    x = np.arange(len(methods))
    width = 0.34
    for index, method in enumerate(methods):
        color = METHOD_COLORS[method]
        ax.bar(x[index] - width / 2, base_values[index], width=width, color=color, edgecolor="black", linewidth=0.5)
        ax.bar(x[index] + width / 2, scaled_values[index], width=width, color=color, edgecolor="black", linewidth=0.5, hatch="////", alpha=0.78)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[method] for method in methods], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="0.6", edgecolor="black", label="[-1, 1]"),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.6", edgecolor="black", hatch="////", alpha=0.78, label="[-5, 5]"),
    ]
    ax.legend(handles=legend_handles, frameon=True, framealpha=0.95)
    if metric in {"update_variance", "variance", "sample_variance"}:
        positive_values = all_values[all_values > 0.0]
        ax.set_yscale("log")
        ax.set_ylabel(f"{ylabel} (log scale)")
        ax.set_ylim(float(positive_values.min()) / 3.0, float(positive_values.max()) * 3.0)
        ax.grid(True, which="both", axis="y", alpha=0.25)
    else:
        ax.set_ylim(*cosine_axis_limits(all_values))
    fig.tight_layout()
    return fig
