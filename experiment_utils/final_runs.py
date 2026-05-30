from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from .data import load_task_data
from .diagnostics import analyze_frozen_backprop_estimators, train_backprop_checkpoint_states
from .evaluation import aggregate_diagnostics, aggregate_history, aggregate_results_table
from .formatting import format_results_table
from .plotting import display_figures, plot_final_result_figures, plot_scaled_input_comparison
from .runtime import METHOD_LABELS, PERTURBATION_METHODS, clear_memory, get_device, ordered_methods, setup_matplotlib
from .training import history_rows, summary_row, train_one_run


def _display_table(df: pd.DataFrame, title: str | None = None) -> None:
    try:
        from IPython.display import display
    except Exception:
        display = None
    if title:
        print(title)
    if display is not None:
        display(df)
    else:
        print(df.to_string(index=False))


def run_training_and_diagnostics(config: dict, device=None) -> dict:
    """Run all methods/seeds for one final task configuration."""
    device = get_device() if device is None else device
    data = load_task_data(config, device)
    score_key = "acc" if config["task_type"] == "classification" else "r2"
    methods = ordered_methods(config["methods"])
    perturbation_methods = [method for method in methods if method in PERTURBATION_METHODS]

    history_rows_out: list[dict] = []
    summary_rows_out: list[dict] = []
    diagnostic_frames: list[pd.DataFrame] = []

    print(
        f"Dataset: train={data['train_size']}, train_eval={data.get('train_eval_size', data['train_size'])}, "
        f"test={data['test_size']} | device={device}"
    )

    for seed_index, seed in enumerate(config["seeds"], start=1):
        print(f"\n===== Seed {seed_index}/{len(config['seeds'])}: {seed} =====")
        for method in methods:
            run_config = config["run_configs"][method]
            print(f"\n--- Training {METHOD_LABELS[method]} ---")
            result = train_one_run(method, run_config, data, config, seed, device)
            history = result["history"]
            history_rows_out.extend(history_rows(config["task_key"], method, seed, run_config, history, score_key))
            summary_rows_out.append(summary_row(config["task_key"], method, seed, run_config, history, result, score_key))
            clear_memory()

        if perturbation_methods:
            print("\n--- Frozen-backprop diagnostics ---")
            checkpoint_states = train_backprop_checkpoint_states(data, config, seed, device)
            diagnostics_df = analyze_frozen_backprop_estimators(checkpoint_states, data, config, device)
            diagnostics_df["task"] = config["task_key"]
            diagnostics_df["seed"] = seed
            diagnostic_frames.append(diagnostics_df)
            clear_memory()

    history_df = pd.DataFrame(history_rows_out)
    summary_seed_df = pd.DataFrame(summary_rows_out)
    diagnostics_df = pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame()
    return {
        "config": config,
        "data": data,
        "history_df": history_df,
        "summary_seed_df": summary_seed_df,
        "diagnostics_df": diagnostics_df,
    }


def summarize_final_outputs(run_outputs: dict) -> dict:
    """Aggregate raw run outputs into tables and plotting data."""
    history_df = run_outputs["history_df"]
    diagnostics_df = run_outputs["diagnostics_df"]
    history_summary_df = aggregate_history(history_df)

    if diagnostics_df.empty:
        checkpoint_seed_df = pd.DataFrame()
        checkpoint_summary_df = pd.DataFrame()
        seed_over_checkpoints_df = pd.DataFrame()
        diagnostic_summary_df = pd.DataFrame()
    else:
        (
            checkpoint_seed_df,
            checkpoint_summary_df,
            seed_over_checkpoints_df,
            diagnostic_summary_df,
        ) = aggregate_diagnostics(diagnostics_df)

    table_seed_df, table_summary_df = aggregate_results_table(run_outputs["summary_seed_df"], seed_over_checkpoints_df)
    formatted_table_df = format_results_table(table_summary_df)
    return {
        "history_summary_df": history_summary_df,
        "checkpoint_seed_df": checkpoint_seed_df,
        "checkpoint_summary_df": checkpoint_summary_df,
        "seed_over_checkpoints_df": seed_over_checkpoints_df,
        "diagnostic_summary_df": diagnostic_summary_df,
        "table_seed_df": table_seed_df,
        "table_summary_df": table_summary_df,
        "formatted_table_df": formatted_table_df,
    }


def make_final_figures(config: dict, history_df: pd.DataFrame, diagnostic_summary_df: pd.DataFrame, checkpoint_summary_df: pd.DataFrame) -> dict:
    return plot_final_result_figures(config, history_df, diagnostic_summary_df, checkpoint_summary_df)


def run_final_config(config: dict, project_root: Path | None = None, show: bool = True, device=None) -> dict:
    """Run a complete final multi-seed experiment and display notebook outputs."""
    del project_root
    setup_matplotlib()
    device = get_device() if device is None else device
    raw_outputs = run_training_and_diagnostics(config, device=device)
    summary_outputs = summarize_final_outputs(raw_outputs)
    figures = make_final_figures(
        config,
        raw_outputs["history_df"],
        summary_outputs["diagnostic_summary_df"],
        summary_outputs["checkpoint_summary_df"],
    )
    outputs = {**raw_outputs, **summary_outputs, "figures": figures}

    if show:
        display_figures(figures)
        _display_table(summary_outputs["formatted_table_df"], title="\nSummary table")
    return outputs


def _condition_config(base_config: dict, condition: dict) -> dict:
    config = deepcopy(base_config)
    data_kwargs = dict(config["data_kwargs"])
    data_kwargs.update(condition.get("data_kwargs", {}))
    if "input_scale" in condition:
        data_kwargs["input_scale"] = condition["input_scale"]
    config["data_kwargs"] = data_kwargs
    return config


def aggregate_scaled_input_diagnostics(diagnostics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_params = diagnostics_df[diagnostics_df["component"] == "all"].copy()
    checkpoint_seed = (
        all_params.groupby(["condition", "condition_label", "input_scale", "seed", "checkpoint_epoch", "method"], as_index=False)
        .agg(cosine=("mean_estimate_cosine", "mean"), variance=("sample_variance", "mean"), sigma=("sigma", "mean"))
        .sort_values(["condition", "seed", "checkpoint_epoch", "method"])
    )
    checkpoint_summary = (
        checkpoint_seed.groupby(["condition", "condition_label", "input_scale", "checkpoint_epoch", "method"], as_index=False)
        .agg(
            cosine=("cosine", "mean"),
            cosine_std=("cosine", "std"),
            variance=("variance", "mean"),
            variance_std=("variance", "std"),
            sigma=("sigma", "mean"),
        )
        .sort_values(["condition", "checkpoint_epoch", "method"])
    )
    seed_summary = (
        checkpoint_seed.groupby(["condition", "condition_label", "input_scale", "seed", "method"], as_index=False)
        .agg(cosine=("cosine", "mean"), variance=("variance", "mean"), sigma=("sigma", "mean"))
        .sort_values(["condition", "seed", "method"])
    )
    overall_summary = (
        seed_summary.groupby(["condition", "condition_label", "input_scale", "method"], as_index=False)
        .agg(
            cosine=("cosine", "mean"),
            cosine_std=("cosine", "std"),
            variance=("variance", "mean"),
            variance_std=("variance", "std"),
            sigma=("sigma", "mean"),
        )
        .sort_values(["condition", "method"])
    )
    return checkpoint_summary, seed_summary, overall_summary


def run_scaled_input_diagnostics(config: dict, project_root: Path | None = None, show: bool = True, device=None) -> dict:
    """Run the sinus scaled-input diagnostic experiment."""
    del project_root
    setup_matplotlib()
    device = get_device() if device is None else device
    diagnostic_frames: list[pd.DataFrame] = []

    for seed_index, seed in enumerate(config["seeds"], start=1):
        print(f"\n===== Seed {seed_index}/{len(config['seeds'])}: {seed} =====")
        for condition in config["conditions"]:
            condition_name = condition["condition"]
            print(f"\n=== Condition {condition['condition_label']} ===")
            condition_config = _condition_config(config, condition)
            data = load_task_data(condition_config, device)
            checkpoint_states = train_backprop_checkpoint_states(data, condition_config, seed, device)
            diagnostics_df = analyze_frozen_backprop_estimators(checkpoint_states, data, condition_config, device)
            diagnostics_df["task"] = condition_config["task_key"]
            diagnostics_df["seed"] = seed
            diagnostics_df["condition"] = condition_name
            diagnostics_df["condition_label"] = condition["condition_label"]
            diagnostics_df["input_scale"] = float(condition_config["data_kwargs"].get("input_scale", float("nan")))
            diagnostic_frames.append(diagnostics_df)
            clear_memory()

    diagnostics_df = pd.concat(diagnostic_frames, ignore_index=True)
    checkpoint_summary_df, seed_summary_df, overall_summary_df = aggregate_scaled_input_diagnostics(diagnostics_df)
    figures = {
        "scaled_input_cosine": plot_scaled_input_comparison(overall_summary_df, "cosine", "Cosine similarity"),
        "scaled_input_variance": plot_scaled_input_comparison(overall_summary_df, "variance", "Estimator variance"),
    }
    for checkpoint_epoch in sorted(checkpoint_summary_df["checkpoint_epoch"].unique()):
        checkpoint_df = checkpoint_summary_df[checkpoint_summary_df["checkpoint_epoch"] == checkpoint_epoch]
        figures[f"scaled_input_cosine_checkpoint_{int(checkpoint_epoch):03d}"] = plot_scaled_input_comparison(
            checkpoint_df, "cosine", "Cosine similarity", checkpoint_epoch=checkpoint_epoch
        )
        figures[f"scaled_input_variance_checkpoint_{int(checkpoint_epoch):03d}"] = plot_scaled_input_comparison(
            checkpoint_df, "variance", "Estimator variance", checkpoint_epoch=checkpoint_epoch
        )

    outputs = {
        "config": config,
        "diagnostics_df": diagnostics_df,
        "checkpoint_summary_df": checkpoint_summary_df,
        "seed_summary_df": seed_summary_df,
        "overall_summary_df": overall_summary_df,
        "figures": figures,
    }
    if show:
        display_figures(figures)
        _display_table(overall_summary_df, title="\nScaled-input summary")
    return outputs
