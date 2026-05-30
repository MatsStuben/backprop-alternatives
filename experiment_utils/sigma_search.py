from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import load_task_data
from .diagnostics import analyze_frozen_backprop_estimators, summarize_sigma_diagnostics, train_backprop_checkpoint_states
from .plotting import display_figures, plot_sigma_search_bars
from .runtime import METHOD_LABELS, clear_memory, get_device, setup_matplotlib


def run_sigma_search(config: dict, project_root: Path | None = None, show: bool = True, device=None) -> dict:
    """Run frozen-backprop sigma diagnostics over configured sigma grids."""
    del project_root
    setup_matplotlib()
    device = get_device() if device is None else device
    data = load_task_data(config, device)
    analysis_key = "sigma_search"
    batch_frames: list[pd.DataFrame] = []

    print(
        f"Dataset: train={data['train_size']}, train_eval={data.get('train_eval_size', data['train_size'])}, "
        f"test={data['test_size']} | device={device}"
    )
    for seed_index, seed in enumerate(config[analysis_key].get("seeds", [0]), start=1):
        print(f"\n===== Sigma-search seed {seed_index}/{len(config[analysis_key].get('seeds', [0]))}: {seed} =====")
        checkpoint_states = train_backprop_checkpoint_states(data, config, seed, device, analysis_key=analysis_key)
        for method, sigma_values in config[analysis_key]["sigma_grids"].items():
            for sigma in sigma_values:
                print(f"\n--- {METHOD_LABELS[method]} | sigma={sigma:.6g} ---")
                diagnostics_df = analyze_frozen_backprop_estimators(
                    checkpoint_states,
                    data,
                    config,
                    device,
                    analysis_key=analysis_key,
                    method_sigmas={method: float(sigma)},
                )
                diagnostics_df["task"] = config["task_key"]
                diagnostics_df["seed"] = seed
                batch_frames.append(diagnostics_df)
                clear_memory()

    batch_df = pd.concat(batch_frames, ignore_index=True)
    checkpoint_summary_df, overall_summary_df = summarize_sigma_diagnostics(batch_df)
    figures = plot_sigma_search_bars(overall_summary_df)
    outputs = {
        "config": config,
        "data": data,
        "batch_df": batch_df,
        "checkpoint_summary_df": checkpoint_summary_df,
        "overall_summary_df": overall_summary_df,
        "figures": figures,
    }
    if show:
        display_figures(figures)
        try:
            from IPython.display import display

            display(overall_summary_df)
        except Exception:
            print(overall_summary_df.to_string(index=False))
    return outputs
