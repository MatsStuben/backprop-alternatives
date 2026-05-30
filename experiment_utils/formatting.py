from __future__ import annotations

import numpy as np
import pandas as pd

from .runtime import SUMMARY_SIGNIFICANT_FIGURES


def format_scientific_sigfigs(value: float, significant_figures: int = SUMMARY_SIGNIFICANT_FIGURES) -> str:
    mantissa, exponent = f"{value:.{significant_figures - 1}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_summary_number(value: float) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if value == 0.0:
        return "0.00"
    abs_value = abs(value)
    if abs_value < 1e-3 or abs_value >= 1e4:
        return format_scientific_sigfigs(value)
    exponent = int(np.floor(np.log10(abs_value)))
    decimals = SUMMARY_SIGNIFICANT_FIGURES - exponent - 1
    rounded_value = round(value, decimals)
    rounded_abs = abs(rounded_value)
    if rounded_abs == 0.0:
        return "0.00"
    if rounded_abs < 1e-3 or rounded_abs >= 1e4:
        return format_scientific_sigfigs(rounded_value)
    rounded_exponent = int(np.floor(np.log10(rounded_abs)))
    rounded_decimals = max(0, SUMMARY_SIGNIFICANT_FIGURES - rounded_exponent - 1)
    return f"{rounded_value:.{rounded_decimals}f}"


def format_pm(mean: float, std: float, is_epoch: bool = False) -> str:
    if pd.isna(mean):
        return ""
    if is_epoch:
        if pd.isna(std):
            return str(int(round(float(mean))))
        return f"{int(round(float(mean)))} ± {int(round(float(std)))}"
    if pd.isna(std):
        return format_summary_number(mean)
    return f"{format_summary_number(mean)} ± {format_summary_number(std)}"


def format_results_table(table_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in table_summary_df.iterrows():
        rows.append(
            {
                "Method": row["method_label"],
                "Best train loss": format_pm(row["best_train_loss_mean"], row["best_train_loss_std"]),
                "Best test loss": format_pm(row["best_test_loss_mean"], row["best_test_loss_std"]),
                "Best train score": format_pm(row["best_train_score_mean"], row["best_train_score_std"]),
                "Best test score": format_pm(row["best_test_score_mean"], row["best_test_score_std"]),
                "Convergence epoch": format_pm(row["convergence_epoch_mean"], row["convergence_epoch_std"], is_epoch=True),
                "Cosine": format_pm(row["cosine_mean"], row["cosine_std"]),
                "Variance": format_pm(row["variance_mean"], row["variance_std"]),
            }
        )
    return pd.DataFrame(rows)
