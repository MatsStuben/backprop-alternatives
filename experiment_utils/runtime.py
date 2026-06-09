from __future__ import annotations

import gc
import os
import random
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


METHOD_LABELS = {
    "bp": "BP",
    "np": "IS-NP",
    "np_fan_in": "Fan-in NP",
    "np_fixed": "Vanilla NP",
    "wp": "WP",
}
METHOD_COLORS = {
    "bp": "#1f77b4",
    "np": "#ff7f0e",
    "np_fan_in": "#9467bd",
    "np_fixed": "#d62728",
    "wp": "#2ca02c",
}
METHOD_ORDER = ["bp", "np", "np_fan_in", "np_fixed", "wp"]
PERTURBATION_METHODS = ["np", "np_fan_in", "np_fixed", "wp"]
COSINE_COLUMN = "mean_estimate_cosine"
VARIANCE_COLUMN = "sample_variance"
CONVERGENCE_FRACTION = 0.90
SUMMARY_SIGNIFICANT_FIGURES = 3


def is_project_root(path: Path) -> bool:
    return (path / "learning_rules_MLP.py").is_file()


def project_root_candidates(start: Path | None = None):
    starts = [Path.cwd() if start is None else Path(start)]
    for env_name in ["PROJECT_ROOT", "COLAB_PROJECT_ROOT"]:
        value = os.environ.get(env_name)
        if value:
            starts.append(Path(value))
    starts.extend([
        Path("/content/backprop-alternatives"),
        Path("/content/drive/MyDrive/backprop-alternatives"),
        Path("/content/drive/MyDrive/colab-folder"),
    ])

    seen = set()
    for item in starts:
        try:
            resolved = item.expanduser().resolve()
        except Exception:
            continue
        for candidate in [resolved, *resolved.parents]:
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def find_project_root(start: Path | None = None, search_drive: bool = False) -> Path | None:
    for candidate in project_root_candidates(start):
        if is_project_root(candidate):
            return candidate

    if search_drive:
        for root in [Path("/content"), Path("/content/drive/MyDrive")]:
            if not root.exists():
                continue
            for hit in root.rglob("learning_rules_MLP.py"):
                candidate = hit.parent
                if is_project_root(candidate):
                    return candidate
    return None


def mount_drive_if_available() -> None:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except Exception:
        pass


def setup_project_paths(start: Path | None = None, search_drive: bool = True) -> tuple[Path, Path, Path]:
    project_root = find_project_root(start=start, search_drive=False)
    if project_root is None and search_drive:
        mount_drive_if_available()
        project_root = find_project_root(start=start, search_drive=True)
    if project_root is None:
        raise FileNotFoundError("Could not find project root containing learning_rules_MLP.py.")

    notebook_dir = project_root / "notebooks"
    data_dir = project_root / "data"
    for path in [project_root, notebook_dir]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return project_root, notebook_dir, data_dir


def setup_matplotlib() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "thesis_experiment_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ordered_methods(methods: list[str]) -> list[str]:
    return [method for method in METHOD_ORDER if method in methods]
