from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name)


def export_outputs(outputs: dict, archive_name: str, destination: str | Path = "notebook_exports") -> Path:
    """Manually export DataFrames and figures from a notebook run to a zip archive."""
    destination = Path(destination)
    export_dir = destination / _safe_name(archive_name)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    for key, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(export_dir / f"{_safe_name(key)}.csv", index=False)

    figures = outputs.get("figures", {})
    if isinstance(figures, dict):
        figure_dir = export_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(figure_dir / f"{_safe_name(name)}.pdf", bbox_inches="tight")
            figure.savefig(figure_dir / f"{_safe_name(name)}.png", bbox_inches="tight", dpi=300)

    archive_path = shutil.make_archive(str(export_dir), "zip", root_dir=export_dir)
    return Path(archive_path)


def download_if_colab(path: str | Path) -> None:
    """Download an exported archive when running in Google Colab."""
    try:
        from google.colab import files

        files.download(str(path))
    except Exception:
        print(f"Archive created at: {Path(path).resolve()}")
