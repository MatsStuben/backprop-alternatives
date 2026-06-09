# Backpropagation Alternatives

This repository contains the code, experiment notebooks, results, and thesis source for a master's thesis on biologically plausible alternatives to backpropagation, with a focus on perturbation-based learning.

The thesis studies weight perturbation (WP), node perturbation (NP), fan-in-scaled NP, and a theoretically motivated variant called input-scaled node perturbation (IS-NP). The main contribution is to relate weight and node perturbation through the pre-activation noise induced by weight perturbations, and to use this relationship to derive and evaluate IS-NP.

## Repository Structure

```text
learning_rules_MLP.py        Core implementation of MLPs and perturbation learning rules
experiment_utils/            Shared utilities for data loading, training, diagnostics, plotting, and exports
notebooks/                   Colab/local notebooks for final runs and hyperparameter tuning
Masteroppgave/               LaTeX thesis source, figures, tables, and preview PDF
results/                     Stored experiment outputs used to generate thesis figures and tables
testing_files/               Older exploratory and validation scripts
papers/                      Reference papers used during the thesis work
```

## Main Code

`learning_rules_MLP.py` is the trusted implementation of the learning rules. The shared modules in `experiment_utils/` wrap this implementation to run full experiments, diagnostic analyses, plots, tables, and exports without duplicating learning-rule logic.

The most important utility modules are:

- `experiment_utils/data.py`: dataset loading and preprocessing
- `experiment_utils/training.py`: model construction and training loops
- `experiment_utils/diagnostics.py`: cosine similarity and estimator variance diagnostics
- `experiment_utils/final_runs.py`: orchestration for final multi-seed experiments
- `experiment_utils/sigma_search.py`: frozen-backprop perturbation-scale diagnostics
- `experiment_utils/grid_search.py`: local hyperparameter grid searches
- `experiment_utils/plotting.py`: figure generation

## Notebooks

The `notebooks/` folder contains the runnable experiment notebooks:

- `sinus-final-run.ipynb`
- `california-housing-final-run.ipynb`
- `mnist-final-run.ipynb`
- `cifar10-final-run.ipynb`
- `sinus-hyperparam-tuning.ipynb`
- `california-housing-hyperparam-tuning.ipynb`
- `mnist-hyperparam-tuning.ipynb`
- `cifar10-hyperparam-tuning.ipynb`
- `sinus-scaled-input-diagnostics.ipynb`

The final-run notebooks generate training/test performance plots, cosine similarity plots, estimator variance plots, checkpoint-specific diagnostics, and summary tables. The hyperparameter notebooks contain the frozen-backprop sigma search, local grid search, and editable full-length run.

The notebooks are intended to work both locally and in Google Colab. In Colab, upload or mount a folder containing at least:

```text
learning_rules_MLP.py
experiment_utils/
notebooks/
```

The notebook setup cells locate the project root and add it to `sys.path`.

## Local Setup

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

or, if using the project metadata:

```bash
pip install -e .
```

The main dependencies are PyTorch, torchvision, NumPy, matplotlib, pandas, and scikit-learn.

## Thesis Preview

The thesis source is in `Masteroppgave/`. A local preview can be built with:

```bash
cd Masteroppgave
tectonic main_preview.tex
```

The generated preview is written to:

```text
Masteroppgave/main_preview.pdf
```

The full thesis build uses `main.tex` and the bibliography in `Masteroppgave/references.bib`.

## Results

The `results/` folder contains stored experiment outputs. The thesis-facing figures and tables are placed under:

```text
Masteroppgave/ResultAssets/
```

These assets are included by the LaTeX source so that the `Masteroppgave/` folder can be uploaded to Overleaf or compiled locally.

## Notes

This repository is organized around the final thesis workflow. Older cluster-specific instructions are no longer part of the main README because the current experiment workflow is based on local/Colab notebooks and shared utility modules.
