# Agent Instructions

These instructions are meant to keep Codex focused and avoid wasting context in this repository.

## Scope and Token Use

- Do not inspect `Masteroppgave/` unless the user explicitly asks for thesis edits, thesis wording, compilation, figures/tables inside the thesis, or bibliography/reference work.
- Do not inspect `papers/` unless the user explicitly asks for literature, papers, citations, or source checking.
- When thesis context is needed but exact source files are not necessary, prefer `Masteroppgave/main_preview.pdf` as the first reference point.
- Avoid broad recursive reads of notebooks, PDFs, thesis files, result files, or paper folders. Use targeted `rg`, `find`, and focused file reads.
- Ignore generated output folders unless the user asks for results, figures, tables, or data products.

## Code Quality

- Code quality and structure matter in this project because the code is delivered as part of the master thesis.
- Prefer clear, reusable, well-named functions over notebook-local duplicated code.
- Shared code that will be reused across notebooks or tasks should live in shared utility files, not copied into each notebook.
- Keep notebooks thin: configuration, short orchestration cells, and display/download cells only.
- Avoid hacky one-off changes, stale method filters, hidden state, and old exploratory cells in notebooks intended for delivery.

## Trusted Codebase

- Use `learning_rules_MLP.py` as the trusted implementation of the model and learning rules whenever possible.
- Do not reimplement `MLP`, `backprop_step`, `node_perturbation_step`, `node_perturbation_step_fan_in_scaled`, `node_perturbation_step_fixed_sigma`, or `weight_perturb_step` in notebooks unless explicitly asked.


## Experiment Code Conventions

- Keep the four task notebooks structurally consistent.
- Dataset loading, training loops, diagnostics, aggregation, plotting, and saving should be shared when possible.
- Notebook task cells should make the task-specific values visible: dataset settings, architecture, epochs, seeds, grids, and selected hyperparameters.
- Outputs should be saved in predictable folders and archives so they can be copied into `Masteroppgave/` only when the user asks.

## Thesis Work

- Always rerun the preview after modifying files in `Masteroppgave/`.
- Use this command from `Masteroppgave/`:

```bash
~/.local/bin/tectonic main_preview.tex
```

- Do not update thesis figures/tables unless the user explicitly asks.
- If adding thesis assets, place them under `Masteroppgave/ResultAssets/` so uploading only `Masteroppgave/` to Overleaf remains sufficient.
