# Colab Runs

Upload this whole `colab-folder` directory to Google Drive, preferably directly under `MyDrive`:

```text
MyDrive/
└── colab-folder/
    ├── learning_rules_MLP.py
    └── final-config-runs/
```

Open one notebook from `final-config-runs`, choose a GPU runtime, and run all cells. The first code cell mounts Google Drive when needed, finds `colab-folder`, sets the working directory, and imports the shared utilities.

Generated figures, tables, CSV files, and zip archives are written next to the uploaded folder, for example:

```text
colab-folder/results_sinus/
colab-folder/results_mnist/
colab-folder/results_cifar10/
colab-folder/results_cali_housing/
colab-folder/results_sinus_scaled_input/
```

If the folder is not placed directly under `MyDrive`, the notebooks still try to search for it. If that search is slow or fails, add this before the setup cell:

```python
import os
os.environ["COLAB_PROJECT_ROOT"] = "/content/drive/MyDrive/path/to/colab-folder"
```
