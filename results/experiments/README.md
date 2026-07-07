# Experiment Run History

This directory contains timestamped logs of historical model training runs. Each subdirectory corresponds to a specific training execution and contains a `run_summary.json` file.

## Schema of `run_summary.json`

Each summary file logs:
* **Run Metadata**: Unique `run_id` and UTC execution timestamp.
* **Model Parameters**: Hyperparameters used for the frequency (e.g., Poisson, Random Forest, XGBoost) and severity (e.g., Gamma, Random Forest, XGBoost) models.
* **Evaluation Metrics**: Performance stats (RMSE, deviance, observed/predicted means) computed on the holdout test split.
* **Dataset Versioning**: Path, size, modification timestamp, and SHA-256 checksums of the training datasets (`freMTPL2freq.csv` and `freMTPL2sev.csv`) to ensure full reproducibility.
* **Dataset Quality Summary**: Row counts, merged statistics, and filtering counts.
* **Pricing Configuration**: Business loadings, thresholds, and fixed expense loads used for dynamic premiums.

## Run Archive

* **`20260420T164457Z`** — Initial baseline testing with default GLM Poisson and Gamma regressors.
* **`20260420T165705Z`** — Feature engineering testing, validating the `LogDensity` derived feature.
* **`20260421T183151Z`** — Parameter tuning for tree-based ensemble options (Random Forest).
* **`20260422T142604Z`** — Production baseline run utilizing Poisson frequency model and Gamma severity model.
