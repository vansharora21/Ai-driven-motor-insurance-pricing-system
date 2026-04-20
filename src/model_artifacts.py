from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "models"
LEGACY_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "premium_reports"
EVALUATION_DIR = RESULTS_DIR / "evaluation"

FREQUENCY_MODEL_PATH = ARTIFACTS_DIR / "frequency_model.joblib"
SEVERITY_MODEL_PATH = ARTIFACTS_DIR / "severity_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
METRICS_PATH = EVALUATION_DIR / "metrics.json"


def _resolve_existing_artifact_path(primary: Path, legacy_dir: Path) -> Path:
    """Resolve artifact path with fallback to legacy artifacts directory."""
    if primary.exists():
        return primary

    legacy_path = legacy_dir / primary.name
    if legacy_path.exists():
        return legacy_path
    return primary


def ensure_output_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def save_artifacts(
    frequency_model: Any,
    severity_model: Any,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Persist trained models plus the metadata needed for inference and UI defaults."""
    ensure_output_dirs()
    joblib.dump(frequency_model, FREQUENCY_MODEL_PATH)
    joblib.dump(severity_model, SEVERITY_MODEL_PATH)

    with METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


def load_artifacts() -> dict[str, Any]:
    """Load trained models, metadata, and saved evaluation metrics."""
    frequency_model_path = _resolve_existing_artifact_path(FREQUENCY_MODEL_PATH, LEGACY_ARTIFACTS_DIR)
    severity_model_path = _resolve_existing_artifact_path(SEVERITY_MODEL_PATH, LEGACY_ARTIFACTS_DIR)
    metadata_path = _resolve_existing_artifact_path(METADATA_PATH, LEGACY_ARTIFACTS_DIR)

    if not frequency_model_path.exists() or not severity_model_path.exists():
        raise FileNotFoundError(
            "Trained model artifacts are missing. Run `python train.py` before starting inference."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Model metadata is missing. Run `python train.py` to regenerate artifacts."
        )

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    metrics = {}
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)

    return {
        "frequency_model": joblib.load(frequency_model_path),
        "severity_model": joblib.load(severity_model_path),
        "metadata": metadata,
        "metrics": metrics,
    }
