from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
DEFAULT_FREQUENCY_PATH = PROJECT_ROOT / "data" / "freMTPL2freq.csv"
DEFAULT_SEVERITY_PATH = PROJECT_ROOT / "data" / "freMTPL2sev.csv"


def _file_info(path: Path) -> dict[str, Any]:
    """Return basic version metadata for a dataset file."""
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return info

    stat = path.stat()
    info.update(
        {
            "size_bytes": int(stat.st_size),
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "sha256": _sha256(path),
        }
    )
    return info


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as data_file:
        while True:
            chunk = data_file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_version_info(
    frequency_path: str | Path = DEFAULT_FREQUENCY_PATH,
    severity_path: str | Path = DEFAULT_SEVERITY_PATH,
) -> dict[str, Any]:
    """Build reproducibility metadata for input datasets."""
    freq_path = Path(frequency_path)
    sev_path = Path(severity_path)
    return {
        "frequency_dataset": _file_info(freq_path),
        "severity_dataset": _file_info(sev_path),
    }


def persist_experiment_run(
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    model_config: dict[str, Any],
    data_quality: dict[str, Any],
) -> Path:
    """Write a structured experiment run record under results/experiments."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    run_summary = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_parameters": model_config,
        "evaluation_metrics": metrics,
        "dataset_version": build_dataset_version_info(),
        "dataset_summary": metadata.get("dataset_summary", {}),
        "data_quality": data_quality,
        "pricing_config": metadata.get("pricing_config", {}),
    }

    summary_path = run_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(run_summary, summary_file, indent=2)

    return summary_path
