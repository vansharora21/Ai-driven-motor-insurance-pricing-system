"""Download model artifacts from the Hugging Face Hub into models/.

Used by the Render backend at startup so the deployed service always
pulls the latest published model version instead of shipping joblib
files inside the container.

Usage:
    python scripts/download_model_from_hf.py --repo yourname/motor-insurance-pricing

Env vars:
    HF_TOKEN (optional) — required only for private repos.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

FILES_TO_DOWNLOAD = [
    "frequency_model.joblib",
    "severity_model.joblib",
    "model_metadata.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download motor pricing model artifacts from HF Hub.")
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. yourname/motor-insurance-pricing")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    for filename in FILES_TO_DOWNLOAD:
        print(f"Downloading {filename} ...")
        local_path = hf_hub_download(
            repo_id=args.repo,
            filename=filename,
            token=token,
        )
        destination = MODELS_DIR / filename
        destination.write_bytes(Path(local_path).read_bytes())
        print(f"  -> {destination}")

    print("\nAll model artifacts downloaded.")


if __name__ == "__main__":
    main()