"""Upload trained model artifacts to the Hugging Face Hub.

Usage:
    python scripts/upload_model_to_hf.py --repo yourname/motor-insurance-pricing

The script uploads:
    - frequency_model.joblib   (Poisson GLM pipeline)
    - severity_model.joblib    (Gamma GLM pipeline)
    - model_metadata.json      (defaults, ranges, pricing config)
    - MODEL_CARD.md            (model documentation)

Requires a Hugging Face token. Set it via:
    export HF_TOKEN=hf_xxx            (Linux/macOS)
    $env:HF_TOKEN = "hf_xxx"          (PowerShell)
or pass --token.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, login

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_CARD_PATH = PROJECT_ROOT / "MODEL_CARD.md"

FILES_TO_UPLOAD = [
    "frequency_model.joblib",
    "severity_model.joblib",
    "model_metadata.json",
    "MODEL_CARD.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload motor pricing model artifacts to HF Hub.")
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. yourname/motor-insurance-pricing")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var).")
    parser.add_argument("--private", action="store_true", help="Create a private repo.")
    args = parser.parse_args()

    if args.token:
        login(token=args.token)

    api = HfApi()

    # Create the repo if it does not exist yet.
    try:
        api.create_repo(repo_id=args.repo, private=args.private, exist_ok=True)
        print(f"Repo ready: {args.repo}")
    except Exception as exc:
        print(f"Could not create repo (may already exist): {exc}")

    for filename in FILES_TO_UPLOAD:
        file_path = MODELS_DIR / filename
        if not file_path.exists():
            print(f"SKIP {filename}: not found at {file_path}")
            continue
        print(f"Uploading {filename} ...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=filename,
            repo_id=args.repo,
        )

    print(f"\nDone. Models live at https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()