"""FastAPI inference backend for the motor insurance pricing system.

Wraps the existing `predict_premiums()` pipeline so the Next.js frontend
can score policies over HTTP. This file is deployment-agnostic: it runs
locally with uvicorn, on Render, or inside any ASGI serverless adapter.

Endpoints:
    GET  /health          -> liveness + artifact status
    POST /predict         -> score a single policy
    POST /predict/batch   -> score many policies + portfolio summary
    GET  /model-info      -> model metadata + saved evaluation metrics
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data_loader import DataValidationError
from src.db import save_quote, table_counts
from src.model_artifacts import load_artifacts

logger = logging.getLogger("motor-pricing-api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Motor Insurance Pricing API",
    description="AI-driven motor insurance premium pricing (frequency x severity).",
    version="1.0.0",
)

# The frontend is served from a different origin in production (Vercel).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PolicyInput(BaseModel):
    """Policy attributes accepted by the pricing models."""

    IDpol: int | None = None
    Exposure: float = Field(gt=0, description="Policy exposure in years.")
    VehPower: float = Field(gt=0, description="Vehicle horsepower class (1-20).")
    VehAge: float = Field(ge=0, description="Vehicle age in years.")
    DrivAge: float = Field(ge=18, description="Driver age in years.")
    BonusMalus: float = Field(ge=50, description="Bonus-malus coefficient.")
    VehBrand: str = Field(min_length=1, description="Vehicle brand code (B1-B14).")
    VehGas: str = Field(min_length=1, description="Fuel type (Regular/Diesel).")
    Area: str = Field(min_length=1, description="Area risk segment (A-F).")
    Density: float = Field(ge=0, description="Population density.")
    Region: str = Field(min_length=1, description="French region label.")


class SinglePredictRequest(BaseModel):
    policy: PolicyInput
    consent: bool = Field(
        default=False,
        description="True when the user consented to their anonymized quote being used for research/retraining.",
    )


class BatchPredictRequest(BaseModel):
    policies: list[PolicyInput] = Field(min_length=1)
    consent: bool = Field(
        default=False,
        description="Applied to every policy in the batch.",
    )


# ---------------------------------------------------------------------------
# Artifact loading (lazy singleton)
# ---------------------------------------------------------------------------

_artifacts: dict[str, Any] | None = None


def get_artifacts() -> dict[str, Any]:
    """Load trained models + metadata once and reuse them across requests."""
    global _artifacts
    if _artifacts is None:
        try:
            _artifacts = load_artifacts()
            logger.info("Loaded model artifacts from %s", "models/")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _artifacts


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars into native Python types for JSON."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {column: _to_jsonable(row[column]) for column in row.index}


def _score_policies(policies: list[PolicyInput]) -> pd.DataFrame:
    """Run the shared inference pipeline over a list of policy inputs."""
    from scripts.predict import predict_premiums

    input_df = pd.DataFrame([policy.model_dump() for policy in policies])
    try:
        return predict_premiums(input_df, get_artifacts())
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        artifacts = get_artifacts()
        return {
            "status": "ok",
            "models_loaded": True,
            "frequency_model": artifacts["metadata"].get("modeling_config", {}).get("models", {}).get("frequency_model"),
            "severity_model": artifacts["metadata"].get("modeling_config", {}).get("models", {}).get("severity_model"),
        }
    except HTTPException as exc:
        return {"status": "degraded", "models_loaded": False, "detail": exc.detail}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Health check failed")
        return {"status": "error", "models_loaded": False, "detail": str(exc)}


@app.post("/predict")
def predict_single(request: SinglePredictRequest) -> dict[str, Any]:
    scored = _score_policies([request.policy])
    result = _row_to_dict(scored.iloc[0])
    # Best-effort persistence: never fails the request if Supabase is down.
    save_quote(
        request.policy.model_dump(),
        result,
        source="single",
        consent=request.consent,
    )
    return {"policy": result}


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest) -> dict[str, Any]:
    scored = _score_policies(request.policies)
    results = [_row_to_dict(row) for _, row in scored.iterrows()]

    for policy, result in zip(request.policies, results):
        save_quote(
            policy.model_dump(),
            result,
            source="batch",
            consent=request.consent,
        )

    risk_counts = (
        scored["risk_category"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .to_dict()
    )

    return {
        "policies": results,
        "summary": {
            "total": int(len(scored)),
            "risk_counts": {key: int(risk_counts.get(key, 0)) for key in ["Low", "Medium", "High"]},
            "avg_premium": float(scored["final_premium"].mean()),
            "min_premium": float(scored["final_premium"].min()),
            "max_premium": float(scored["final_premium"].max()),
        },
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    artifacts = get_artifacts()
    return {
        "metadata": artifacts["metadata"],
        "metrics": artifacts.get("metrics", {}),
    }


@app.get("/data-stats")
def data_stats() -> dict[str, Any]:
    """Row counts per Supabase table (for the admin/flywheel dashboard)."""
    return table_counts()