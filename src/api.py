from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import set_global_determinism
from src.data_loader import DataValidationError, build_inference_frame
from src.frequency_model import predict_frequency
from src.logger import get_logger
from src.model_artifacts import load_artifacts
from src.pricing_engine import calculate_premium
from src.severity_model import predict_severity

logger = get_logger(__name__)

app = FastAPI(
    title="Motor Insurance Pricing API",
    version="0.2.0",
    description="Production REST API for actuarial motor insurance premium prediction",
)
router = APIRouter(prefix="/api/v1")


class PolicyInput(BaseModel):
    Exposure: float = Field(gt=0, le=2.5, description="Policy exposure in years")
    VehPower: float = Field(ge=1, le=20, description="Vehicle horsepower class")
    VehAge: float = Field(ge=0, le=100, description="Vehicle age in years")
    DrivAge: float = Field(ge=18, le=100, description="Driver age in years")
    BonusMalus: float = Field(ge=50, le=350, description="Bonus-malus coefficient")
    VehBrand: str = Field(description="Vehicle brand code (e.g. B12)")
    VehGas: str = Field(description="Fuel type: Regular or Diesel")
    Area: str = Field(description="Area risk segment (A-F)")
    Density: float = Field(ge=0, le=27000, description="Population density")
    Region: str = Field(description="French region")


class PremiumOutput(BaseModel):
    predicted_annual_frequency: float
    predicted_claim_count: float
    predicted_claim_severity: float
    annualized_expected_loss: float
    expected_loss: float
    pure_premium: float
    technical_premium: float
    final_premium: float
    risk_score: float
    risk_category: str
    frequency_relativity: float
    severity_relativity: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


_artifacts: dict[str, Any] | None = None


def _get_artifacts() -> dict[str, Any]:
    global _artifacts
    if _artifacts is None:
        _artifacts = load_artifacts()
        seed = int(_artifacts["metadata"].get("random_seed", 42))
        set_global_determinism(seed)
        logger.info("Loaded model artifacts with seed %d", seed)
    return _artifacts


@router.get("/health", response_model=HealthResponse)
async def health():
    try:
        _get_artifacts()
        return HealthResponse(
            status="healthy", model_loaded=True, version="0.2.0"
        )
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        return HealthResponse(
            status="degraded", model_loaded=False, version="0.2.0"
        )


@router.post("/predict", response_model=list[PremiumOutput])
async def predict(policies: list[PolicyInput]) -> list[dict[str, Any]]:
    artifacts = _get_artifacts()
    metadata = artifacts["metadata"]

    rows = [policy.model_dump() for policy in policies]
    df = pd.DataFrame(rows)

    try:
        prepared = build_inference_frame(df, metadata)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    freq = predict_frequency(artifacts["frequency_model"], prepared)
    sev = predict_severity(artifacts["severity_model"], prepared)

    scored = calculate_premium(
        prepared,
        freq,
        sev,
        pricing_config=metadata.get("pricing_config"),
        risk_thresholds=metadata.get("risk_thresholds"),
        portfolio_baselines=metadata.get("portfolio_baselines"),
    )

    output_columns = [
        "predicted_annual_frequency",
        "predicted_claim_count",
        "predicted_claim_severity",
        "annualized_expected_loss",
        "expected_loss",
        "pure_premium",
        "technical_premium",
        "final_premium",
        "risk_score",
        "risk_category",
        "frequency_relativity",
        "severity_relativity",
    ]
    return scored[output_columns].to_dict(orient="records")


@router.post("/predict/csv")
async def predict_csv(file: UploadFile):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    artifacts = _get_artifacts()
    metadata = artifacts["metadata"]

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}")

    try:
        prepared = build_inference_frame(df, metadata)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    freq = predict_frequency(artifacts["frequency_model"], prepared)
    sev = predict_severity(artifacts["severity_model"], prepared)

    scored = calculate_premium(
        prepared,
        freq,
        sev,
        pricing_config=metadata.get("pricing_config"),
        risk_thresholds=metadata.get("risk_thresholds"),
        portfolio_baselines=metadata.get("portfolio_baselines"),
    )
    return scored.to_dict(orient="records")


app.include_router(router)
