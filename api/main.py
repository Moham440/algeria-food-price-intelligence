"""
Algeria Food Price Intelligence — FastAPI REST API (Bonus).

Provides HTTP endpoints for:
    GET  /health              — liveness probe
    GET  /products            — list monitored products
    GET  /regions             — list monitored regions
    POST /predict             — price prediction for a product/region
    POST /anomaly/score       — compute anomaly score for a price observation
    GET  /alerts              — retrieve latest anomaly alerts
    GET  /prices              — query historical prices with filters

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from src.data_ingestion.wfp_connector import WFPConnector
from src.models.anomaly_detector import AnomalyDetector
from src.models.price_predictor import EnsemblePredictor
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineering import FeatureEngineer
from src.utils.helpers import load_config

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Algeria Food Price Intelligence API",
    description="REST API for food price monitoring, anomaly detection, and forecasting.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State (lazy-loaded) ────────────────────────────────────────────────

_config: dict | None = None
_features_df: pd.DataFrame | None = None
_detector: AnomalyDetector | None = None
_predictor: EnsemblePredictor | None = None


def _get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config(str(ROOT / "config" / "config.yaml"))
    return _config


def _get_features() -> pd.DataFrame:
    global _features_df
    if _features_df is None:
        config = _get_config()
        processed = ROOT / "data" / "processed" / "features.parquet"
        if processed.exists():
            _features_df = pd.read_parquet(processed)
        else:
            raw = WFPConnector(config=config).fetch_price_data()
            clean = DataCleaner(config=config).clean(raw)
            fe = FeatureEngineer(config=config)
            _features_df = fe.build_features(clean)
    return _features_df


def _get_detector() -> AnomalyDetector:
    global _detector
    if _detector is None:
        config = _get_config()
        model_path = ROOT / "models" / "saved" / "anomaly" / "anomaly_v1.joblib"
        if model_path.exists():
            _detector = AnomalyDetector.load(str(model_path), config=config)
        else:
            _detector = AnomalyDetector(config=config)
            _detector.fit(_get_features())
    return _detector


def _get_predictor() -> EnsemblePredictor:
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor(config=_get_config())
    return _predictor


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    product: str = Field(..., example="Tomates")
    region: str = Field(..., example="Alger")
    horizon_months: int = Field(default=6, ge=1, le=24, description="Forecast horizon in months")


class PredictResponse(BaseModel):
    product: str
    region: str
    horizon_months: int
    forecast: list[dict]
    model: str = "ensemble"


class AnomalyRequest(BaseModel):
    product: str = Field(..., example="Tomates")
    region: str = Field(..., example="Alger")
    price: float = Field(..., gt=0, example=320.0)
    date: date = Field(..., example="2024-06-01")


class AnomalyResponse(BaseModel):
    product: str
    region: str
    price: float
    date: str
    anomaly_score: float
    is_anomaly: bool
    severity: str


class AlertResponse(BaseModel):
    total: int
    alerts: list[dict]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Liveness probe."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
    )


@app.get("/products", tags=["Reference"])
def list_products():
    """Return the list of monitored food products."""
    df = _get_features()
    return {"products": sorted(df["product"].unique().tolist())}


@app.get("/regions", tags=["Reference"])
def list_regions():
    """Return the list of monitored Algerian regions (wilayas)."""
    df = _get_features()
    return {"regions": sorted(df["region"].unique().tolist())}


@app.get("/prices", tags=["Data"])
def get_prices(
    product: Optional[str] = Query(None, description="Filter by product name"),
    region: Optional[str] = Query(None, description="Filter by region"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(default=100, le=5000),
):
    """Query historical food prices with optional filters."""
    df = _get_features().copy()
    if product:
        df = df[df["product"] == product]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Product '{product}' not found.")
    if region:
        df = df[df["region"] == region]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Region '{region}' not found.")
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    cols = ["date", "product", "region", "price"]
    df = df[cols].sort_values("date", ascending=False).head(limit)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return {"count": len(df), "data": df.to_dict(orient="records")}


@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict_price(req: PredictRequest):
    """Forecast food prices using the Prophet + LSTM ensemble model."""
    df = _get_features()
    predictor = _get_predictor()

    # Check product/region existence
    products = df["product"].unique()
    regions = df["region"].unique()
    if req.product not in products:
        raise HTTPException(status_code=404, detail=f"Product '{req.product}' not found.")
    if req.region not in regions:
        raise HTTPException(status_code=404, detail=f"Region '{req.region}' not found.")

    key = f"{req.product}_{req.region}"
    if key not in predictor._fitted:
        try:
            predictor.fit(df, req.product, req.region)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Model fitting failed: {exc}")

    try:
        forecast = predictor.predict(df, req.product, req.region, n_steps=req.horizon_months)
        records = []
        for _, row in forecast.iterrows():
            records.append({
                "date": row["ds"].strftime("%Y-%m-%d") if hasattr(row["ds"], "strftime") else str(row["ds"]),
                "ensemble_yhat": round(float(row["ensemble_yhat"]), 2),
                "prophet_yhat": round(float(row.get("prophet_yhat", 0)), 2),
                "yhat_lower": round(float(row["yhat_lower"]), 2),
                "yhat_upper": round(float(row["yhat_upper"]), 2),
            })
        return PredictResponse(
            product=req.product,
            region=req.region,
            horizon_months=req.horizon_months,
            forecast=records,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/anomaly/score", response_model=AnomalyResponse, tags=["Anomaly Detection"])
def score_anomaly(req: AnomalyRequest):
    """Compute an anomaly score for a single price observation."""
    df = _get_features()
    detector = _get_detector()

    # Get historical prices for the group
    hist = df[
        (df["product"] == req.product) & (df["region"] == req.region)
    ].sort_values("date").copy()

    if hist.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data for {req.product}/{req.region}."
        )

    # Append the new observation
    new_row = hist.iloc[-1].copy()
    new_row["date"] = pd.Timestamp(req.date)
    new_row["price"] = req.price
    extended = pd.concat([hist, new_row.to_frame().T], ignore_index=True)

    scored = detector.predict(extended)
    last_row = scored.iloc[-1]
    score = float(last_row.get("anomaly_score", 0.0))
    is_anomaly = bool(last_row.get("is_anomaly", False))
    severity = AnomalyDetector._score_to_severity(score)

    return AnomalyResponse(
        product=req.product,
        region=req.region,
        price=req.price,
        date=str(req.date),
        anomaly_score=round(score, 4),
        is_anomaly=is_anomaly,
        severity=severity,
    )


@app.get("/alerts", response_model=AlertResponse, tags=["Anomaly Detection"])
def get_alerts(
    severity: Optional[str] = Query(None, description="Filter: LOW | MEDIUM | HIGH | CRITICAL"),
    product: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    limit: int = Query(default=50, le=500),
):
    """Retrieve the latest anomaly alerts."""
    df = _get_features()
    detector = _get_detector()

    scored = detector.predict(df)
    alerts = detector.generate_alerts(scored)
    alerts_df = detector.alerts_to_dataframe()

    if alerts_df.empty:
        return AlertResponse(total=0, alerts=[])

    if severity:
        alerts_df = alerts_df[alerts_df["severity"] == severity.upper()]
    if product:
        alerts_df = alerts_df[alerts_df["product"] == product]
    if region:
        alerts_df = alerts_df[alerts_df["region"] == region]

    alerts_df = alerts_df.sort_values("anomaly_score", ascending=False).head(limit)
    return AlertResponse(
        total=len(alerts_df),
        alerts=alerts_df.to_dict(orient="records"),
    )
