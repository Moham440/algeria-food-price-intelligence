"""
MLflow Experiment Tracking (Bonus).

Trains the ensemble forecasting model while logging all parameters,
metrics, and artefacts to an MLflow tracking server.

Usage:
    python scripts/train_with_mlflow.py --product Tomates --region Alger
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger

from src.data_ingestion.wfp_connector import WFPConnector
from src.models.anomaly_detector import AnomalyDetector
from src.models.price_predictor import EnsemblePredictor
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineering import FeatureEngineer
from src.utils.helpers import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Algeria food price models with MLflow.")
    parser.add_argument("--product", default="Tomates", help="Product name to train on.")
    parser.add_argument("--region", default="Alger", help="Region name to train on.")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon in months.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(str(ROOT / "config" / "config.yaml"))

    # Configure MLflow
    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "algeria-food-price"))

    # ── Data pipeline ─────────────────────────────────────────
    logger.info("Building data pipeline…")
    raw = WFPConnector(config=config).fetch_price_data()
    clean = DataCleaner(config=config).clean(raw)
    fe = FeatureEngineer(config=config)
    features = fe.build_features(clean)

    # ── Run 1: Anomaly Detector ───────────────────────────────
    with mlflow.start_run(run_name="anomaly_detector"):
        ad_cfg = config.get("anomaly_detection", {})
        mlflow.log_params({
            "alert_threshold": ad_cfg.get("alert_threshold", 0.7),
            "window_size": ad_cfg.get("window_size", 30),
            "product": args.product,
            "region": args.region,
        })

        detector = AnomalyDetector(config=config)
        detector.fit(features)
        scored = detector.predict(features)
        alerts = detector.generate_alerts(scored)

        n_anomalies = scored["is_anomaly"].sum()
        mlflow.log_metrics({
            "n_anomalies": int(n_anomalies),
            "anomaly_rate": float(scored["is_anomaly"].mean()),
            "n_alerts": len(alerts),
        })

        # Save and log model artefact
        model_path = detector.save("anomaly_mlflow")
        mlflow.log_artifact(str(model_path), artifact_path="models")
        logger.info(f"Anomaly detector logged — {n_anomalies} anomalies detected.")

    # ── Run 2: Ensemble Forecaster ───────────────────────────
    with mlflow.start_run(run_name=f"ensemble_{args.product}_{args.region}"):
        fc_cfg = config.get("forecasting", {})
        prophet_cfg = fc_cfg.get("prophet", {})
        lstm_cfg = fc_cfg.get("lstm", {})
        ens_cfg = fc_cfg.get("ensemble", {})

        mlflow.log_params({
            "product": args.product,
            "region": args.region,
            "horizon": args.horizon,
            "prophet_seasonality_mode": prophet_cfg.get("seasonality_mode", "multiplicative"),
            "prophet_changepoint_prior": prophet_cfg.get("changepoint_prior_scale", 0.05),
            "lstm_seq_len": lstm_cfg.get("sequence_length", 60),
            "lstm_hidden_units": str(lstm_cfg.get("hidden_units", [128, 64])),
            "lstm_dropout": lstm_cfg.get("dropout", 0.2),
            "lstm_epochs": lstm_cfg.get("epochs", 100),
            "ensemble_prophet_weight": ens_cfg.get("prophet_weight", 0.5),
            "ensemble_lstm_weight": ens_cfg.get("lstm_weight", 0.5),
        })

        predictor = EnsemblePredictor(config=config)
        metrics = predictor.evaluate(features, args.product, args.region)

        for model_name, m in metrics.items():
            safe_name = model_name.lower().replace(" ", "_")
            mlflow.log_metrics({
                f"{safe_name}_mae": m["MAE"],
                f"{safe_name}_rmse": m["RMSE"],
                f"{safe_name}_mape": m["MAPE"],
            })

        # Forecast and log as CSV artefact
        predictor.fit(features, args.product, args.region)
        forecast = predictor.predict(features, args.product, args.region, n_steps=args.horizon)
        forecast_path = ROOT / "models" / "saved" / "forecasting" / "forecast_output.csv"
        forecast_path.parent.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(forecast_path, index=False)
        mlflow.log_artifact(str(forecast_path), artifact_path="forecasts")

        logger.info(f"Ensemble metrics for {args.product}/{args.region}:")
        for model_name, m in metrics.items():
            logger.info(f"  {model_name}: MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}, MAPE={m['MAPE']:.1f}%")

    logger.info("MLflow experiment run complete. View at: mlflow ui")


if __name__ == "__main__":
    main()
