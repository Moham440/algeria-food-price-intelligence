"""
Master Pipeline Script — Algeria Food Price Intelligence System.

Runs the complete end-to-end pipeline:
    1. Data ingestion (FAO + WFP)
    2. Data cleaning
    3. Feature engineering
    4. Anomaly detection
    5. Price forecasting
    6. Export results

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --product Tomates --region Alger --no-lstm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger

from src.data_ingestion.wfp_connector import WFPConnector
from src.models.anomaly_detector import AnomalyDetector
from src.models.price_predictor import EnsemblePredictor
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineering import FeatureEngineer
from src.utils.helpers import load_config, save_parquet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Algeria Food Price Intelligence pipeline.")
    p.add_argument("--product", default="Tomates")
    p.add_argument("--region", default="Alger")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--year-start", type=int, default=2015)
    p.add_argument("--year-end", type=int, default=2024)
    p.add_argument("--no-lstm", action="store_true", help="Skip LSTM training (faster).")
    p.add_argument("--skip-ingestion", action="store_true", help="Use cached raw data if available.")
    p.add_argument("--force", action="store_true", help="Delete cached data and re-ingest everything.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(str(ROOT / "config" / "config.yaml"))

    logger.info("=" * 60)
    logger.info("Algeria Food Price Intelligence — Full Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Ingestion ─────────────────────────────────────
    raw_wfp = ROOT / "data" / "raw" / "wfp" / "wfp_algeria_prices.parquet"

    # --force: wipe cached files so data is fully re-ingested
    if getattr(args, "force", False):
        for cached in [raw_wfp,
                       ROOT / "data" / "processed" / "features.parquet",
                       ROOT / "data" / "processed" / "alerts.csv"]:
            if cached.exists():
                cached.unlink()
                logger.info(f"  Deleted cache: {cached.name}")

    if args.skip_ingestion and raw_wfp.exists():
        logger.info("Step 1/5 — Loading cached WFP data…")
        import pandas as pd
        raw = pd.read_parquet(raw_wfp)
    else:
        logger.info("Step 1/5 — Data Ingestion…")
        wfp = WFPConnector(config=config)
        raw = wfp.fetch_price_data()
        save_parquet(raw, raw_wfp)
        logger.info(f"  WFP: {len(raw):,} records")

    # ── Step 2: Cleaning ──────────────────────────────────────
    logger.info("Step 2/5 — Data Cleaning…")
    cleaner = DataCleaner(config=config)
    clean = cleaner.clean(raw)
    logger.info(f"  Clean: {len(clean):,} records")

    # ── Step 3: Feature Engineering ───────────────────────────
    logger.info("Step 3/5 — Feature Engineering…")
    fe = FeatureEngineer(config=config)
    features_path = fe.build_and_save(clean, filename="features.parquet")
    import pandas as pd
    features = pd.read_parquet(features_path)
    logger.info(f"  Features: {features.shape}")

    # ── Step 4: Anomaly Detection ──────────────────────────────
    logger.info("Step 4/5 — Anomaly Detection…")
    detector = AnomalyDetector(config=config)
    detector.fit(features)
    scored = detector.predict(features)
    alerts = detector.generate_alerts(scored)

    n_anomalies = scored["is_anomaly"].sum()
    logger.info(f"  Anomalies: {n_anomalies:,} | Alerts: {len(alerts):,}")

    alerts_df = detector.alerts_to_dataframe()
    if not alerts_df.empty:
        alerts_path = ROOT / "data" / "processed" / "alerts.csv"
        alerts_df.to_csv(alerts_path, index=False)
        logger.info(f"  Alerts saved → {alerts_path}")

    detector.save("anomaly_pipeline")

    # ── Step 5: Forecasting ───────────────────────────────────
    available_products = sorted(features["product"].unique().tolist())
    available_regions  = sorted(features["region"].unique().tolist())

    # Find the product/region combo with the most data (≥ 24 months preferred)
    def best_match(requested: str, available: list[str]) -> str:
        req_lower = requested.lower()
        for a in available:
            if a.lower() == req_lower:
                return a
        for a in available:
            if req_lower in a.lower() or a.lower() in req_lower:
                return a
        return None  # no match

    def pick_best_combo(df: pd.DataFrame, req_product: str, req_region: str) -> tuple[str, str]:
        """Return (product, region) with best data coverage."""
        # Count months per combo
        counts = (
            df.groupby(["product", "region"])["date"]
            .nunique()
            .reset_index(name="n_months")
            .sort_values("n_months", ascending=False)
        )
        # Try exact/partial match first
        p_match = best_match(req_product, sorted(df["product"].unique()))
        r_match = best_match(req_region,  sorted(df["region"].unique()))

        if p_match and r_match:
            row = counts[(counts["product"] == p_match) & (counts["region"] == r_match)]
            if not row.empty and row.iloc[0]["n_months"] >= 24:
                return p_match, r_match

        # Fallback: pick combo with most data overall
        best = counts.iloc[0]
        logger.warning(
            f"Requested '{req_product}/{req_region}' has insufficient data. "
            f"Using best available: '{best['product']}/{best['region']}' "
            f"({best['n_months']} months)."
        )
        return best["product"], best["region"]

    product, region = pick_best_combo(features, args.product, args.region)

    logger.info(f"Step 5/5 — Forecasting ({product}/{region})…")
    logger.info(f"  Available products : {available_products}")
    logger.info(f"  Available regions  : {available_regions}")

    predictor = EnsemblePredictor(config=config)

    # If --no-lstm is set, disable LSTM by overriding weights
    if args.no_lstm:
        predictor.prophet_weight = 1.0
        predictor.lstm_weight = 0.0

    predictor.fit(features, product, region)

    key = f"{product}_{region}"
    if key not in predictor._fitted:
        logger.error(
            f"Could not fit model for {product}/{region}. "
            f"Check that this combination has enough data (≥ 24 months)."
        )
        logger.info("Pipeline completed with anomaly detection only.")
    else:
        forecast = predictor.predict(features, product, region, n_steps=args.horizon)
        metrics  = predictor.evaluate(features, product, region)

        forecast_out = ROOT / "data" / "processed" / f"forecast_{product}_{region}.csv"
        forecast.to_csv(forecast_out, index=False)
        logger.info(f"  Forecast saved → {forecast_out}")

        logger.info("  Metrics:")
        for model_name, m in metrics.items():
            logger.info(f"    {model_name:10s}: MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  MAPE={m['MAPE']:.1f}%")

    logger.info("=" * 60)
    logger.info("Pipeline complete ✅")
    logger.info("  → Launch dashboard: streamlit run dashboard/app.py")
    logger.info("  → Launch API:       uvicorn api.main:app --reload")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
