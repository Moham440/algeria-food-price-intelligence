"""
Anomaly Detection Module for Algeria Food Price Intelligence System.

Implements Isolation Forest and One-Class SVM to detect abnormal
price spikes across Algerian food markets, with alerting support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.utils.helpers import load_config, validate_dataframe


@dataclass
class PriceAlert:
    """Represents a single anomaly alert.

    Attributes:
        date: Date of the anomaly.
        product: Food product name.
        region: Algerian region.
        price: Observed price (DZD).
        anomaly_score: Composite score in [0, 1]; higher = more anomalous.
        algorithm: Algorithm that flagged the point.
        severity: One of 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'.
        message: Human-readable description.
    """

    date: datetime
    product: str
    region: str
    price: float
    anomaly_score: float
    algorithm: str
    severity: str = "MEDIUM"
    message: str = ""

    def to_dict(self) -> dict:
        """Serialise alert to a plain dict."""
        return {
            "date": self.date.isoformat() if hasattr(self.date, "isoformat") else str(self.date),
            "product": self.product,
            "region": self.region,
            "price": self.price,
            "anomaly_score": round(self.anomaly_score, 4),
            "algorithm": self.algorithm,
            "severity": self.severity,
            "message": self.message,
        }


class AnomalyDetector:
    """Dual-algorithm anomaly detector (Isolation Forest + One-Class SVM).

    Detection workflow per (product, region) group:
        1. Extract price features (level, lag-diff, rolling-z-score).
        2. Fit both algorithms.
        3. Combine scores with a weighted average.
        4. Generate PriceAlert objects where score ≥ threshold.

    Args:
        config: Full project configuration dict.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        ad_cfg = config.get("anomaly_detection", {})
        self.alert_threshold: float = ad_cfg.get("alert_threshold", 0.7)
        self.window_size: int = ad_cfg.get("window_size", 30)

        # Algorithm hyperparameters
        if_cfg = next(
            (a for a in ad_cfg.get("algorithms", []) if a["name"] == "isolation_forest"),
            {},
        )
        svm_cfg = next(
            (a for a in ad_cfg.get("algorithms", []) if a["name"] == "one_class_svm"),
            {},
        )

        self._if_params = {
            "contamination": if_cfg.get("contamination", 0.05),
            "n_estimators": if_cfg.get("n_estimators", 100),
            "random_state": if_cfg.get("random_state", 42),
        }
        self._svm_params = {
            "nu": svm_cfg.get("nu", 0.05),
            "kernel": svm_cfg.get("kernel", "rbf"),
            "gamma": svm_cfg.get("gamma", "auto"),
        }

        self._models: dict[str, dict] = {}  # {group_key: {if: model, svm: model, scaler: scaler}}
        self.alerts: list[PriceAlert] = []
        self._models_path = Path(config["storage"].get("models_path", "models/saved")) / "anomaly"
        self._models_path.mkdir(parents=True, exist_ok=True)

    # ── Feature Construction ──────────────────────────────────────────────────

    @staticmethod
    def _build_detection_features(series: pd.Series) -> np.ndarray:
        """Build a 2-D feature matrix from a price time series.

        Features per time step:
            - price (raw)
            - price_diff_1: first difference
            - price_diff_pct: % change
            - rolling_mean_6: 6-period rolling mean
            - rolling_std_6: 6-period rolling std
            - z_score_6: z-score relative to rolling window

        Args:
            series: Sorted price Series (numeric).

        Returns:
            2-D numpy array of shape (n, 6).
        """
        s = series.values.astype(float)
        diff1 = np.diff(s, prepend=s[0])
        pct = np.where(s[:-1] != 0, diff1[1:] / s[:-1] * 100, 0)
        pct = np.append(0, pct)

        roll_mean = pd.Series(s).rolling(6, min_periods=1).mean().values
        roll_std = pd.Series(s).rolling(6, min_periods=1).std().fillna(0).values
        z_score = np.where(roll_std > 0, (s - roll_mean) / roll_std, 0)

        return np.column_stack([s, diff1, pct, roll_mean, roll_std, z_score])

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """Fit detectors for every (product, region) combination.

        Args:
            df: Feature-engineered DataFrame with columns:
                date, product, region, price.

        Returns:
            Self (for chaining).
        """
        validate_dataframe(df, required_columns=["date", "product", "region", "price"])
        groups = df.groupby(["product", "region"])
        logger.info(f"Fitting anomaly detectors for {len(groups)} groups…")

        for (product, region), grp in groups:
            grp = grp.sort_values("date")
            if len(grp) < 10:
                logger.debug(f"Skipping {product}/{region}: too few samples ({len(grp)}).")
                continue

            key = f"{product}_{region}"
            X = self._build_detection_features(grp["price"])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            iso = IsolationForest(**self._if_params)
            iso.fit(X_scaled)

            ocsvm = OneClassSVM(**self._svm_params)
            ocsvm.fit(X_scaled)

            self._models[key] = {"if": iso, "svm": ocsvm, "scaler": scaler}

        logger.info(f"Trained {len(self._models)} anomaly detectors.")
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score each row in the DataFrame for anomaly.

        Adds the following columns:
            - ``anomaly_score``: Composite score in [0, 1].
            - ``is_anomaly``: Boolean flag (score ≥ threshold).
            - ``if_score``: Raw Isolation Forest decision function.
            - ``svm_score``: Raw One-Class SVM decision function.

        Args:
            df: DataFrame to score (must contain date, product, region, price).

        Returns:
            DataFrame with anomaly score columns added.
        """
        validate_dataframe(df, required_columns=["date", "product", "region", "price"])
        df = df.copy()
        df["anomaly_score"] = 0.0
        df["if_score"] = 0.0
        df["svm_score"] = 0.0
        df["is_anomaly"] = False

        groups = df.groupby(["product", "region"])
        for (product, region), idx in groups.groups.items():
            key = f"{product}_{region}"
            if key not in self._models:
                logger.debug(f"No model for {product}/{region}, skipping.")
                continue

            grp = df.loc[idx].sort_values("date")
            models = self._models[key]
            X = self._build_detection_features(grp["price"])
            X_scaled = models["scaler"].transform(X)

            # Decision functions: negative = anomalous
            if_dec = models["if"].decision_function(X_scaled)
            svm_dec = models["svm"].decision_function(X_scaled)

            # Normalise to [0, 1] where 1 = most anomalous
            if_score = self._normalise_score(if_dec)
            svm_score = self._normalise_score(svm_dec)
            composite = 0.5 * if_score + 0.5 * svm_score

            df.loc[grp.index, "if_score"] = if_score
            df.loc[grp.index, "svm_score"] = svm_score
            df.loc[grp.index, "anomaly_score"] = composite
            df.loc[grp.index, "is_anomaly"] = composite >= self.alert_threshold

        n_anomalies = df["is_anomaly"].sum()
        logger.info(f"Detected {n_anomalies:,} anomalies ({self.alert_threshold:.0%} threshold).")
        return df

    @staticmethod
    def _normalise_score(decision: np.ndarray) -> np.ndarray:
        """Convert decision function scores (lower = more anomalous) to [0, 1].

        A score of 1 means maximally anomalous.

        Args:
            decision: Raw decision function values.

        Returns:
            Inverted and min-max scaled array.
        """
        inverted = -decision  # flip so high = anomalous
        lo, hi = inverted.min(), inverted.max()
        if hi == lo:
            return np.zeros_like(inverted)
        return (inverted - lo) / (hi - lo)

    # ── Alerting ──────────────────────────────────────────────────────────────

    def generate_alerts(self, scored_df: pd.DataFrame) -> list[PriceAlert]:
        """Generate PriceAlert objects for all flagged rows.

        Args:
            scored_df: Output of ``predict()``.

        Returns:
            List of PriceAlert instances.
        """
        anomaly_rows = scored_df[scored_df["is_anomaly"] == True]  # noqa: E712
        alerts = []

        for _, row in anomaly_rows.iterrows():
            score = row["anomaly_score"]
            severity = self._score_to_severity(score)
            pct_change = row.get("pct_change_1m", 0) or 0

            message = (
                f"{row['product']} price in {row['region']} on "
                f"{row['date'].strftime('%Y-%m') if hasattr(row['date'], 'strftime') else row['date']}: "
                f"{row['price']:.0f} DZD "
                f"({'▲' if pct_change >= 0 else '▼'}{abs(pct_change):.1f}% MoM). "
                f"Anomaly score: {score:.2f}."
            )

            alerts.append(
                PriceAlert(
                    date=row["date"],
                    product=row["product"],
                    region=row["region"],
                    price=row["price"],
                    anomaly_score=score,
                    algorithm="ensemble",
                    severity=severity,
                    message=message,
                )
            )

        self.alerts = alerts
        logger.info(f"Generated {len(alerts)} price alerts.")
        return alerts

    @staticmethod
    def _score_to_severity(score: float) -> str:
        """Map anomaly score to a severity label.

        Args:
            score: Float in [0, 1].

        Returns:
            Severity string.
        """
        if score >= 0.95:
            return "CRITICAL"
        if score >= 0.85:
            return "HIGH"
        if score >= 0.75:
            return "MEDIUM"
        return "LOW"

    def alerts_to_dataframe(self) -> pd.DataFrame:
        """Convert stored alerts to a DataFrame for display.

        Returns:
            DataFrame of alert records.
        """
        if not self.alerts:
            return pd.DataFrame()
        return pd.DataFrame([a.to_dict() for a in self.alerts])

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, name: str = "anomaly_detector") -> Path:
        """Persist trained models to disk using joblib.

        Args:
            name: Base filename (without extension).

        Returns:
            Path to the saved file.
        """
        out_path = self._models_path / f"{name}.joblib"
        joblib.dump(
            {"models": self._models, "threshold": self.alert_threshold},
            out_path,
        )
        logger.info(f"Anomaly detector saved to {out_path}")
        return out_path

    @classmethod
    def load(cls, path: str, config: Optional[dict] = None) -> "AnomalyDetector":
        """Load a previously saved detector from disk.

        Args:
            path: Path to the .joblib file.
            config: Optional config dict.

        Returns:
            Loaded AnomalyDetector instance.
        """
        data = joblib.load(path)
        instance = cls(config=config)
        instance._models = data["models"]
        instance.alert_threshold = data["threshold"]
        logger.info(f"Loaded anomaly detector from {path}")
        return instance
