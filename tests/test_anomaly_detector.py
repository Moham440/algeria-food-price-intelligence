"""
Unit tests for the AnomalyDetector module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.anomaly_detector import AnomalyDetector, PriceAlert


@pytest.fixture
def config():
    return {
        "anomaly_detection": {
            "alert_threshold": 0.7,
            "window_size": 6,
            "algorithms": [
                {
                    "name": "isolation_forest",
                    "contamination": 0.05,
                    "n_estimators": 50,
                    "random_state": 42,
                },
                {
                    "name": "one_class_svm",
                    "nu": 0.05,
                    "kernel": "rbf",
                    "gamma": "auto",
                },
            ],
        },
        "storage": {"models_path": "/tmp/models"},
    }


@pytest.fixture
def price_df():
    """Generate clean monthly price data with injected spikes."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    rows = []
    for product in ["Tomates", "Poulet"]:
        for region in ["Alger", "Oran"]:
            base = 100 if product == "Tomates" else 350
            prices = base + rng.normal(0, base * 0.05, len(dates))
            # Inject 3 spikes
            prices[[10, 30, 55]] *= 3.0
            for dt, p in zip(dates, prices):
                rows.append(
                    {
                        "date": dt,
                        "product": product,
                        "region": region,
                        "price": max(p, 1),
                        "pct_change_1m": 0.0,
                    }
                )
    return pd.DataFrame(rows)


class TestAnomalyDetectorInit:
    def test_default_threshold(self, config):
        det = AnomalyDetector(config=config)
        assert det.alert_threshold == pytest.approx(0.7)

    def test_models_empty_before_fit(self, config):
        det = AnomalyDetector(config=config)
        assert len(det._models) == 0


class TestFit:
    def test_fit_creates_models(self, config, price_df):
        det = AnomalyDetector(config=config)
        det.fit(price_df)
        assert len(det._models) == 4  # 2 products × 2 regions

    def test_fit_raises_on_missing_columns(self, config):
        det = AnomalyDetector(config=config)
        bad_df = pd.DataFrame({"price": [100, 120], "region": ["Alger", "Oran"]})
        with pytest.raises(ValueError):
            det.fit(bad_df)


class TestPredict:
    def test_predict_adds_columns(self, config, price_df):
        det = AnomalyDetector(config=config)
        det.fit(price_df)
        result = det.predict(price_df)
        for col in ("anomaly_score", "is_anomaly", "if_score", "svm_score"):
            assert col in result.columns

    def test_anomaly_scores_in_range(self, config, price_df):
        det = AnomalyDetector(config=config)
        det.fit(price_df)
        result = det.predict(price_df)
        assert result["anomaly_score"].between(0, 1).all()

    def test_spikes_detected(self, config, price_df):
        det = AnomalyDetector(config=config)
        det.fit(price_df)
        result = det.predict(price_df)
        # At least one anomaly should be detected
        assert result["is_anomaly"].sum() >= 1


class TestAlerts:
    def test_generate_alerts_returns_list(self, config, price_df):
        det = AnomalyDetector(config=config)
        det.fit(price_df)
        scored = det.predict(price_df)
        alerts = det.generate_alerts(scored)
        assert isinstance(alerts, list)
        assert all(isinstance(a, PriceAlert) for a in alerts)

    def test_alert_severity_mapping(self, config):
        assert AnomalyDetector._score_to_severity(0.97) == "CRITICAL"
        assert AnomalyDetector._score_to_severity(0.87) == "HIGH"
        assert AnomalyDetector._score_to_severity(0.78) == "MEDIUM"
        assert AnomalyDetector._score_to_severity(0.72) == "LOW"

    def test_alerts_to_dataframe(self, config, price_df):
        det = AnomalyDetector(config=config)
        det.fit(price_df)
        scored = det.predict(price_df)
        det.generate_alerts(scored)
        df = det.alerts_to_dataframe()
        if not df.empty:
            assert "anomaly_score" in df.columns


class TestNormaliseScore:
    def test_output_range(self):
        scores = np.array([-0.5, 0.0, 0.5, 1.0])
        result = AnomalyDetector._normalise_score(scores)
        assert result.min() >= 0
        assert result.max() <= 1

    def test_constant_input(self):
        scores = np.array([0.3, 0.3, 0.3])
        result = AnomalyDetector._normalise_score(scores)
        assert (result == 0).all()


class TestPriceAlert:
    def test_to_dict(self):
        alert = PriceAlert(
            date=pd.Timestamp("2023-06-01"),
            product="Tomates",
            region="Alger",
            price=250.0,
            anomaly_score=0.88,
            algorithm="ensemble",
            severity="HIGH",
            message="Test alert",
        )
        d = alert.to_dict()
        assert d["product"] == "Tomates"
        assert d["anomaly_score"] == pytest.approx(0.88, rel=1e-3)
        assert d["severity"] == "HIGH"
