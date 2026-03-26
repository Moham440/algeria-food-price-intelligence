"""
Unit tests for the DataCleaner preprocessing module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.data_cleaner import DataCleaner


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Generate a small clean DataFrame for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    rows = []
    for product in ["Tomates", "Oignons"]:
        for region in ["Alger", "Oran"]:
            prices = rng.uniform(50, 150, len(dates))
            for dt, p in zip(dates, prices):
                rows.append({"date": dt, "product": product, "region": region, "price": round(p, 2)})
    return pd.DataFrame(rows)


@pytest.fixture
def cleaner() -> DataCleaner:
    """Return a DataCleaner with default config."""
    config = {
        "preprocessing": {
            "missing_value_strategy": "interpolate",
            "outlier_method": "iqr",
            "outlier_threshold": 3.0,
            "min_data_points": 10,
        },
        "storage": {"processed_data_path": "/tmp/processed"},
    }
    return DataCleaner(config=config)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDataCleanerInit:
    def test_default_attributes(self, cleaner):
        assert cleaner.missing_strategy == "interpolate"
        assert cleaner.outlier_method == "iqr"
        assert cleaner.outlier_threshold == 3.0

    def test_invalid_config_still_initialises(self):
        """DataCleaner should handle partial config gracefully."""
        c = DataCleaner(
            config={
                "preprocessing": {},
                "storage": {"processed_data_path": "/tmp"},
            }
        )
        assert c.missing_strategy == "interpolate"


class TestStandardiseTypes:
    def test_price_coercion(self, cleaner):
        df = pd.DataFrame(
            {"date": ["2023-01-01", "2023-02-01"], "price": ["100.5", "  80"]}
        )
        result = cleaner._standardise_types(df)
        assert result["price"].dtype == float

    def test_future_dates_removed(self, cleaner):
        future = pd.Timestamp.today() + pd.DateOffset(months=3)
        df = pd.DataFrame(
            {"date": [pd.Timestamp("2023-01-01"), future], "price": [100.0, 200.0]}
        )
        result = cleaner._standardise_types(df)
        assert len(result) == 1

    def test_negative_prices_removed(self, cleaner):
        df = pd.DataFrame(
            {"date": pd.date_range("2023-01-01", periods=3, freq="MS"), "price": [100, -5, 0]}
        )
        result = cleaner._standardise_types(df)
        assert (result["price"] > 0).all()


class TestDuplicateRemoval:
    def test_removes_exact_duplicates(self, cleaner):
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2023-01-01")] * 3,
                "product": ["Tomates"] * 3,
                "region": ["Alger"] * 3,
                "price": [100.0] * 3,
            }
        )
        result = cleaner._remove_duplicates(df)
        assert len(result) == 1

    def test_non_duplicates_preserved(self, cleaner, sample_df):
        result = cleaner._remove_duplicates(sample_df)
        assert len(result) == len(sample_df)


class TestMissingValues:
    def test_interpolate_fills_gaps(self, cleaner):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=6, freq="MS"),
                "product": ["Tomates"] * 6,
                "region": ["Alger"] * 6,
                "price": [100, np.nan, np.nan, 130, np.nan, 160],
            }
        )
        result = cleaner._handle_missing(df)
        assert result["price"].isna().sum() == 0

    def test_drop_strategy(self, cleaner):
        cleaner.missing_strategy = "drop"
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=4, freq="MS"),
                "product": ["A"] * 4,
                "region": ["B"] * 4,
                "price": [100, np.nan, 120, np.nan],
            }
        )
        result = cleaner._handle_missing(df)
        assert len(result) == 2

    def test_mean_strategy(self, cleaner):
        cleaner.missing_strategy = "mean"
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="MS"),
                "product": ["P"] * 5,
                "region": ["R"] * 5,
                "price": [100, 120, np.nan, 140, 160],
            }
        )
        result = cleaner._handle_missing(df)
        assert result["price"].isna().sum() == 0
        # Mean of [100,120,140,160] = 130
        assert result.iloc[2]["price"] == pytest.approx(130.0, rel=0.01)


class TestOutlierRemoval:
    def test_iqr_removes_extreme_spikes(self, cleaner, sample_df):
        # Add a massive outlier
        spike_row = sample_df.iloc[0].copy()
        spike_row["price"] = 99_999.0
        df_with_spike = pd.concat([sample_df, spike_row.to_frame().T], ignore_index=True)
        result = cleaner._outlier_iqr(df_with_spike, ["product", "region"])
        assert 99_999.0 not in result["price"].values

    def test_zscore_removes_outliers(self, cleaner, sample_df):
        cleaner.outlier_method = "zscore"
        spike_row = sample_df.iloc[0].copy()
        spike_row["price"] = 99_999.0
        df_with_spike = pd.concat([sample_df, spike_row.to_frame().T], ignore_index=True)
        result = cleaner._outlier_zscore(df_with_spike, ["product", "region"])
        assert 99_999.0 not in result["price"].values


class TestFullPipeline:
    def test_clean_returns_dataframe(self, cleaner, sample_df):
        result = cleaner.clean(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_clean_no_missing_after(self, cleaner, sample_df):
        sample_df.loc[sample_df.index[:5], "price"] = np.nan
        result = cleaner.clean(sample_df)
        assert result["price"].isna().sum() == 0

    def test_clean_raises_on_empty(self, cleaner):
        with pytest.raises(ValueError):
            cleaner.clean(pd.DataFrame())

    def test_clean_raises_on_missing_column(self, cleaner):
        df = pd.DataFrame({"date": ["2023-01-01"], "value": [100]})
        with pytest.raises(ValueError, match="Missing required columns"):
            cleaner.clean(df)


class TestAggregation:
    def test_aggregate_by_region(self, cleaner, sample_df):
        result = cleaner.aggregate_by_region(sample_df)
        assert "price" in result.columns
        assert "record_count" in result.columns
        assert result["price"].notna().all()

    def test_aggregation_reduces_rows(self, cleaner, sample_df):
        """Monthly aggregation should reduce the row count."""
        result = cleaner.aggregate_by_region(sample_df)
        assert len(result) <= len(sample_df)
