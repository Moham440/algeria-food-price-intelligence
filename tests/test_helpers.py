"""
Unit tests for the utility helpers module.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from src.utils.helpers import (
    add_temporal_features,
    compute_metrics,
    format_price,
    get_season,
    is_algerian_holiday,
    is_ramadan,
    make_cache_key,
    mean_absolute_percentage_error,
    pct_change_label,
    validate_dataframe,
)


class TestIsRamadan:
    def test_known_ramadan_day(self):
        assert is_ramadan(date(2023, 3, 25)) is True

    def test_non_ramadan_day(self):
        assert is_ramadan(date(2023, 7, 1)) is False

    def test_unknown_year_returns_false(self):
        assert is_ramadan(date(2050, 6, 1)) is False

    def test_accepts_datetime(self):
        assert isinstance(is_ramadan(datetime(2022, 4, 5)), bool)


class TestIsAlgerianHoliday:
    def test_independence_day(self):
        assert is_algerian_holiday(date(2023, 7, 5)) is True

    def test_revolution_day(self):
        assert is_algerian_holiday(date(2023, 11, 1)) is True

    def test_labour_day(self):
        assert is_algerian_holiday(date(2024, 5, 1)) is True

    def test_regular_day(self):
        assert is_algerian_holiday(date(2023, 3, 15)) is False


class TestGetSeason:
    @pytest.mark.parametrize(
        "month, expected",
        [
            (1, "Winter"), (2, "Winter"), (12, "Winter"),
            (3, "Spring"), (4, "Spring"), (5, "Spring"),
            (6, "Summer"), (7, "Summer"), (8, "Summer"),
            (9, "Autumn"), (10, "Autumn"), (11, "Autumn"),
        ],
    )
    def test_seasons(self, month, expected):
        assert get_season(date(2023, month, 15)) == expected


class TestAddTemporalFeatures:
    def test_adds_expected_columns(self):
        df = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=12, freq="MS")})
        result = add_temporal_features(df)
        for col in ("year", "month", "season", "is_ramadan", "is_holiday", "month_sin", "month_cos"):
            assert col in result.columns

    def test_cyclical_encoding_range(self):
        df = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=12, freq="MS")})
        result = add_temporal_features(df)
        assert result["month_sin"].between(-1, 1).all()
        assert result["month_cos"].between(-1, 1).all()

    def test_is_ramadan_column_is_int(self):
        df = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=24, freq="MS")})
        result = add_temporal_features(df)
        assert result["is_ramadan"].isin([0, 1]).all()


class TestValidateDataFrame:
    def test_passes_valid_df(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert validate_dataframe(df, required_columns=["a", "b"]) is True

    def test_raises_on_none(self):
        with pytest.raises(ValueError, match="None or empty"):
            validate_dataframe(None, required_columns=["a"])

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df, required_columns=["a", "b"])

    def test_raises_on_too_few_rows(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="at least"):
            validate_dataframe(df, required_columns=["a"], min_rows=5)


class TestMAPE:
    def test_exact_prediction_is_zero(self):
        y = np.array([100.0, 200.0, 300.0])
        assert mean_absolute_percentage_error(y, y) == pytest.approx(0.0)

    def test_double_prediction(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([200.0, 400.0])
        assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(100.0)

    def test_ignores_zero_actual(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([50.0, 110.0])
        result = mean_absolute_percentage_error(y_true, y_pred)
        assert result == pytest.approx(10.0)


class TestComputeMetrics:
    def test_returns_expected_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y, y)
        assert set(m.keys()) == {"MAE", "RMSE", "MAPE"}

    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0])
        m = compute_metrics(y, y)
        assert m["MAE"] == pytest.approx(0.0)
        assert m["RMSE"] == pytest.approx(0.0)


class TestFormatting:
    def test_format_price(self):
        assert "DZD" in format_price(1234.5)
        assert "1,234" in format_price(1234.5)

    def test_pct_change_label_positive(self):
        label = pct_change_label(5.2)
        assert label.startswith("+")

    def test_pct_change_label_negative(self):
        label = pct_change_label(-3.1)
        assert label.startswith("-")


class TestMakeCacheKey:
    def test_deterministic(self):
        k1 = make_cache_key("func", 1, 2, kwarg="val")
        k2 = make_cache_key("func", 1, 2, kwarg="val")
        assert k1 == k2

    def test_different_args_give_different_keys(self):
        k1 = make_cache_key("func", 1)
        k2 = make_cache_key("func", 2)
        assert k1 != k2
