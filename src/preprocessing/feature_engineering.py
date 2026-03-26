"""
Feature Engineering for Algeria Food Price Intelligence System.

Builds temporal, lag, rolling, and economic features suitable for
the downstream anomaly detection and price forecasting models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utils.helpers import add_temporal_features, load_config, save_parquet, validate_dataframe


class FeatureEngineer:
    """Constructs the full feature matrix from cleaned price data.

    Args:
        config: Full project configuration dict.
    """

    _SCALER_MAP = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "robust": RobustScaler,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        prep = config.get("preprocessing", {})
        self.normalization: str = prep.get("normalization", "minmax")
        self.processed_path = Path(config["storage"]["processed_data_path"])
        self._scalers: dict[str, object] = {}  # keyed by (product, region)

    # ── Main Entry-Point ──────────────────────────────────────────────────────

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the complete feature engineering pipeline.

        Steps:
            1. Temporal / calendar features.
            2. Lag features (1, 7, 14, 30 days).
            3. Rolling statistics (7-day and 30-day windows).
            4. Percentage change features.
            5. Normalise prices per product/region group.

        Args:
            df: Cleaned DataFrame with at minimum:
                date, product, region, price.

        Returns:
            DataFrame enriched with engineered features.
        """
        validate_dataframe(df, required_columns=["date", "product", "region", "price"])

        df = df.copy().sort_values(["product", "region", "date"])
        logger.info(f"Feature engineering on {len(df):,} rows…")

        df = add_temporal_features(df, date_col="date")
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_pct_change_features(df)
        df = self._add_price_ratio_features(df)
        df = self._normalise(df)

        logger.info(f"Feature matrix shape: {df.shape}")
        return df.reset_index(drop=True)

    # ── Feature Steps ─────────────────────────────────────────────────────────

    @staticmethod
    def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged price observations within each product/region group.

        Lags: 1, 2, 3, 6, 12 months (assumes monthly aggregated input).

        Args:
            df: Sorted DataFrame.

        Returns:
            DataFrame with new lag_price_* columns.
        """
        group = ["product", "region"]
        lag_periods = [1, 2, 3, 6, 12]
        for lag in lag_periods:
            col = f"lag_price_{lag}m"
            df[col] = df.groupby(group)["price"].shift(lag)
            logger.debug(f"Added feature: {col}")
        return df

    @staticmethod
    def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling mean, std, min, and max within each group.

        Windows: 3-month and 6-month.

        Args:
            df: Sorted DataFrame.

        Returns:
            DataFrame with rolling_* columns.
        """
        group = ["product", "region"]
        windows = [3, 6]

        for w in windows:
            prefix = f"rolling_{w}m"
            roll = df.groupby(group)["price"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1)
            )
            # We need to call agg methods on the rolled object
            df[f"{prefix}_mean"] = df.groupby(group)["price"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).mean()
            )
            df[f"{prefix}_std"] = df.groupby(group)["price"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).std().fillna(0)
            )
            df[f"{prefix}_min"] = df.groupby(group)["price"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).min()
            )
            df[f"{prefix}_max"] = df.groupby(group)["price"].transform(
                lambda s, ww=w: s.rolling(ww, min_periods=1).max()
            )
        return df

    @staticmethod
    def _add_pct_change_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add month-over-month and year-over-year price changes.

        Args:
            df: Sorted DataFrame.

        Returns:
            DataFrame with pct_change_1m and pct_change_12m columns.
        """
        group = ["product", "region"]
        df["pct_change_1m"] = df.groupby(group)["price"].pct_change(1) * 100
        df["pct_change_3m"] = df.groupby(group)["price"].pct_change(3) * 100
        df["pct_change_12m"] = df.groupby(group)["price"].pct_change(12) * 100
        return df

    @staticmethod
    def _add_price_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute each region's price relative to the national mean.

        For each product and date, calculate:
            price_vs_national = price / mean_price_across_regions

        Args:
            df: Sorted DataFrame.

        Returns:
            DataFrame with price_vs_national column.
        """
        national_mean = df.groupby(["product", "date"])["price"].transform("mean")
        df["price_vs_national"] = df["price"] / national_mean.replace(0, np.nan)
        return df

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise prices per (product, region) group.

        Fits a scaler on each group and stores it for inverse-transform.
        Adds a ``price_norm`` column.

        Args:
            df: DataFrame with a ``price`` column.

        Returns:
            DataFrame with an additional ``price_norm`` column.
        """
        ScalerClass = self._SCALER_MAP.get(self.normalization, MinMaxScaler)
        df["price_norm"] = np.nan
        group_cols = ["product", "region"]

        for keys, grp in df.groupby(group_cols):
            key_str = "_".join(str(k) for k in (keys if isinstance(keys, tuple) else (keys,)))
            values = grp["price"].values.reshape(-1, 1)
            scaler = ScalerClass()
            normed = scaler.fit_transform(values).flatten()
            df.loc[grp.index, "price_norm"] = normed
            self._scalers[key_str] = scaler

        logger.info(f"Normalised prices for {len(self._scalers)} groups.")
        return df

    def inverse_normalise(
        self, values: np.ndarray, product: str, region: str
    ) -> np.ndarray:
        """Reverse normalisation for a given product/region group.

        Args:
            values: Normalised price array.
            product: Product name.
            region: Region name.

        Returns:
            Array of original-scale prices.
        """
        key = f"{product}_{region}"
        if key not in self._scalers:
            raise KeyError(f"No scaler found for key '{key}'.")
        scaler = self._scalers[key]
        return scaler.inverse_transform(values.reshape(-1, 1)).flatten()

    # ── Sequence Builder (for LSTM) ───────────────────────────────────────────

    def build_sequences(
        self,
        df: pd.DataFrame,
        product: str,
        region: str,
        sequence_length: int = 60,
        target_col: str = "price_norm",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build sliding-window sequences for LSTM training.

        Args:
            df: Feature-engineered DataFrame.
            product: Product to filter.
            region: Region to filter.
            sequence_length: Length of each input sequence.
            target_col: Column to predict.

        Returns:
            Tuple (X, y) of shape
                X: (n_samples, sequence_length, n_features)
                y: (n_samples,)
        """
        subset = df[(df["product"] == product) & (df["region"] == region)].copy()
        subset = subset.sort_values("date")

        feature_cols = [
            target_col,
            "month_sin", "month_cos",
            "is_ramadan", "is_holiday",
            "pct_change_1m", "pct_change_12m",
            "price_vs_national",
        ]
        feature_cols = [c for c in feature_cols if c in subset.columns]
        values = subset[feature_cols].fillna(0).values

        X, y = [], []
        for i in range(len(values) - sequence_length):
            X.append(values[i : i + sequence_length])
            y.append(values[i + sequence_length, 0])  # target is first column

        return np.array(X), np.array(y)

    # ── Save ──────────────────────────────────────────────────────────────────

    def build_and_save(
        self, df: pd.DataFrame, filename: str = "features.parquet"
    ) -> Path:
        """Build features and save to processed directory.

        Args:
            df: Cleaned input DataFrame.
            filename: Output Parquet filename.

        Returns:
            Path to the saved file.
        """
        features = self.build_features(df)
        out = self.processed_path / filename
        save_parquet(features, out)
        return out
