"""
Data Cleaning Pipeline for Algeria Food Price Intelligence System.

Handles missing values, outliers, duplicates, and type coercions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest

from src.utils.helpers import load_config, save_parquet, validate_dataframe


class DataCleaner:
    """Cleans raw food price DataFrames before feature engineering.

    Supports three outlier strategies:
        - ``'iqr'``: Remove points outside [Q1 - k·IQR, Q3 + k·IQR].
        - ``'zscore'``: Remove points where |z| > threshold.
        - ``'isolation_forest'``: Use an Isolation Forest per product group.

    Args:
        config: Full project configuration dict (loads from file if None).
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()
        prep = config.get("preprocessing", {})
        self.missing_strategy: str = prep.get("missing_value_strategy", "interpolate")
        self.outlier_method: Literal["iqr", "zscore", "isolation_forest"] = prep.get(
            "outlier_method", "iqr"
        )
        self.outlier_threshold: float = prep.get("outlier_threshold", 3.0)
        self.min_data_points: int = prep.get("min_data_points", 30)
        self.processed_path = Path(config["storage"]["processed_data_path"])
        self.processed_path.mkdir(parents=True, exist_ok=True)

    # ── Main Entry-Point ──────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline.

        Steps:
            1. Validate input.
            2. Standardise column names and types.
            3. Remove exact duplicates.
            4. Handle missing values.
            5. Remove or cap outliers.
            6. Filter groups with insufficient data.

        Args:
            df: Raw DataFrame from FAO or WFP connector.

        Returns:
            Cleaned DataFrame.
        """
        validate_dataframe(df, required_columns=["date", "price"])

        n_raw = len(df)
        logger.info(f"Starting clean pipeline on {n_raw:,} rows…")

        df = self._standardise_types(df)
        df = self._remove_duplicates(df)
        df = self._handle_missing(df)
        df = self._remove_outliers(df)
        df = self._filter_sparse_groups(df)

        n_clean = len(df)
        logger.info(f"Cleaned: {n_raw:,} → {n_clean:,} rows ({n_raw - n_clean:,} removed).")
        return df.reset_index(drop=True)

    # ── Steps ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _standardise_types(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce column types to expected dtypes.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with corrected dtypes.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        if "price_usd" in df.columns:
            df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
        # Remove future dates
        df = df[df["date"] <= pd.Timestamp.today()]
        # Remove non-positive prices
        df = df[df["price"] > 0]
        return df

    @staticmethod
    def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Drop exact duplicate rows.

        Args:
            df: Input DataFrame.

        Returns:
            De-duplicated DataFrame.
        """
        dupes = df.duplicated()
        n = dupes.sum()
        if n:
            logger.info(f"Removed {n:,} duplicate rows.")
        return df[~dupes].copy()

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill or drop missing price values.

        Strategy is controlled by ``self.missing_strategy``:
            - ``'interpolate'``: Linear interpolation within each product/region group.
            - ``'mean'``: Fill with group mean.
            - ``'median'``: Fill with group median.
            - ``'drop'``: Drop rows with missing price.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values handled.
        """
        n_missing = df["price"].isna().sum()
        if n_missing == 0:
            return df

        logger.info(f"Handling {n_missing:,} missing price values ({self.missing_strategy})…")

        if self.missing_strategy == "drop":
            return df.dropna(subset=["price"]).copy()

        group_cols = [c for c in ["product", "region", "market"] if c in df.columns]

        if self.missing_strategy == "interpolate":
            df = df.sort_values("date")
            if group_cols:
                df["price"] = df.groupby(group_cols)["price"].transform(
                    lambda s: s.interpolate(method="linear").ffill().bfill()
                )
            else:
                df["price"] = df["price"].interpolate(method="linear").ffill().bfill()

        elif self.missing_strategy in ("mean", "median"):
            agg_fn = "mean" if self.missing_strategy == "mean" else "median"
            if group_cols:
                fill_val = df.groupby(group_cols)["price"].transform(agg_fn)
                df["price"] = df["price"].fillna(fill_val)
            else:
                df["price"] = df["price"].fillna(df["price"].agg(agg_fn))

        # Any remaining NaNs → drop
        remaining = df["price"].isna().sum()
        if remaining:
            df = df.dropna(subset=["price"]).copy()
            logger.warning(f"Dropped {remaining:,} rows with unfillable NaN prices.")

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove or clip outlier price observations.

        Args:
            df: Input DataFrame with a ``price`` column.

        Returns:
            Filtered DataFrame.
        """
        group_cols = [c for c in ["product", "region"] if c in df.columns]
        method = self.outlier_method
        logger.info(f"Outlier removal via '{method}' method…")

        if method == "iqr":
            return self._outlier_iqr(df, group_cols)
        if method == "zscore":
            return self._outlier_zscore(df, group_cols)
        if method == "isolation_forest":
            return self._outlier_isolation_forest(df, group_cols)

        logger.warning(f"Unknown outlier method '{method}', skipping.")
        return df

    def _outlier_iqr(self, df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        """IQR-based outlier removal.

        Args:
            df: Input DataFrame.
            group_cols: Columns to group by before computing IQR.

        Returns:
            Filtered DataFrame.
        """
        k = self.outlier_threshold

        def _iqr_mask(series: pd.Series) -> pd.Series:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            return (series >= q1 - k * iqr) & (series <= q3 + k * iqr)

        if group_cols:
            mask = df.groupby(group_cols)["price"].transform(_iqr_mask).astype(bool)
        else:
            mask = _iqr_mask(df["price"])

        removed = (~mask).sum()
        if removed:
            logger.info(f"IQR filter removed {removed:,} outliers.")
        return df[mask].copy()

    def _outlier_zscore(self, df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        """Z-score based outlier removal.

        Args:
            df: Input DataFrame.
            group_cols: Columns to group by.

        Returns:
            Filtered DataFrame.
        """
        threshold = self.outlier_threshold

        def _z_mask(series: pd.Series) -> pd.Series:
            mu, sigma = series.mean(), series.std()
            if sigma == 0:
                return pd.Series(True, index=series.index)
            return ((series - mu) / sigma).abs() <= threshold

        if group_cols:
            mask = df.groupby(group_cols)["price"].transform(_z_mask).astype(bool)
        else:
            mask = _z_mask(df["price"])

        removed = (~mask).sum()
        if removed:
            logger.info(f"Z-score filter removed {removed:,} outliers.")
        return df[mask].copy()

    def _outlier_isolation_forest(
        self, df: pd.DataFrame, group_cols: list[str]
    ) -> pd.DataFrame:
        """Isolation Forest outlier removal per product/region group.

        Args:
            df: Input DataFrame.
            group_cols: Columns to group by.

        Returns:
            Filtered DataFrame.
        """
        df = df.copy()
        df["_keep"] = True

        if not group_cols:
            model = IsolationForest(contamination=0.05, random_state=42)
            preds = model.fit_predict(df[["price"]])
            df["_keep"] = preds == 1
        else:
            for keys, grp in df.groupby(group_cols):
                if len(grp) < 10:
                    continue
                model = IsolationForest(contamination=0.05, random_state=42)
                preds = model.fit_predict(grp[["price"]])
                df.loc[grp.index, "_keep"] = preds == 1

        removed = (~df["_keep"]).sum()
        if removed:
            logger.info(f"Isolation Forest removed {removed:,} outliers.")
        return df[df["_keep"]].drop(columns=["_keep"]).copy()

    def _filter_sparse_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove product/region groups that have too few observations.

        Args:
            df: Input DataFrame.

        Returns:
            Filtered DataFrame.
        """
        group_cols = [c for c in ["product", "region"] if c in df.columns]
        if not group_cols:
            return df

        counts = df.groupby(group_cols).size()
        valid = counts[counts >= self.min_data_points].index
        if isinstance(valid, pd.MultiIndex):
            mask = df.set_index(group_cols).index.isin(valid)
        else:
            mask = df[group_cols[0]].isin(valid)

        removed_groups = len(counts) - len(valid)
        if removed_groups:
            logger.info(
                f"Dropped {removed_groups} sparse groups "
                f"(< {self.min_data_points} records)."
            )
        return df[mask].reset_index(drop=True)

    # ── Region Aggregation ────────────────────────────────────────────────────

    def aggregate_by_region(
        self, df: pd.DataFrame, freq: str = "MS"
    ) -> pd.DataFrame:
        """Aggregate daily/weekly prices to a monthly regional mean.

        Args:
            df: Cleaned DataFrame with ``date``, ``product``, ``region``, ``price``.
            freq: Resampling frequency (default ``'MS'`` = month start).

        Returns:
            Aggregated DataFrame with mean price and record count.
        """
        required = ["date", "product", "region", "price"]
        validate_dataframe(df, required_columns=required)

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        agg = (
            df.groupby(["product", "region"])["price"]
            .resample(freq)
            .agg(price_mean="mean", price_std="std", record_count="count")
            .reset_index()
        )
        agg.rename(columns={"price_mean": "price"}, inplace=True)
        logger.info(f"Aggregated to {len(agg):,} monthly region-level records.")
        return agg

    # ── Save ──────────────────────────────────────────────────────────────────

    def clean_and_save(self, df: pd.DataFrame, filename: str = "cleaned.parquet") -> Path:
        """Clean data and persist to the processed directory.

        Args:
            df: Raw DataFrame.
            filename: Output filename.

        Returns:
            Path to the saved Parquet file.
        """
        cleaned = self.clean(df)
        out = self.processed_path / filename
        save_parquet(cleaned, out)
        return out
