"""
Price Forecasting Module for Algeria Food Price Intelligence System.

Implements:
    - Prophet  : seasonal decomposition + trend modelling
    - LSTM     : sequence-to-one deep learning (via TensorFlow/Keras)
    - Ensemble : weighted average of both models
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.helpers import compute_metrics, load_config, save_parquet

warnings.filterwarnings("ignore")


# ── Prophet Forecaster ────────────────────────────────────────────────────────

class ProphetForecaster:
    """Time-series forecaster based on Facebook/Meta Prophet.

    Prophet handles yearly and weekly seasonality out-of-the-box and is
    robust to missing data and outliers.

    Args:
        config: Full project configuration dict.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()
        prophet_cfg = config.get("forecasting", {}).get("prophet", {})
        self.seasonality_mode: str = prophet_cfg.get("seasonality_mode", "multiplicative")
        self.yearly: bool = prophet_cfg.get("yearly_seasonality", True)
        self.weekly: bool = prophet_cfg.get("weekly_seasonality", True)
        self.changepoint_prior: float = prophet_cfg.get("changepoint_prior_scale", 0.05)
        self.forecast_periods: int = prophet_cfg.get("forecast_periods", 30)
        self._models: dict[str, object] = {}  # {group_key: Prophet}

    def _get_prophet(self):
        """Lazy import of Prophet to avoid hard dependency at module level."""
        try:
            from prophet import Prophet  # type: ignore
            return Prophet
        except ImportError:
            raise ImportError(
                "prophet is not installed. Run: pip install prophet"
            )

    def fit(self, series: pd.DataFrame, product: str, region: str) -> "ProphetForecaster":
        """Fit a Prophet model for a single (product, region) pair.

        Args:
            series: DataFrame with columns ``ds`` (datetime) and ``y`` (price).
            product: Product name used for keying the model.
            region: Region name used for keying the model.

        Returns:
            Self.
        """
        Prophet = self._get_prophet()
        model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly,
            weekly_seasonality=self.weekly,
            changepoint_prior_scale=self.changepoint_prior,
            interval_width=0.95,
        )
        # Add Ramadan as a custom holiday/regressor
        model.add_country_holidays(country_name="DZ")
        model.fit(series[["ds", "y"]])
        key = f"{product}_{region}"
        self._models[key] = model
        logger.info(f"Prophet fitted for {product}/{region}.")
        return self

    def predict(
        self,
        product: str,
        region: str,
        periods: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate a forecast for a fitted (product, region) pair.

        Args:
            product: Product name.
            region: Region name.
            periods: Number of future periods to forecast (overrides config).

        Returns:
            Prophet forecast DataFrame with ds, yhat, yhat_lower, yhat_upper.
        """
        key = f"{product}_{region}"
        if key not in self._models:
            raise KeyError(f"No Prophet model found for '{key}'. Call fit() first.")
        model = self._models[key]
        periods = periods or self.forecast_periods
        future = model.make_future_dataframe(periods=periods, freq="MS")
        forecast = model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def evaluate(self, series: pd.DataFrame, product: str, region: str) -> dict:
        """Evaluate the in-sample fit with a train/test split (80/20).

        Args:
            series: Full ``ds``/``y`` DataFrame.
            product: Product name.
            region: Region name.

        Returns:
            Dict with MAE, RMSE, MAPE keys.
        """
        n = len(series)
        train_size = int(n * 0.8)
        train, test = series.iloc[:train_size], series.iloc[train_size:]

        self.fit(train, product, region)
        forecast = self.predict(product, region, periods=len(test))
        forecast = forecast.tail(len(test))
        y_true = test["y"].values
        y_pred = forecast["yhat"].values[: len(y_true)]
        return compute_metrics(y_true, y_pred)


# ── LSTM Forecaster ───────────────────────────────────────────────────────────

class LSTMForecaster:
    """Sequence-to-one LSTM price forecaster (TensorFlow/Keras backend).

    Args:
        config: Full project configuration dict.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()
        lstm_cfg = config.get("forecasting", {}).get("lstm", {})
        self.seq_len: int = lstm_cfg.get("sequence_length", 60)
        self.hidden_units: list[int] = lstm_cfg.get("hidden_units", [128, 64])
        self.dropout: float = lstm_cfg.get("dropout", 0.2)
        self.epochs: int = lstm_cfg.get("epochs", 100)
        self.batch_size: int = lstm_cfg.get("batch_size", 32)
        self.lr: float = lstm_cfg.get("learning_rate", 0.001)
        self.val_split: float = lstm_cfg.get("validation_split", 0.2)
        self._models: dict[str, object] = {}
        self._scalers: dict[str, object] = {}

    def _build_model(self, n_features: int):
        """Build the Keras LSTM model architecture.

        Args:
            n_features: Number of input features per time step.

        Returns:
            Compiled Keras model.
        """
        try:
            from tensorflow import keras  # type: ignore
            from tensorflow.keras import layers  # type: ignore
        except ImportError:
            raise ImportError("TensorFlow is not installed. Run: pip install tensorflow")

        model = keras.Sequential()
        for i, units in enumerate(self.hidden_units):
            return_sequences = i < len(self.hidden_units) - 1
            if i == 0:
                model.add(
                    layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        input_shape=(self.seq_len, n_features),
                    )
                )
            else:
                model.add(layers.LSTM(units, return_sequences=return_sequences))
            model.add(layers.Dropout(self.dropout))

        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(1))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss="mse",
            metrics=["mae"],
        )
        return model

    @staticmethod
    def _make_sequences(
        values: np.ndarray, seq_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slice a 2-D array into overlapping (X, y) sequences.

        Args:
            values: 2-D array (time, features); first column is the target.
            seq_len: Look-back window length.

        Returns:
            X of shape (n, seq_len, features) and y of shape (n,).
        """
        X, y = [], []
        for i in range(len(values) - seq_len):
            X.append(values[i : i + seq_len])
            y.append(values[i + seq_len, 0])
        return np.array(X), np.array(y)

    def fit(
        self,
        series: np.ndarray,
        product: str,
        region: str,
        feature_names: Optional[list[str]] = None,
    ) -> "LSTMForecaster":
        """Train an LSTM model for a single (product, region).

        Args:
            series: 2-D float array (time × features); column 0 = price_norm.
            product: Product name.
            region: Region name.
            feature_names: Optional list of feature names (for logging).

        Returns:
            Self.
        """
        from sklearn.preprocessing import MinMaxScaler  # type: ignore

        try:
            from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
        except ImportError:
            raise ImportError("TensorFlow is not installed. Run: pip install tensorflow")

        key = f"{product}_{region}"
        n_features = series.shape[1]

        # Normalise each feature column
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series)
        self._scalers[key] = scaler

        X, y = self._make_sequences(series_scaled, self.seq_len)
        if len(X) == 0:
            logger.warning(f"Not enough data for LSTM: {product}/{region}")
            return self

        model = self._build_model(n_features)
        cb = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.val_split,
            callbacks=[cb],
            verbose=0,
        )
        self._models[key] = model
        logger.info(f"LSTM fitted for {product}/{region} (seq_len={self.seq_len}).")
        return self

    def predict(
        self,
        series: np.ndarray,
        product: str,
        region: str,
        n_steps: int = 12,
    ) -> np.ndarray:
        """Iteratively forecast ``n_steps`` ahead.

        Uses the last ``seq_len`` observations as the initial seed.

        Args:
            series: Historical feature array (time × features).
            product: Product name.
            region: Region name.
            n_steps: Number of future steps to forecast.

        Returns:
            1-D array of forecasted (normalised) prices.
        """
        key = f"{product}_{region}"
        if key not in self._models:
            raise KeyError(f"No LSTM model for '{key}'.")

        model = self._models[key]
        scaler = self._scalers[key]
        series_scaled = scaler.transform(series)

        seed = series_scaled[-self.seq_len :].copy()
        preds = []
        for _ in range(n_steps):
            x = seed.reshape(1, self.seq_len, -1)
            p = model.predict(x, verbose=0)[0, 0]
            preds.append(p)
            # Shift seed and insert prediction as new "price" column
            new_row = seed[-1].copy()
            new_row[0] = p
            seed = np.vstack([seed[1:], new_row])

        return np.array(preds)

    def evaluate(
        self, series: np.ndarray, product: str, region: str
    ) -> dict:
        """Evaluate on a held-out test set (last 20%).

        Args:
            series: Full 2-D feature array.
            product: Product name.
            region: Region name.

        Returns:
            Dict with MAE, RMSE, MAPE.
        """
        split = int(len(series) * 0.8)
        train = series[:split]
        test = series[split:]

        self.fit(train, product, region)
        preds = self.predict(train, product, region, n_steps=len(test))

        # Inverse transform first column only
        scaler = self._scalers[f"{product}_{region}"]
        dummy = np.zeros((len(test), series.shape[1]))
        dummy[:, 0] = test[:, 0]
        y_true = scaler.inverse_transform(dummy)[:, 0]
        dummy2 = np.zeros((len(preds), series.shape[1]))
        dummy2[:, 0] = preds
        y_pred = scaler.inverse_transform(dummy2)[:, 0]

        return compute_metrics(y_true, y_pred)


# ── Ensemble Predictor ────────────────────────────────────────────────────────

class EnsemblePredictor:
    """Weighted ensemble of Prophet + LSTM forecasters.

    Args:
        config: Full project configuration dict.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()
        ens_cfg = config.get("forecasting", {}).get("ensemble", {})
        self.prophet_weight: float = ens_cfg.get("prophet_weight", 0.5)
        self.lstm_weight: float = ens_cfg.get("lstm_weight", 0.5)

        self.prophet = ProphetForecaster(config)
        self.lstm = LSTMForecaster(config)
        self._fitted: set[str] = set()
        self._models_path = Path(config["storage"].get("models_path", "models/saved")) / "forecasting"
        self._models_path.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        df: pd.DataFrame,
        product: str,
        region: str,
    ) -> "EnsemblePredictor":
        """Fit both sub-models for a (product, region) pair.

        Args:
            df: Monthly price DataFrame sorted by date with columns
                date, price, plus optional engineered features.
            product: Product name.
            region: Region name.

        Returns:
            Self.
        """
        subset = df[(df["product"] == product) & (df["region"] == region)].sort_values("date")
        if len(subset) == 0:
            available_products = sorted(df["product"].unique().tolist())
            available_regions  = sorted(df["region"].unique().tolist())
            logger.error(
                f"No data found for product='{product}', region='{region}'.\n"
                f"  Available products : {available_products}\n"
                f"  Available regions  : {available_regions}"
            )
            return self
        if len(subset) < 24:
            logger.warning(
                f"Insufficient data for {product}/{region} "
                f"({len(subset)} rows — need ≥ 24 months)."
            )
            return self

        # Prophet expects ds/y
        prophet_df = subset[["date", "price"]].rename(columns={"date": "ds", "price": "y"})
        self.prophet.fit(prophet_df, product, region)

        # LSTM expects 2-D feature array
        feature_cols = [
            "price_norm",
            "month_sin", "month_cos",
            "is_ramadan",
            "pct_change_1m",
        ]
        available = [c for c in feature_cols if c in subset.columns]
        if "price_norm" not in available:
            # Fallback: just use raw price normalised
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            subset = subset.copy()
            subset["price_norm"] = scaler.fit_transform(subset[["price"]])
            available = ["price_norm"]

        series_arr = subset[available].fillna(0).values
        self.lstm.fit(series_arr, product, region)

        key = f"{product}_{region}"
        self._fitted.add(key)
        logger.info(f"Ensemble fitted for {product}/{region}.")
        return self

    def predict(
        self,
        df: pd.DataFrame,
        product: str,
        region: str,
        n_steps: int = 12,
    ) -> pd.DataFrame:
        """Generate an ensemble forecast for the next ``n_steps`` months.

        Args:
            df: Historical DataFrame used for seeding the LSTM.
            product: Product name.
            region: Region name.
            n_steps: Forecast horizon (months).

        Returns:
            DataFrame with columns:
                ds, prophet_yhat, lstm_yhat, ensemble_yhat,
                yhat_lower, yhat_upper.
        """
        key = f"{product}_{region}"
        if key not in self._fitted:
            raise RuntimeError(f"Ensemble not fitted for '{key}'. Call fit() first.")

        # Prophet forecast
        prophet_fc = self.prophet.predict(product, region, periods=n_steps)
        prophet_fc = prophet_fc.tail(n_steps).reset_index(drop=True)

        # LSTM forecast
        subset = df[(df["product"] == product) & (df["region"] == region)].sort_values("date")
        feature_cols = [c for c in ["price_norm", "month_sin", "month_cos", "is_ramadan", "pct_change_1m"] if c in subset.columns]
        if not feature_cols:
            feature_cols = ["price"]
        series_arr = subset[feature_cols].fillna(0).values

        try:
            lstm_preds_norm = self.lstm.predict(series_arr, product, region, n_steps=n_steps)
            # Rough inverse: scale by mean/std of historical prices
            price_mean = subset["price"].mean()
            price_std = subset["price"].std() or 1
            lstm_preds = lstm_preds_norm * price_std + price_mean
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"LSTM predict failed for {product}/{region}: {exc}. Using Prophet only.")
            lstm_preds = prophet_fc["yhat"].values

        # Ensemble
        ensemble_yhat = (
            self.prophet_weight * prophet_fc["yhat"].values
            + self.lstm_weight * lstm_preds[: len(prophet_fc)]
        )

        result = pd.DataFrame(
            {
                "ds": prophet_fc["ds"],
                "prophet_yhat": prophet_fc["yhat"].values,
                "lstm_yhat": lstm_preds[: len(prophet_fc)],
                "ensemble_yhat": ensemble_yhat,
                "yhat_lower": prophet_fc["yhat_lower"].values,
                "yhat_upper": prophet_fc["yhat_upper"].values,
                "product": product,
                "region": region,
            }
        )
        return result

    def evaluate(
        self, df: pd.DataFrame, product: str, region: str
    ) -> dict:
        """Evaluate ensemble on a held-out 20% test set.

        Args:
            df: Full historical DataFrame.
            product: Product name.
            region: Region name.

        Returns:
            Dict with MAE, RMSE, MAPE for each sub-model + ensemble.
        """
        subset = df[(df["product"] == product) & (df["region"] == region)].sort_values("date")
        n = len(subset)
        train = subset.iloc[: int(n * 0.8)]
        test = subset.iloc[int(n * 0.8) :]

        self.fit(train, product, region)
        forecast = self.predict(train, product, region, n_steps=len(test))

        y_true = test["price"].values
        metrics = {}
        for col, label in [
            ("prophet_yhat", "Prophet"),
            ("lstm_yhat", "LSTM"),
            ("ensemble_yhat", "Ensemble"),
        ]:
            y_pred = forecast[col].values[: len(y_true)]
            metrics[label] = compute_metrics(y_true, y_pred)
        return metrics

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, name: str = "ensemble_predictor") -> Path:
        """Save the ensemble state (Prophet models + LSTM weights).

        Args:
            name: Base filename.

        Returns:
            Path to saved file.
        """
        out = self._models_path / f"{name}.joblib"
        joblib.dump(
            {
                "prophet_models": self.prophet._models,
                "lstm_scalers": self.lstm._scalers,
                "fitted": self._fitted,
                "prophet_weight": self.prophet_weight,
                "lstm_weight": self.lstm_weight,
            },
            out,
        )
        # Save LSTM Keras models separately
        for key, model in self.lstm._models.items():
            model_dir = self._models_path / f"lstm_{key}"
            model.save(str(model_dir))

        logger.info(f"Ensemble predictor saved to {self._models_path}")
        return out
