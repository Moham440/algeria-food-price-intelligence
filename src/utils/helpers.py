"""
Utility helpers for Algeria Food Price Intelligence System.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger


# ── Configuration ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary with configuration values.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: Optional[dict] = None) -> None:
    """Configure loguru logger from config dict.

    Args:
        config: Logging section of the config file.
    """
    if config is None:
        config = {}
    log_file = config.get("log_file", "logs/app.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level=config.get("level", "INFO"),
        format=config.get(
            "format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
        ),
        rotation=config.get("rotation", "10 MB"),
        retention=config.get("retention", "1 week"),
    )


# ── Caching ───────────────────────────────────────────────────────────────────

def make_cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate a deterministic cache key from arguments.

    Returns:
        MD5 hex-digest string.
    """
    raw = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def disk_cache(cache_dir: str = ".cache", ttl: int = 3600):
    """Decorator that caches the return value of a function to disk.

    Args:
        cache_dir: Directory to store cached files.
        ttl: Time-to-live in seconds.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            key = make_cache_key(func.__name__, *args, **kwargs)
            cache_file = Path(cache_dir) / f"{key}.json"

            if cache_file.exists():
                age = time.time() - cache_file.stat().st_mtime
                if age < ttl:
                    logger.debug(f"Cache hit for {func.__name__} (key={key[:8]})")
                    with open(cache_file) as f:
                        return json.load(f)

            result = func(*args, **kwargs)
            with open(cache_file, "w") as f:
                json.dump(result, f, default=str)
            logger.debug(f"Cache stored for {func.__name__} (key={key[:8]})")
            return result
        return wrapper
    return decorator


# ── Retry Logic ───────────────────────────────────────────────────────────────

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator that retries a function on failure with exponential back-off.

    Args:
        max_attempts: Maximum number of attempts.
        delay: Initial delay in seconds.
        backoff: Multiplier applied to delay after each failure.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {exc}"
                        )
                        raise
                    logger.warning(
                        f"{func.__name__} attempt {attempt} failed: {exc}. "
                        f"Retrying in {current_delay:.1f}s…"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


# ── Date Helpers ──────────────────────────────────────────────────────────────

# Approximate Ramadan dates for 2020-2030 (start date of the holy month)
_RAMADAN_DATES: dict[int, tuple[date, date]] = {
    2020: (date(2020, 4, 23), date(2020, 5, 23)),
    2021: (date(2021, 4, 12), date(2021, 5, 12)),
    2022: (date(2022, 4, 1), date(2022, 5, 1)),
    2023: (date(2023, 3, 22), date(2023, 4, 21)),
    2024: (date(2024, 3, 10), date(2024, 4, 9)),
    2025: (date(2025, 2, 28), date(2025, 3, 30)),
    2026: (date(2026, 2, 17), date(2026, 3, 19)),
    2027: (date(2027, 2, 6), date(2027, 3, 8)),
    2028: (date(2028, 1, 26), date(2028, 2, 24)),
    2029: (date(2029, 1, 14), date(2029, 2, 13)),
    2030: (date(2030, 1, 3), date(2030, 2, 2)),
}

_ALGERIAN_HOLIDAYS: list[tuple[int, int]] = [
    (1, 1),   # New Year
    (5, 1),   # Labour Day
    (7, 5),   # Amazigh New Year
    (7, 19),  # July 5 Independence Day
    (11, 1),  # Revolution Day
]


def is_ramadan(dt: Union[date, datetime]) -> bool:
    """Return True if the given date falls within Ramadan.

    Args:
        dt: A date or datetime object.

    Returns:
        Boolean flag.
    """
    d = dt.date() if isinstance(dt, datetime) else dt
    year = d.year
    if year in _RAMADAN_DATES:
        start, end = _RAMADAN_DATES[year]
        return start <= d <= end
    return False


def is_algerian_holiday(dt: Union[date, datetime]) -> bool:
    """Return True if the date is an Algerian public holiday.

    Args:
        dt: A date or datetime object.

    Returns:
        Boolean flag.
    """
    d = dt.date() if isinstance(dt, datetime) else dt
    return (d.month, d.day) in _ALGERIAN_HOLIDAYS


def get_season(dt: Union[date, datetime]) -> str:
    """Return Northern-Hemisphere season name for the given date.

    Args:
        dt: A date or datetime object.

    Returns:
        One of 'Winter', 'Spring', 'Summer', 'Autumn'.
    """
    d = dt.date() if isinstance(dt, datetime) else dt
    month = d.month
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Enrich a DataFrame with calendar and Islamic calendar features.

    Args:
        df: Input DataFrame with a date-like column.
        date_col: Name of the date column.

    Returns:
        DataFrame with additional temporal columns.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["week"] = df[date_col].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["quarter"] = df[date_col].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([4, 5]).astype(int)  # Fri/Sat in Algeria
    df["season"] = df[date_col].apply(get_season)
    df["is_ramadan"] = df[date_col].apply(is_ramadan).astype(int)
    df["is_holiday"] = df[date_col].apply(is_algerian_holiday).astype(int)

    # Cyclical encoding for month and day_of_week
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


# ── DataFrame Helpers ─────────────────────────────────────────────────────────

def safe_read_parquet(path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
    """Read a Parquet file safely; return None if the file is missing.

    Args:
        path: Path to the Parquet file.
        **kwargs: Additional kwargs forwarded to pd.read_parquet.

    Returns:
        DataFrame or None.
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Parquet file not found: {path}")
        return None
    try:
        return pd.read_parquet(path, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to read {path}: {exc}")
        return None


def save_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """Save a DataFrame to Parquet, creating parent directories as needed.

    Args:
        df: DataFrame to save.
        path: Destination path.
        **kwargs: Additional kwargs forwarded to DataFrame.to_parquet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)
    logger.info(f"Saved {len(df):,} rows → {path}")


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    min_rows: int = 1,
) -> bool:
    """Validate that a DataFrame meets basic requirements.

    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must be present.
        min_rows: Minimum number of rows required.

    Returns:
        True if valid, raises ValueError otherwise.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty.")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, expected at least {min_rows}.")
    return True


# ── Metric Helpers ────────────────────────────────────────────────────────────

def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Compute MAPE, ignoring zero actual values.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        MAPE as a percentage.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute a standard set of regression metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dict with MAE, RMSE, and MAPE keys.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


# ── Formatting ────────────────────────────────────────────────────────────────

def format_price(price: float, currency: str = "DZD") -> str:
    """Format a numeric price value as a human-readable string.

    Args:
        price: Numeric price.
        currency: Currency code suffix.

    Returns:
        Formatted string, e.g. '1 234.50 DZD'.
    """
    return f"{price:,.2f} {currency}"


def pct_change_label(change: float) -> str:
    """Return a coloured label string for a percentage change.

    Args:
        change: Percentage change value.

    Returns:
        String like '+5.2%' or '-3.1%'.
    """
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.1f}%"
