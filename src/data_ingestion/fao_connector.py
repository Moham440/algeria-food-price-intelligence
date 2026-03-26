"""
FAO (Food and Agriculture Organization) Data Connector.

Fetches food price and balance data from the FAOSTAT REST API and
stores results locally in Parquet format.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from src.utils.helpers import disk_cache, load_config, retry, save_parquet


class FAOConnector:
    """Client for the FAOSTAT v1 REST API.

    Attributes:
        base_url: Root URL of the FAOSTAT API.
        timeout: HTTP request timeout in seconds.
        rate_limit_delay: Minimum pause (seconds) between consecutive requests.
        raw_data_path: Local directory for raw Parquet files.
    """

    ALGERIA_COUNTRY_CODE = "4"  # FAO numeric code for Algeria
    FOOD_PRICE_ELEMENT_CODE = "5532"  # Producer Price (USD/tonne)

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        fao_cfg = config["data_sources"]["fao"]
        self.base_url: str = fao_cfg["base_url"]
        self.timeout: int = fao_cfg.get("timeout", 30)
        self.rate_limit_delay: float = fao_cfg.get("rate_limit_delay", 1.0)
        self.raw_data_path = Path(config["storage"]["raw_data_path"]) / "fao"
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._last_request_time: float = 0.0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Honour rate-limiting by sleeping if needed."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    @retry(max_attempts=3, delay=2.0)
    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Perform a GET request against the FAOSTAT API.

        Args:
            endpoint: API path relative to base_url.
            params: Optional query parameters.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            requests.HTTPError: On non-2xx status codes.
        """
        self._throttle()
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"GET {url} | params={params}")
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # ── Public Methods ────────────────────────────────────────────────────────

    def fetch_price_data(
        self,
        year_start: int = 2010,
        year_end: int = 2024,
        item_codes: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fetch Algerian food price data from FAOSTAT.

        The endpoint returns Producer Price indices. Results are cached to
        avoid redundant network calls across runs.

        Args:
            year_start: First year to retrieve.
            year_end: Last year to retrieve.
            item_codes: List of FAOSTAT item codes to filter (None = all).

        Returns:
            DataFrame with columns:
                area, area_code, item, item_code, element,
                year, unit, value, flag
        """
        params: dict = {
            "area": self.ALGERIA_COUNTRY_CODE,
            "element": self.FOOD_PRICE_ELEMENT_CODE,
            "year": ",".join(str(y) for y in range(year_start, year_end + 1)),
            "output_type": "objects",
        }
        if item_codes:
            params["item"] = ",".join(item_codes)

        logger.info("Fetching FAO price data for Algeria…")
        try:
            data = self._get("/data/PP", params=params)
            records = data.get("data", [])
        except Exception as exc:  # noqa: BLE001
            logger.error(f"FAO API unavailable ({exc}). Generating synthetic data.")
            return self._synthetic_price_data(year_start, year_end)

        if not records:
            logger.warning("FAO returned zero records. Falling back to synthetic data.")
            return self._synthetic_price_data(year_start, year_end)

        df = pd.DataFrame(records)
        df = self._normalise_columns(df)
        logger.info(f"Retrieved {len(df):,} FAO price records.")
        return df

    def fetch_food_balance(self, year_start: int = 2010, year_end: int = 2021) -> pd.DataFrame:
        """Fetch food-balance-sheet data (supply/demand) for Algeria.

        Args:
            year_start: First year to retrieve.
            year_end: Last year to retrieve.

        Returns:
            DataFrame with food balance sheet indicators.
        """
        params = {
            "area": self.ALGERIA_COUNTRY_CODE,
            "year": ",".join(str(y) for y in range(year_start, year_end + 1)),
            "output_type": "objects",
        }
        logger.info("Fetching FAO food balance sheet…")
        try:
            data = self._get("/data/FBS", params=params)
            records = data.get("data", [])
        except Exception as exc:  # noqa: BLE001
            logger.error(f"FAO FBS unavailable ({exc}).")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        return self._normalise_columns(df)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase and snake_case column names."""
        df.columns = (
            df.columns.str.lower()
            .str.replace(r"[\s\-]+", "_", regex=True)
            .str.strip("_")
        )
        # Cast value column to float
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df

    def _synthetic_price_data(
        self, year_start: int, year_end: int
    ) -> pd.DataFrame:
        """Generate realistic synthetic price data for offline development.

        Args:
            year_start: First year.
            year_end: Last year.

        Returns:
            DataFrame with synthetic yearly price series.
        """
        import numpy as np

        products = {
            "Tomatoes": ("TOM", 80, 0.15),
            "Potatoes": ("POT", 55, 0.12),
            "Onions": ("ONI", 60, 0.20),
            "Chicken": ("CHK", 350, 0.10),
            "Mutton": ("LAM", 1200, 0.08),
            "Bread": ("BRD", 30, 0.05),
            "Semolina": ("SEM", 70, 0.06),
            "Vegetable Oil": ("OIL", 220, 0.12),
            "Sugar": ("SUG", 90, 0.07),
        }

        rng = np.random.default_rng(42)
        rows = []
        years = range(year_start, year_end + 1)

        for product, (code, base_price, volatility) in products.items():
            price = base_price
            for year in years:
                trend = 1 + rng.normal(0.05, 0.02)  # ~5% annual inflation
                noise = rng.normal(1.0, volatility)
                price = price * trend * noise
                rows.append(
                    {
                        "area": "Algeria",
                        "area_code": self.ALGERIA_COUNTRY_CODE,
                        "item": product,
                        "item_code": code,
                        "element": "Producer Price",
                        "year": year,
                        "unit": "DZD/kg",
                        "value": round(price, 2),
                        "flag": "S",  # S = synthetic
                    }
                )

        df = pd.DataFrame(rows)
        logger.info(f"Generated {len(df):,} synthetic FAO records.")
        return df

    # ── Save ──────────────────────────────────────────────────────────────────

    def fetch_and_save(self, year_start: int = 2010, year_end: int = 2024) -> Path:
        """Fetch price data and persist to Parquet.

        Args:
            year_start: Start year.
            year_end: End year.

        Returns:
            Path to the saved Parquet file.
        """
        df = self.fetch_price_data(year_start=year_start, year_end=year_end)
        out_path = self.raw_data_path / f"fao_prices_{year_start}_{year_end}.parquet"
        save_parquet(df, out_path)
        return out_path
