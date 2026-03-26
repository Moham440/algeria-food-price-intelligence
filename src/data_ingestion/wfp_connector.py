"""
WFP (World Food Programme) Data Connector.

Downloads and parses food price market data for Algeria from the
HDX / VAM platforms maintained by WFP.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from src.utils.helpers import load_config, retry, save_parquet


class WFPConnector:
    """Client for WFP / HDX Algeria food price datasets.

    The primary source is the Humanitarian Data Exchange (HDX) which hosts
    WFP's VAM price monitoring CSV files.

    Attributes:
        hdx_base_url: Base URL for the HDX API.
        dataset_id: HDX dataset identifier for Algeria WFP prices.
        raw_data_path: Local directory for raw Parquet files.
    """

    # HDX resource URLs for Algeria WFP food prices
    _RESOURCE_URLS = [
        "https://data.humdata.org/dataset/wfp-food-prices-for-algeria/resource/"
        "9dc4a3c1-5d76-4e16-b6ad-d3e3b0a0e4b0",
        # Fallback: VAM direct API
        "https://api.vam.wfp.org/api/GetCommodityPrice?ac=DZ",
    ]

    # Column mapping from WFP CSV headers
    _COLUMN_MAP = {
        "date": "date",
        "admin1": "region",
        "admin2": "district",
        "market": "market",
        "latitude": "latitude",
        "longitude": "longitude",
        "category": "category",
        "commodity": "product",
        "unit": "unit",
        "priceflag": "price_flag",
        "pricetype": "price_type",
        "currency": "currency",
        "price": "price",
        "usdprice": "price_usd",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_config()

        wfp_cfg = config["data_sources"]["wfp"]
        self.hdx_base_url: str = wfp_cfg["base_url"]
        self.dataset_id: str = wfp_cfg["dataset_id"]
        self.timeout: int = wfp_cfg.get("timeout", 30)
        self.raw_data_path = Path(config["storage"]["raw_data_path"]) / "wfp"
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "AlgeriaFoodPriceIntelligence/1.0"})

    # ── Internal ──────────────────────────────────────────────────────────────

    @retry(max_attempts=3, delay=2.0)
    def _download_csv(self, url: str) -> pd.DataFrame:
        """Download a CSV file from a URL and return as a DataFrame.

        Args:
            url: Direct download URL.

        Returns:
            Raw DataFrame.

        Raises:
            requests.HTTPError: On non-2xx status codes.
        """
        logger.debug(f"Downloading: {url}")
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))

    def _fetch_hdx_resource_url(self) -> Optional[str]:
        """Query the HDX API to find the latest CSV resource URL.

        Returns:
            Direct download URL as a string, or None on failure.
        """
        api_url = f"{self.hdx_base_url}/api/3/action/package_show"
        try:
            response = self._session.get(
                api_url, params={"id": self.dataset_id}, timeout=self.timeout
            )
            response.raise_for_status()
            pkg = response.json()
            resources = pkg.get("result", {}).get("resources", [])
            for r in resources:
                if r.get("format", "").upper() == "CSV":
                    return r["url"]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"HDX API query failed: {exc}")
        return None

    # ── Public Methods ────────────────────────────────────────────────────────

    def fetch_price_data(self) -> pd.DataFrame:
        """Fetch Algeria food price market data from WFP/HDX.

        Merges real WFP data (southern wilayas monitored by WFP) with
        synthetic data covering the major northern wilayas, giving full
        national coverage in the dashboard.

        Returns:
            Cleaned DataFrame with standardised column names.
        """
        logger.info("Fetching WFP Algeria food price data…")

        real_df = pd.DataFrame()

        # 1. Try to get fresh download URL from HDX
        csv_url = self._fetch_hdx_resource_url()
        if csv_url:
            try:
                raw = self._download_csv(csv_url)
                real_df = self._clean(raw)
                logger.info(f"WFP HDX real data: {len(real_df):,} records "
                            f"| regions: {sorted(real_df['region'].unique().tolist()) if 'region' in real_df.columns else '?'}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"HDX CSV download failed: {exc}")

        # 2. Always add synthetic data for full national coverage
        synthetic_df = self._synthetic_wfp_data()
        logger.info(f"Synthetic national data: {len(synthetic_df):,} records")

        # 3. Merge: real data takes priority, synthetic fills the rest
        if not real_df.empty:
            # Tag sources
            real_df["source"] = "WFP_real"
            synthetic_df["source"] = "synthetic"
            # Remove synthetic rows whose region already exists in real data
            real_regions = real_df["region"].unique() if "region" in real_df.columns else []
            synthetic_df = synthetic_df[~synthetic_df["region"].isin(real_regions)]
            combined = pd.concat([real_df, synthetic_df], ignore_index=True)
        else:
            synthetic_df["source"] = "synthetic"
            combined = synthetic_df

        combined = combined.sort_values("date").reset_index(drop=True)
        logger.info(
            f"Combined dataset: {len(combined):,} records | "
            f"regions: {sorted(combined['region'].unique().tolist())}"
        )
        return combined

    def fetch_market_locations(self) -> pd.DataFrame:
        """Return a DataFrame of market locations with coordinates.

        Returns:
            DataFrame with columns: market, region, latitude, longitude.
        """
        df = self.fetch_price_data()
        if df.empty:
            return pd.DataFrame()
        cols = ["market", "region", "latitude", "longitude"]
        available = [c for c in cols if c in df.columns]
        return df[available].drop_duplicates(subset=["market"]).reset_index(drop=True)

    # ── Cleaning ──────────────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names, types, and filter Algeria records.

        Args:
            df: Raw DataFrame from WFP CSV.

        Returns:
            Cleaned and typed DataFrame.
        """
        # Lowercase headers
        df.columns = df.columns.str.lower().str.strip()

        # Rename to standard names
        df = df.rename(columns=self._COLUMN_MAP)

        # Remove duplicate columns (keep first occurrence of each name)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        # Keep only relevant columns — use dict.fromkeys to deduplicate values
        desired = list(dict.fromkeys(self._COLUMN_MAP.values()))
        keep = [c for c in desired if c in df.columns]
        df = df[keep].copy()

        # If "market" is missing but "district" exists, use district as market
        if "market" not in df.columns and "district" in df.columns:
            df = df.rename(columns={"district": "market"})

        # Type conversions
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        if "price_usd" in df.columns:
            df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")

        # Drop rows with missing price or date
        df.dropna(subset=["price", "date"], inplace=True)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ── Synthetic Data ────────────────────────────────────────────────────────

    def _synthetic_wfp_data(self) -> pd.DataFrame:
        """Generate realistic synthetic WFP-style monthly price observations.

        Covers 2015-2024, major Algerian markets and food commodities.

        Returns:
            DataFrame with standardised WFP columns.
        """
        import numpy as np

        markets = {
            "Alger - Bachdjarah": ("Alger", 36.71, 3.11),
            "Oran - Médina Jedida": ("Oran", 35.70, -0.62),
            "Constantine - Centre": ("Constantine", 36.37, 6.61),
            "Annaba - Centre": ("Annaba", 36.90, 7.77),
            "Tlemcen - Centre": ("Tlemcen", 34.88, -1.32),
            "Sétif - Centrale": ("Sétif", 36.18, 5.40),
            "Batna - Centre": ("Batna", 35.56, 6.17),
        }

        products = {
            "Tomatoes": ("Vegetables", "kg", 70, 0.18),
            "Potatoes": ("Vegetables", "kg", 50, 0.14),
            "Onions": ("Vegetables", "kg", 55, 0.20),
            "Chicken": ("Meat and Fish", "kg", 340, 0.11),
            "Mutton": ("Meat and Fish", "kg", 1150, 0.09),
            "Bread": ("Cereals and Tubers", "kg", 28, 0.05),
            "Semolina": ("Cereals and Tubers", "kg", 65, 0.06),
            "Vegetable Oil": ("Oil and Fats", "litre", 210, 0.13),
            "Sugar": ("Miscellaneous Food", "kg", 85, 0.07),
            "Eggs": ("Miscellaneous Food", "dozen", 200, 0.12),
        }

        rng = np.random.default_rng(1234)
        dates = pd.date_range("2015-01-01", "2024-06-01", freq="MS")
        rows = []

        for market_name, (region, lat, lon) in markets.items():
            for product, (category, unit, base, vol) in products.items():
                price = base
                for dt in dates:
                    # Annual inflation + seasonal + random noise
                    annual_growth = 1 + rng.normal(0.06, 0.02)
                    seasonal = 1 + 0.10 * np.sin(2 * np.pi * dt.month / 12)
                    noise = rng.normal(1.0, vol)
                    price = max(price * (annual_growth ** (1 / 12)) * noise * seasonal, 5)

                    # Ramadan bump (higher demand)
                    from src.utils.helpers import is_ramadan
                    if is_ramadan(dt):
                        price *= rng.uniform(1.05, 1.25)

                    rows.append(
                        {
                            "date": dt,
                            "region": region,
                            "market": market_name,
                            "latitude": lat,
                            "longitude": lon,
                            "category": category,
                            "product": product,
                            "unit": unit,
                            "price_type": "Retail",
                            "currency": "DZD",
                            "price": round(price, 2),
                            "price_usd": round(price / 135, 3),
                        }
                    )

        df = pd.DataFrame(rows)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info(f"Generated {len(df):,} synthetic WFP records.")
        return df

    # ── Save ──────────────────────────────────────────────────────────────────

    def fetch_and_save(self) -> Path:
        """Fetch WFP data and persist to Parquet.

        Returns:
            Path to the saved Parquet file.
        """
        df = self.fetch_price_data()
        out_path = self.raw_data_path / "wfp_algeria_prices.parquet"
        save_parquet(df, out_path)
        return out_path
