# economic_indicators.py
# TODO: implement
# ingestion/economic_indicators.py
# Loads macroeconomic context data: unemployment rates, GDP, inflation,
# sector growth, and regional economic health indicators

import os
import json
import logging
import requests
import pandas as pd
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EconomicIndicatorsLoader:
    """
    Loads and aligns macroeconomic indicators with the participant timeline.

    Indicators include:
    - Regional unemployment rate (monthly)
    - GDP growth rate (quarterly)
    - Inflation / CPI index
    - Sector-level employment growth
    - Job vacancy rate by region
    - Minimum wage trends

    Sources supported:
    - Local CSV/JSON file
    - World Bank API
    - Government open data APIs (e.g. data.gov.in, BLS)
    """

    SUPPORTED_EXTENSIONS = [".csv", ".json"]

    INDICATOR_COLUMNS = [
        "region",
        "date",
        "unemployment_rate",
        "gdp_growth_rate",
        "inflation_rate",
        "sector",
        "sector_employment_growth",
        "job_vacancy_rate"
    ]

    def __init__(
        self,
        source_path: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.source_path = source_path
        self.api_url = api_url
        self.api_key = api_key
        self.data = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_file(self) -> pd.DataFrame:
        """Load economic indicators from a local file."""
        if not self.source_path:
            raise ValueError("source_path not set.")
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"File not found: {self.source_path}")

        ext = os.path.splitext(self.source_path)[-1].lower()

        if ext == ".csv":
            self.data = pd.read_csv(self.source_path)
        elif ext == ".json":
            with open(self.source_path, "r") as f:
                raw = json.load(f)
            self.data = pd.DataFrame(raw if isinstance(raw, list) else raw.get("data", []))
        else:
            raise ValueError(f"Unsupported format: {ext}")

        logger.info(f"Loaded {len(self.data)} economic indicator records.")
        return self.data

    def load_from_world_bank(
        self,
        country_code: str = "IN",
        indicator: str = "SL.UEM.TOTL.ZS",
        start_year: int = 2015,
        end_year: int = 2024
    ) -> pd.DataFrame:
        """
        Fetch data from the World Bank Open Data API.

        Args:
            country_code: ISO 2-letter country code (e.g. 'IN', 'US')
            indicator:    World Bank indicator code
                          SL.UEM.TOTL.ZS = Unemployment rate
                          NY.GDP.MKTP.KD.ZG = GDP growth
                          FP.CPI.TOTL.ZG = Inflation
            start_year:   Start of time range
            end_year:     End of time range
        """
        url = (
            f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
            f"?format=json&date={start_year}:{end_year}&per_page=100"
        )

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            raw = response.json()

            if len(raw) < 2 or not raw[1]:
                logger.warning("World Bank API returned no data.")
                return pd.DataFrame()

            records = []
            for entry in raw[1]:
                records.append({
                    "region": entry.get("country", {}).get("value", country_code),
                    "date": entry.get("date"),
                    "indicator": indicator,
                    "value": entry.get("value")
                })

            self.data = pd.DataFrame(records)
            logger.info(f"Fetched {len(self.data)} records from World Bank API.")
            return self.data

        except requests.exceptions.RequestException as e:
            logger.error(f"World Bank API request failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean(self) -> pd.DataFrame:
        """
        Clean and normalize economic indicator data:
        - Parse date column to datetime
        - Normalize region and sector names
        - Fill short gaps in time series (forward fill)
        - Clip extreme outlier values
        """
        if self.data is None:
            raise RuntimeError("Data not loaded.")

        df = self.data.copy()

        # Parse date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Normalize string fields
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda c: c.str.strip().str.lower())

        # Sort by region and date
        sort_cols = [c for c in ["region", "date"] if c in df.columns]
        df.sort_values(sort_cols, inplace=True)

        # Forward-fill short gaps in numeric columns (max 2 periods)
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(method="ffill", limit=2)

        # Clip unemployment and inflation to sane ranges
        if "unemployment_rate" in df.columns:
            df["unemployment_rate"] = df["unemployment_rate"].clip(0, 100)
        if "inflation_rate" in df.columns:
            df["inflation_rate"] = df["inflation_rate"].clip(-20, 100)

        logger.info(f"Cleaned economic data shape: {df.shape}")
        self.data = df
        return df

    # ------------------------------------------------------------------
    # Alignment with Participant Data
    # ------------------------------------------------------------------

    def align_with_participant_data(
        self,
        participant_df: pd.DataFrame,
        date_col: str = "time_period",
        region_col: str = "region"
    ) -> pd.DataFrame:
        """
        Left-join economic indicators onto the participant timeline DataFrame
        by matching region and time period.

        Args:
            participant_df: Unified participant temporal DataFrame
            date_col:       Date/period column in participant_df
            region_col:     Region column in participant_df

        Returns:
            Merged DataFrame with economic context appended
        """
        if self.data is None:
            raise RuntimeError("Economic data not loaded.")

        econ = self.data.copy()

        # Ensure both date columns are the same type
        participant_df = participant_df.copy()
        participant_df[date_col] = pd.to_datetime(
            participant_df[date_col].astype(str), errors="coerce"
        )
        econ["date"] = pd.to_datetime(econ["date"], errors="coerce")

        # Round participant dates to month for join
        participant_df["_join_date"] = participant_df[date_col].dt.to_period("M")
        econ["_join_date"] = econ["date"].dt.to_period("M")

        merged = participant_df.merge(
            econ.drop(columns=["date"], errors="ignore"),
            left_on=["_join_date", region_col],
            right_on=["_join_date", "region"],
            how="left"
        ).drop(columns=["_join_date"])

        logger.info(
            f"Aligned economic indicators. "
            f"Merged shape: {merged.shape}"
        )
        return merged

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_regional_summary(self) -> pd.DataFrame:
        """Return average indicator values grouped by region."""
        if self.data is None:
            raise RuntimeError("Data not loaded.")
        if "region" not in self.data.columns:
            raise ValueError("'region' column not found.")

        numeric_cols = self.data.select_dtypes(include="number").columns.tolist()
        return (
            self.data.groupby("region")[numeric_cols]
            .mean()
            .round(3)
            .reset_index()
        )

    def get_trend(self, region: str, indicator_col: str) -> pd.DataFrame:
        """
        Return the time series of a specific indicator for a given region.

        Args:
            region:        Region name (lowercase)
            indicator_col: Column name of the indicator

        Returns:
            DataFrame with date and indicator value
        """
        if self.data is None:
            raise RuntimeError("Data not loaded.")
        if indicator_col not in self.data.columns:
            raise ValueError(f"Column '{indicator_col}' not found.")

        filtered = self.data[
            self.data["region"] == region.lower()
        ][["date", indicator_col]].dropna()

        return filtered.sort_values("date").reset_index(drop=True)

    def get_summary(self) -> dict:
        """Return summary statistics of the economic indicators dataset."""
        if self.data is None:
            raise RuntimeError("Data not loaded.")

        summary = {
            "total_records": len(self.data),
            "regions": self.data["region"].unique().tolist() if "region" in self.data.columns else [],
            "date_range": {
                "start": str(self.data["date"].min()) if "date" in self.data.columns else "N/A",
                "end": str(self.data["date"].max()) if "date" in self.data.columns else "N/A"
            }
        }
        for col in ["unemployment_rate", "gdp_growth_rate", "inflation_rate"]:
            if col in self.data.columns:
                summary[f"avg_{col}"] = round(self.data[col].mean(), 3)
        return summary

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_processed(self, output_path: str) -> None:
        """Save cleaned economic indicator data to CSV."""
        if self.data is None:
            raise RuntimeError("No data to save.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.data.to_csv(output_path, index=False)
        logger.info(f"Saved economic indicators to {output_path}")


if __name__ == "__main__":
    # Example: load from file
    loader = EconomicIndicatorsLoader(source_path="data/raw/economic_indicators.csv")
    loader.load_from_file()
    loader.clean()
    print(loader.get_summary())
    print(loader.get_regional_summary())
    loader.save_processed("data/processed/economic_indicators_clean.csv")

    # Example: load from World Bank API
    # wb_loader = EconomicIndicatorsLoader()
    # wb_loader.load_from_world_bank(country_code="IN", indicator="SL.UEM.TOTL.ZS")
    # wb_loader.clean()
    # wb_loader.save_processed("data/processed/wb_unemployment.csv")