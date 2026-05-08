# job_market_connector.py
# TODO: implement
# ingestion/job_market_connector.py
# Loads job market demand data from files or live APIs (e.g. Adzuna, ONET, Lightcast)

import os
import json
import logging
import requests
import pandas as pd
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobMarketConnector:
    """
    Loads job market data including:
    - Job role demand by region
    - Required skills per job category
    - Salary ranges
    - Industry growth trends
    - Time-to-fill metrics (employer difficulty)

    Supports:
    - Local file loading (CSV/JSON)
    - REST API integration (Adzuna, ONET, or custom endpoint)
    """

    SUPPORTED_EXTENSIONS = [".csv", ".json"]

    def __init__(
        self,
        source_path: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        region_filter: Optional[str] = None
    ):
        """
        Args:
            source_path:   Path to a local job market CSV/JSON file
            api_url:       Base URL of the job market API
            api_key:       API key for authentication
            region_filter: Optional region to filter results
        """
        self.source_path = source_path
        self.api_url = api_url
        self.api_key = api_key
        self.region_filter = region_filter
        self.data = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_file(self) -> pd.DataFrame:
        """Load job market data from a local CSV or JSON file."""
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
            self.data = pd.DataFrame(raw if isinstance(raw, list) else raw.get("results", []))
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.info(f"Loaded {len(self.data)} job market records from file.")
        return self.data

    def load_from_api(
        self,
        endpoint: str = "/jobs",
        params: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Fetch job market data from a REST API.

        Args:
            endpoint: API endpoint path
            params:   Query parameters (e.g. {"region": "Karnataka", "limit": 500})
        """
        if not self.api_url:
            raise ValueError("api_url not set.")

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        params = params or {}
        if self.region_filter:
            params["region"] = self.region_filter

        try:
            response = requests.get(
                self.api_url.rstrip("/") + endpoint,
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            raw = response.json()

            results = raw if isinstance(raw, list) else raw.get("results", raw.get("data", []))
            self.data = pd.DataFrame(results)
            logger.info(f"Fetched {len(self.data)} job records from API.")
            return self.data

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean(self) -> pd.DataFrame:
        """Standardize and clean job market data."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load_from_file() or load_from_api() first.")

        df = self.data.copy()

        # Normalize string columns
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda c: c.str.strip().str.lower())

        # Normalize date columns if present
        for col in ["posted_date", "expiry_date", "updated_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Normalize salary columns
        for col in ["salary_min", "salary_max"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "salary_min" in df.columns and "salary_max" in df.columns:
            df["salary_midpoint"] = (df["salary_min"] + df["salary_max"]) / 2

        # Drop complete duplicates
        df.drop_duplicates(inplace=True)

        logger.info(f"Cleaned job market data shape: {df.shape}")
        self.data = df
        return df

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_top_roles(self, n: int = 10) -> pd.DataFrame:
        """Return the most in-demand job roles."""
        if self.data is None:
            raise RuntimeError("Data not loaded.")
        if "job_title" not in self.data.columns:
            raise ValueError("Column 'job_title' not found.")
        return (
            self.data["job_title"]
            .value_counts()
            .head(n)
            .reset_index()
            .rename(columns={"index": "job_title", "job_title": "count"})
        )

    def get_top_skills_demanded(self, n: int = 15) -> pd.DataFrame:
        """
        Return the most frequently demanded skills across all job postings.
        Expects a 'required_skills' column (comma-separated string or list).
        """
        if self.data is None:
            raise RuntimeError("Data not loaded.")
        if "required_skills" not in self.data.columns:
            raise ValueError("Column 'required_skills' not found.")

        from collections import Counter
        all_skills = []
        for entry in self.data["required_skills"].dropna():
            if isinstance(entry, str):
                all_skills.extend([s.strip() for s in entry.split(",")])
            elif isinstance(entry, list):
                all_skills.extend(entry)

        freq = Counter(all_skills)
        return pd.DataFrame(
            freq.most_common(n), columns=["skill", "demand_count"]
        )

    def get_demand_by_region(self) -> pd.DataFrame:
        """Return job demand count grouped by region."""
        if self.data is None:
            raise RuntimeError("Data not loaded.")
        if "region" not in self.data.columns:
            raise ValueError("Column 'region' not found.")
        return (
            self.data.groupby("region")
            .size()
            .reset_index(name="job_count")
            .sort_values("job_count", ascending=False)
        )

    def get_salary_by_role(self) -> pd.DataFrame:
        """Return average salary midpoint grouped by job role."""
        if self.data is None:
            raise RuntimeError("Data not loaded.")
        if "salary_midpoint" not in self.data.columns:
            raise ValueError("salary_midpoint not computed. Ensure salary_min and salary_max exist.")
        return (
            self.data.groupby("job_title")["salary_midpoint"]
            .mean()
            .reset_index()
            .sort_values("salary_midpoint", ascending=False)
        )

    def get_summary(self) -> dict:
        """Return summary statistics of the job market dataset."""
        if self.data is None:
            raise RuntimeError("Data not loaded.")
        summary = {
            "total_postings": len(self.data),
            "unique_roles": self.data["job_title"].nunique() if "job_title" in self.data.columns else "N/A",
            "regions": self.data["region"].unique().tolist() if "region" in self.data.columns else [],
        }
        if "salary_midpoint" in self.data.columns:
            summary["avg_salary_midpoint"] = round(self.data["salary_midpoint"].mean(), 2)
        return summary

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_processed(self, output_path: str) -> None:
        """Save cleaned job market data to CSV."""
        if self.data is None:
            raise RuntimeError("No data to save.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.data.to_csv(output_path, index=False)
        logger.info(f"Saved job market data to {output_path}")


if __name__ == "__main__":
    connector = JobMarketConnector(source_path="data/raw/job_market.csv")
    connector.load_from_file()
    connector.clean()
    print(connector.get_summary())
    print(connector.get_top_skills_demanded())
    print(connector.get_demand_by_region())
    connector.save_processed("data/processed/job_market_clean.csv")