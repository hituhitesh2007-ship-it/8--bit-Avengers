# ingestion/community_scraper.py
# Loads and parses community observation data from field reports and surveys

import pandas as pd
import json
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommunityScraper:
    """
    Loads community observation data including:
    - Field reports from program coordinators
    - Participant self-reported surveys
    - Mentor/observer notes
    - Community engagement records
    - Local opportunity availability logs
    """

    REQUIRED_COLUMNS = [
        "participant_id",
        "observation_date",
        "observer_type",
        "region",
        "observation_text",
        "engagement_score"
    ]

    VALID_OBSERVER_TYPES = [
        "coordinator",
        "mentor",
        "self",
        "peer",
        "employer",
        "unknown"
    ]

    VALID_FORMATS = [".csv", ".xlsx", ".json"]

    def __init__(self, source_path: str):
        """
        Args:
            source_path: Path to community observation file
                         (.csv, .xlsx, or .json)
        """
        self.source_path = source_path
        self.data = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Load community observation data from file."""
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Source file not found: {self.source_path}")

        ext = os.path.splitext(self.source_path)[-1].lower()

        if ext not in self.VALID_FORMATS:
            raise ValueError(f"Unsupported format: {ext}. Use one of {self.VALID_FORMATS}")

        if ext == ".csv":
            self.data = pd.read_csv(self.source_path)

        elif ext in (".xlsx", ".xls"):
            self.data = pd.read_excel(self.source_path)

        elif ext == ".json":
            self.data = self._load_json()

        logger.info(f"Loaded {len(self.data)} community observation records.")
        return self.data

    def _load_json(self) -> pd.DataFrame:
        """
        Handle JSON input which may be:
        - A flat list of observation objects
        - A nested dict with a records key
        """
        with open(self.source_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            return pd.DataFrame(raw)

        # Try common nested keys
        for key in ["records", "observations", "data", "results"]:
            if key in raw:
                return pd.DataFrame(raw[key])

        raise ValueError(
            "JSON structure not recognized. "
            "Expected a list or a dict with a 'records'/'observations'/'data' key."
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Check required columns and observer type values."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        # Check required columns
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.data.columns]
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            return False

        # Check observer types
        invalid_types = self.data[
            ~self.data["observer_type"].str.lower().isin(self.VALID_OBSERVER_TYPES)
        ]
        if not invalid_types.empty:
            logger.warning(
                f"Unrecognized observer types: {invalid_types['observer_type'].unique()}"
            )

        # Check for empty observation text
        empty_text = self.data["observation_text"].isna().sum()
        if empty_text > 0:
            logger.warning(f"{empty_text} records have missing observation text.")

        logger.info("Validation complete.")
        return True

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean(self) -> pd.DataFrame:
        """
        Cleaning steps:
        - Normalize dates
        - Standardize observer types and regions
        - Strip and lowercase text fields
        - Drop duplicates
        - Clip engagement scores to valid range [0, 10]
        - Flag empty observation text
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        df = self.data.copy()

        # Drop exact duplicates
        df.drop_duplicates(
            subset=["participant_id", "observation_date", "observer_type"],
            inplace=True
        )

        # Normalize dates
        df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")

        # Normalize string fields
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

        # Lowercase categorical fields
        df["observer_type"] = df["observer_type"].str.lower().fillna("unknown")
        df["region"] = df["region"].str.lower().fillna("unknown")

        # Clip engagement score to [0, 10]
        if "engagement_score" in df.columns:
            df["engagement_score"] = pd.to_numeric(
                df["engagement_score"], errors="coerce"
            ).clip(0, 10)

        # Flag missing observation text
        df["has_text"] = df["observation_text"].notna() & (df["observation_text"].str.strip() != "")

        # Word count for later NLP prioritization
        df["word_count"] = df["observation_text"].fillna("").apply(
            lambda x: len(x.split())
        )

        logger.info(f"Cleaned community data shape: {df.shape}")
        self.data = df
        return df

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_by_participant(self, participant_id: str) -> pd.DataFrame:
        """Return all observations for a specific participant."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        result = self.data[
            self.data["participant_id"] == participant_id
        ].sort_values("observation_date")

        if result.empty:
            logger.warning(f"No observations found for participant: {participant_id}")

        return result

    def get_by_observer_type(self, observer_type: str) -> pd.DataFrame:
        """Return observations filtered by observer type."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        if observer_type.lower() not in self.VALID_OBSERVER_TYPES:
            raise ValueError(
                f"Invalid observer type '{observer_type}'. "
                f"Choose from: {self.VALID_OBSERVER_TYPES}"
            )

        return self.data[self.data["observer_type"] == observer_type.lower()]

    def get_by_region(self, region: str) -> pd.DataFrame:
        """Return observations filtered by region."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        return self.data[self.data["region"] == region.lower()]

    def get_low_engagement(self, threshold: float = 4.0) -> pd.DataFrame:
        """
        Return participants with average engagement score below threshold.
        Useful for identifying at-risk individuals early.
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        avg_engagement = (
            self.data.groupby("participant_id")["engagement_score"]
            .mean()
            .reset_index()
            .rename(columns={"engagement_score": "avg_engagement"})
        )

        low = avg_engagement[avg_engagement["avg_engagement"] < threshold]
        logger.info(
            f"Found {len(low)} participants with avg engagement below {threshold}"
        )
        return low

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return a high-level summary of the community observation dataset."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        return {
            "total_records": len(self.data),
            "unique_participants": self.data["participant_id"].nunique(),
            "observer_type_distribution": (
                self.data["observer_type"].value_counts().to_dict()
            ),
            "regions": self.data["region"].unique().tolist(),
            "avg_engagement_score": round(
                self.data["engagement_score"].dropna().mean(), 2
            ),
            "records_with_text": int(self.data["has_text"].sum()),
            "avg_word_count": round(
                self.data["word_count"].mean(), 1
            ),
            "date_range": {
                "start": str(self.data["observation_date"].min()),
                "end": str(self.data["observation_date"].max())
            }
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_processed(
        self,
        output_path: str,
        format: str = "csv"
    ) -> None:
        """
        Save cleaned data to the processed/ directory.

        Args:
            output_path: Destination file path
            format: 'csv' or 'json'
        """
        if self.data is None:
            raise RuntimeError("No data to save. Call load() and clean() first.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if format == "csv":
            self.data.to_csv(output_path, index=False)
        elif format == "json":
            self.data.to_json(output_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported output format: {format}")

        logger.info(f"Saved processed community data to {output_path}")


# --- Quick usage example ---
if __name__ == "__main__":
    scraper = CommunityScraper(source_path="data/raw/community_observations.csv")
    scraper.load()
    scraper.validate()
    scraper.clean()
    print(scraper.get_summary())
    print(scraper.get_low_engagement(threshold=4.0))
    scraper.save_processed("data/processed/community_clean.csv")