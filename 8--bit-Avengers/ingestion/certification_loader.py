# ingestion/certification_loader.py
# Loads and parses certification records from CSV/Excel/DB sources

import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CertificationLoader:
    """
    Loads certification records including:
    - Participant ID
    - Program name & type
    - Completion date
    - Institution / region
    - Assessment scores
    """

    REQUIRED_COLUMNS = [
        "participant_id",
        "program_name",
        "program_type",
        "completion_date",
        "institution",
        "region",
        "assessment_score"
    ]

    def __init__(self, source_path: str):
        """
        Args:
            source_path: Path to the certification data file (.csv or .xlsx)
        """
        self.source_path = source_path
        self.data = None

    def load(self) -> pd.DataFrame:
        """Load raw certification data from file."""
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Source file not found: {self.source_path}")

        ext = os.path.splitext(self.source_path)[-1].lower()

        if ext == ".csv":
            self.data = pd.read_csv(self.source_path)
        elif ext in (".xlsx", ".xls"):
            self.data = pd.read_excel(self.source_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        logger.info(f"Loaded {len(self.data)} certification records from {self.source_path}")
        return self.data

    def validate(self) -> bool:
        """Check that all required columns are present."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.data.columns]

        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False

        logger.info("Validation passed.")
        return True

    def clean(self) -> pd.DataFrame:
        """
        Basic cleaning:
        - Drop duplicates
        - Normalize dates
        - Strip whitespace from string fields
        - Fill missing scores with 0
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        df = self.data.copy()

        # Drop duplicate records
        df.drop_duplicates(subset=["participant_id", "program_name", "completion_date"], inplace=True)

        # Normalize completion_date to datetime
        df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")

        # Strip whitespace from string columns
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

        # Normalize region and program_type to lowercase
        df["region"] = df["region"].str.lower()
        df["program_type"] = df["program_type"].str.lower()

        # Fill missing assessment scores
        df["assessment_score"] = df["assessment_score"].fillna(0)

        logger.info(f"Cleaned data shape: {df.shape}")
        self.data = df
        return df

    def get_summary(self) -> dict:
        """Return a quick summary of the loaded dataset."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        return {
            "total_records": len(self.data),
            "unique_participants": self.data["participant_id"].nunique(),
            "unique_programs": self.data["program_name"].nunique(),
            "regions": self.data["region"].unique().tolist(),
            "date_range": {
                "start": str(self.data["completion_date"].min()),
                "end": str(self.data["completion_date"].max())
            },
            "avg_assessment_score": round(self.data["assessment_score"].mean(), 2)
        }

    def save_processed(self, output_path: str) -> None:
        """Save cleaned data to the processed/ directory."""
        if self.data is None:
            raise RuntimeError("No data to save. Call load() and clean() first.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.data.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")


# --- Quick usage example ---
if __name__ == "__main__":
    loader = CertificationLoader(source_path="data/raw/certifications.csv")
    loader.load()
    loader.validate()
    loader.clean()
    print(loader.get_summary())
    loader.save_processed("data/processed/certifications_clean.csv")