# ingestion/employment_connector.py
# Loads and parses employment trajectory data for trained participants

import pandas as pd
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmploymentConnector:
    """
    Loads employment trajectory records including:
    - Participant ID
    - Employment status over time
    - Job role & industry
    - Time-to-employment after certification
    - Salary band (optional)
    - Employment duration
    - Region
    """

    REQUIRED_COLUMNS = [
        "participant_id",
        "employment_status",
        "job_role",
        "industry",
        "employment_start_date",
        "region"
    ]

    VALID_STATUSES = [
        "employed",
        "unemployed",
        "self_employed",
        "underemployed",
        "unknown"
    ]

    def __init__(self, source_path: str):
        """
        Args:
            source_path: Path to employment data file (.csv or .xlsx)
        """
        self.source_path = source_path
        self.data = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Load raw employment data from file."""
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Source file not found: {self.source_path}")

        ext = os.path.splitext(self.source_path)[-1].lower()

        if ext == ".csv":
            self.data = pd.read_csv(self.source_path)
        elif ext in (".xlsx", ".xls"):
            self.data = pd.read_excel(self.source_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        logger.info(f"Loaded {len(self.data)} employment records from {self.source_path}")
        return self.data

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Check required columns exist and status values are valid."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        # Check required columns
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.data.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False

        # Check employment status values
        invalid_statuses = self.data[
            ~self.data["employment_status"].str.lower().isin(self.VALID_STATUSES)
        ]
        if not invalid_statuses.empty:
            logger.warning(
                f"Invalid employment statuses found: "
                f"{invalid_statuses['employment_status'].unique()}"
            )

        logger.info("Validation passed.")
        return True

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean(self) -> pd.DataFrame:
        """
        Cleaning steps:
        - Normalize dates
        - Standardize employment status labels
        - Strip whitespace
        - Drop duplicates
        - Compute time_to_employment if certification_date is present
        - Compute employment duration if end date is present
        - Flag anomalies
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        df = self.data.copy()

        # Drop duplicates
        df.drop_duplicates(
            subset=["participant_id", "employment_start_date", "job_role"],
            inplace=True
        )

        # Normalize string fields
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda col: col.str.strip().str.lower())

        # Normalize dates
        df["employment_start_date"] = pd.to_datetime(
            df["employment_start_date"], errors="coerce"
        )

        if "employment_end_date" in df.columns:
            df["employment_end_date"] = pd.to_datetime(
                df["employment_end_date"], errors="coerce"
            )
            # Compute employment duration in days
            df["employment_duration_days"] = (
                df["employment_end_date"] - df["employment_start_date"]
            ).dt.days

        # Compute time-to-employment if certification_date exists
        if "certification_date" in df.columns:
            df["certification_date"] = pd.to_datetime(
                df["certification_date"], errors="coerce"
            )
            df["days_to_employment"] = (
                df["employment_start_date"] - df["certification_date"]
            ).dt.days

            # Flag negative values (data errors — employed before certified)
            invalid = df[df["days_to_employment"] < 0]
            if not invalid.empty:
                logger.warning(
                    f"{len(invalid)} records have employment before certification date."
                )
            df["anomaly_flag"] = df["days_to_employment"] < 0

        # Standardize employment status — fill nulls
        df["employment_status"] = df["employment_status"].fillna("unknown")

        # Salary normalization if present
        for col in ["salary_min", "salary_max"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "salary_min" in df.columns and "salary_max" in df.columns:
            df["salary_midpoint"] = (df["salary_min"] + df["salary_max"]) / 2

        logger.info(f"Cleaned employment data shape: {df.shape}")
        self.data = df
        return df

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_trajectory(self, participant_id: str) -> pd.DataFrame:
        """
        Return full employment timeline for a single participant.

        Args:
            participant_id: The participant's unique ID

        Returns:
            DataFrame of employment records sorted by date
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        trajectory = self.data[
            self.data["participant_id"] == participant_id
        ].sort_values("employment_start_date")

        if trajectory.empty:
            logger.warning(f"No records found for participant: {participant_id}")

        return trajectory

    def filter_by_region(self, region: str) -> pd.DataFrame:
        """Return records filtered by a specific region."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        filtered = self.data[self.data["region"] == region.lower()]
        logger.info(f"Filtered to {len(filtered)} records for region: {region}")
        return filtered

    def filter_by_status(self, status: str) -> pd.DataFrame:
        """Return records filtered by employment status."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        if status.lower() not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Choose from: {self.VALID_STATUSES}"
            )

        filtered = self.data[self.data["employment_status"] == status.lower()]
        logger.info(f"Filtered to {len(filtered)} records with status: {status}")
        return filtered

    def filter_by_industry(self, industry: str) -> pd.DataFrame:
        """Return records filtered by industry."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        filtered = self.data[self.data["industry"] == industry.lower()]
        logger.info(f"Filtered to {len(filtered)} records in industry: {industry}")
        return filtered

    def get_long_term_unemployed(self, threshold_days: int = 180) -> pd.DataFrame:
        """
        Return participants who took more than threshold_days to find employment
        after certification. Useful for intervention targeting.

        Args:
            threshold_days: Number of days considered long-term unemployment
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        if "days_to_employment" not in self.data.columns:
            raise ValueError(
                "days_to_employment not computed. "
                "Ensure certification_date column exists and call clean() first."
            )

        long_term = self.data[self.data["days_to_employment"] > threshold_days]
        logger.info(
            f"Found {len(long_term)} records with >{threshold_days} days to employment."
        )
        return long_term

    def get_underemployed(self) -> pd.DataFrame:
        """Return all participants currently marked as underemployed."""
        return self.filter_by_status("underemployed")

    # ------------------------------------------------------------------
    # Summary & Analytics
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return a high-level summary of the employment dataset."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        summary = {
            "total_records": len(self.data),
            "unique_participants": self.data["participant_id"].nunique(),
            "status_distribution": (
                self.data["employment_status"].value_counts().to_dict()
            ),
            "top_industries": (
                self.data["industry"].value_counts().head(5).to_dict()
            ),
            "top_job_roles": (
                self.data["job_role"].value_counts().head(5).to_dict()
            ),
            "regions": self.data["region"].unique().tolist(),
        }

        if "days_to_employment" in self.data.columns:
            valid = self.data["days_to_employment"].dropna()
            summary["avg_days_to_employment"] = round(valid.mean(), 1)
            summary["median_days_to_employment"] = round(valid.median(), 1)
            summary["max_days_to_employment"] = int(valid.max())

        if "employment_duration_days" in self.data.columns:
            valid_dur = self.data["employment_duration_days"].dropna()
            summary["avg_employment_duration_days"] = round(valid_dur.mean(), 1)

        if "salary_midpoint" in self.data.columns:
            summary["avg_salary_midpoint"] = round(
                self.data["salary_midpoint"].dropna().mean(), 2
            )

        if "anomaly_flag" in self.data.columns:
            summary["anomaly_count"] = int(self.data["anomaly_flag"].sum())

        return summary

    def get_status_by_region(self) -> pd.DataFrame:
        """Return employment status distribution broken down by region."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        return (
            self.data.groupby(["region", "employment_status"])
            .size()
            .reset_index(name="count")
            .sort_values(["region", "count"], ascending=[True, False])
        )

    def get_avg_days_by_industry(self) -> pd.DataFrame:
        """Return average days-to-employment grouped by industry."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        if "days_to_employment" not in self.data.columns:
            raise ValueError(
                "days_to_employment not found. "
                "Ensure certification_date exists and call clean() first."
            )

        return (
            self.data.groupby("industry")["days_to_employment"]
            .mean()
            .round(1)
            .reset_index()
            .rename(columns={"days_to_employment": "avg_days_to_employment"})
            .sort_values("avg_days_to_employment")
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_processed(self, output_path: str) -> None:
        """Save cleaned data to the processed/ directory."""
        if self.data is None:
            raise RuntimeError("No data to save. Call load() and clean() first.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.data.to_csv(output_path, index=False)
        logger.info(f"Saved processed employment data to {output_path}")


# ------------------------------------------------------------------
# Quick usage example
# ------------------------------------------------------------------
if __name__ == "__main__":
    connector = EmploymentConnector(source_path="data/raw/employment.csv")
    connector.load()
    connector.validate()
    connector.clean()

    print(connector.get_summary())
    print(connector.get_status_by_region())
    print(connector.get_avg_days_by_industry())
    print(connector.get_long_term_unemployed(threshold_days=180))

    connector.save_processed("data/processed/employment_clean.csv")