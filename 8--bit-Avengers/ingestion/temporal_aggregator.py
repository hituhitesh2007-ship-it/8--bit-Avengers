# ingestion/temporal_aggregator.py
# Aligns and aggregates all data sources along a common time axis

import pandas as pd
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalAggregator:
    """
    Merges certification, employment, and community observation data
    into a single unified time-series DataFrame per participant.

    Produces:
    - A longitudinal record per participant
    - Time-bucketed snapshots (monthly / quarterly)
    - Flags for skill decay, employment gaps, re-engagement events
    """

    VALID_FREQUENCIES = ["D", "W", "ME", "QE"]  # Day, Week, Month-End, Quarter-End

    def __init__(
        self,
        cert_path: str,
        employment_path: str,
        community_path: str,
        frequency: str = "ME"
    ):
        """
        Args:
            cert_path:        Path to processed certifications CSV
            employment_path:  Path to processed employment CSV
            community_path:   Path to processed community observations CSV
            frequency:        Time bucketing frequency — 'D', 'W', 'ME', 'QE'
        """
        if frequency not in self.VALID_FREQUENCIES:
            raise ValueError(
                f"Invalid frequency '{frequency}'. "
                f"Choose from: {self.VALID_FREQUENCIES}"
            )

        self.cert_path = cert_path
        self.employment_path = employment_path
        self.community_path = community_path
        self.frequency = frequency

        self.cert_df = None
        self.employment_df = None
        self.community_df = None
        self.unified_df = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Load all three processed data sources."""
        self.cert_df = self._load_csv(self.cert_path, "certifications")
        self.employment_df = self._load_csv(self.employment_path, "employment")
        self.community_df = self._load_csv(self.community_path, "community")

    def _load_csv(self, path: str, label: str) -> pd.DataFrame:
        """Helper to safely load a CSV with logging."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} {label} records.")
        return df

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_dates(self) -> None:
        """Convert all date columns to datetime across all sources."""
        self.cert_df["completion_date"] = pd.to_datetime(
            self.cert_df["completion_date"], errors="coerce"
        )
        self.employment_df["employment_start_date"] = pd.to_datetime(
            self.employment_df["employment_start_date"], errors="coerce"
        )
        self.community_df["observation_date"] = pd.to_datetime(
            self.community_df["observation_date"], errors="coerce"
        )

    def _bucket_to_period(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Add a time_period column bucketed to the chosen frequency."""
        df = df.copy()
        df["time_period"] = df[date_col].dt.to_period(self.frequency)
        return df

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def build_unified(self) -> pd.DataFrame:
        """
        Merge all sources into one longitudinal DataFrame.

        Steps:
        1. Normalize all dates
        2. Bucket each source into time periods
        3. Aggregate per participant per period
        4. Left-join all sources on participant_id + time_period
        5. Compute derived flags
        """
        if any(df is None for df in [self.cert_df, self.employment_df, self.community_df]):
            raise RuntimeError("Data not loaded. Call load_all() first.")

        self._normalize_dates()

        # --- Bucket each source ---
        cert = self._bucket_to_period(self.cert_df, "completion_date")
        emp = self._bucket_to_period(self.employment_df, "employment_start_date")
        comm = self._bucket_to_period(self.community_df, "observation_date")

        # --- Aggregate certifications per participant per period ---
        cert_agg = (
            cert.groupby(["participant_id", "time_period"])
            .agg(
                num_certifications=("program_name", "count"),
                avg_assessment_score=("assessment_score", "mean"),
                programs_completed=("program_name", lambda x: list(x.unique()))
            )
            .reset_index()
        )

        # --- Aggregate employment per participant per period ---
        emp_agg = (
            emp.groupby(["participant_id", "time_period"])
            .agg(
                employment_status=("employment_status", "last"),
                job_role=("job_role", "last"),
                industry=("industry", "last"),
                days_to_employment=("days_to_employment", "mean")
                if "days_to_employment" in emp.columns
                else ("employment_status", "count")
            )
            .reset_index()
        )

        # --- Aggregate community observations per participant per period ---
        comm_agg = (
            comm.groupby(["participant_id", "time_period"])
            .agg(
                avg_engagement_score=("engagement_score", "mean"),
                num_observations=("observation_text", "count"),
                observer_types=("observer_type", lambda x: list(x.unique()))
            )
            .reset_index()
        )

        # --- Merge all on participant_id + time_period ---
        unified = cert_agg.merge(
            emp_agg, on=["participant_id", "time_period"], how="outer"
        ).merge(
            comm_agg, on=["participant_id", "time_period"], how="outer"
        )

        # Sort for readability
        unified = unified.sort_values(
            ["participant_id", "time_period"]
        ).reset_index(drop=True)

        # --- Compute derived flags ---
        unified = self._add_derived_flags(unified)

        self.unified_df = unified
        logger.info(f"Unified DataFrame shape: {unified.shape}")
        return unified

    # ------------------------------------------------------------------
    # Derived Flags
    # ------------------------------------------------------------------

    def _add_derived_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add binary flags and computed columns useful for modeling:

        - is_employed:         1 if employment_status == 'employed'
        - is_underemployed:    1 if status == 'underemployed'
        - skill_applied:       1 if employed or self_employed
        - engagement_drop:     1 if engagement fell >30% from previous period
        - certified_this_period: 1 if num_certifications > 0
        """
        df = df.copy()

        # Employment flags
        df["is_employed"] = (
            df["employment_status"] == "employed"
        ).astype(int)

        df["is_underemployed"] = (
            df["employment_status"] == "underemployed"
        ).astype(int)

        df["skill_applied"] = (
            df["employment_status"].isin(["employed", "self_employed"])
        ).astype(int)

        # Certification flag
        df["certified_this_period"] = (
            df["num_certifications"].fillna(0) > 0
        ).astype(int)

        # Engagement drop flag (per participant, across time)
        df["engagement_drop"] = 0
        for pid, group in df.groupby("participant_id"):
            idx = group.index
            prev = group["avg_engagement_score"].shift(1)
            curr = group["avg_engagement_score"]
            drop = ((prev - curr) / prev.replace(0, float("nan"))) > 0.30
            df.loc[idx, "engagement_drop"] = drop.fillna(False).astype(int)

        return df

    # ------------------------------------------------------------------
    # Filtering & Access
    # ------------------------------------------------------------------

    def get_participant_timeline(self, participant_id: str) -> pd.DataFrame:
        """Return the full time-series for a single participant."""
        if self.unified_df is None:
            raise RuntimeError("Unified data not built. Call build_unified() first.")

        timeline = self.unified_df[
            self.unified_df["participant_id"] == participant_id
        ]

        if timeline.empty:
            logger.warning(f"No unified records for participant: {participant_id}")

        return timeline

    def get_period_snapshot(self, period: str) -> pd.DataFrame:
        """
        Return all participant records for a specific time period.

        Args:
            period: Period string e.g. '2023-Q1' or '2023-06'
        """
        if self.unified_df is None:
            raise RuntimeError("Unified data not built. Call build_unified() first.")

        snapshot = self.unified_df[
            self.unified_df["time_period"].astype(str) == period
        ]

        logger.info(f"Snapshot for period {period}: {len(snapshot)} records")
        return snapshot

    def get_disengaged_participants(
        self,
        min_drops: int = 2
    ) -> pd.DataFrame:
        """
        Return participants who show repeated engagement drops.
        Useful for early intervention flagging.

        Args:
            min_drops: Minimum number of periods with engagement_drop == 1
        """
        if self.unified_df is None:
            raise RuntimeError("Unified data not built. Call build_unified() first.")

        drop_counts = (
            self.unified_df.groupby("participant_id")["engagement_drop"]
            .sum()
            .reset_index()
            .rename(columns={"engagement_drop": "total_drops"})
        )

        at_risk = drop_counts[drop_counts["total_drops"] >= min_drops]
        logger.info(f"Found {len(at_risk)} participants with >= {min_drops} engagement drops.")
        return at_risk

    # ------------------------------------------------------------------
    # Summary & Export
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return summary statistics of the unified dataset."""
        if self.unified_df is None:
            raise RuntimeError("Unified data not built. Call build_unified() first.")

        df = self.unified_df

        return {
            "total_records": len(df),
            "unique_participants": df["participant_id"].nunique(),
            "time_periods": df["time_period"].nunique(),
            "frequency": self.frequency,
            "skill_application_rate": round(df["skill_applied"].mean() * 100, 2),
            "avg_engagement_score": round(df["avg_engagement_score"].dropna().mean(), 2),
            "participants_with_engagement_drops": int(df[df["engagement_drop"] == 1]["participant_id"].nunique()),
        }

    def save_unified(self, output_path: str) -> None:
        """Save the unified time-series DataFrame to CSV."""
        if self.unified_df is None:
            raise RuntimeError("No unified data to save. Call build_unified() first.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.unified_df.to_csv(output_path, index=False)
        logger.info(f"Saved unified temporal data to {output_path}")


# --- Quick usage example ---
if __name__ == "__main__":
    agg = TemporalAggregator(
        cert_path="data/processed/certifications_clean.csv",
        employment_path="data/processed/employment_clean.csv",
        community_path="data/processed/community_clean.csv",
        frequency="ME"
    )

    agg.load_all()
    unified = agg.build_unified()
    print(agg.get_summary())
    print(agg.get_disengaged_participants(min_drops=2))
    agg.save_unified("data/processed/unified_temporal.csv")