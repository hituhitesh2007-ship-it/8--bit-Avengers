# linkedin_scraper.py
# TODO: implement
# ingestion/linkedin_scraper.py
# Loads LinkedIn-sourced profile and activity data (via export or API proxy)
# NOTE: Direct LinkedIn scraping violates ToS. This module works with
#       exported CSV data (LinkedIn Data Export) or a licensed API proxy.

import os
import json
import logging
import pandas as pd
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinkedInScraper:
    """
    Loads and processes LinkedIn-sourced data including:
    - Profile skills and endorsements
    - Connection network size
    - Job change history
    - Certification badges
    - Activity signals (posts, reactions, comments)
    - Education and volunteer experience

    Data sources supported:
    - LinkedIn Data Export ZIP (CSV files inside)
    - Pre-processed JSON from a licensed API proxy (e.g. Proxycurl)
    """

    PROFILE_COLUMNS = [
        "participant_id",
        "headline",
        "industry",
        "connections",
        "region",
        "skills",
        "num_endorsements",
        "num_positions",
        "num_certifications",
        "activity_score"
    ]

    def __init__(self, source_path: str, id_column: str = "participant_id"):
        """
        Args:
            source_path: Path to LinkedIn export CSV or processed JSON
            id_column:   Column that uniquely identifies each participant
        """
        self.source_path = source_path
        self.id_column = id_column
        self.data = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Load LinkedIn data from CSV or JSON."""
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Source not found: {self.source_path}")

        ext = os.path.splitext(self.source_path)[-1].lower()

        if ext == ".csv":
            self.data = pd.read_csv(self.source_path)
        elif ext == ".json":
            with open(self.source_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.data = pd.DataFrame(raw if isinstance(raw, list) else [raw])
        else:
            raise ValueError(f"Unsupported format: {ext}. Use .csv or .json")

        logger.info(f"Loaded {len(self.data)} LinkedIn profile records.")
        return self.data

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Check that participant ID column exists."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        if self.id_column not in self.data.columns:
            logger.error(f"ID column '{self.id_column}' not found.")
            return False

        logger.info("Validation passed.")
        return True

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean(self) -> pd.DataFrame:
        """
        Cleaning steps:
        - Normalize string fields
        - Parse skills into lists
        - Fill missing numeric fields with 0
        - Compute derived engagement signals
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        df = self.data.copy()

        # Drop duplicates by participant ID
        df.drop_duplicates(subset=[self.id_column], inplace=True)

        # Normalize string fields
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

        # Lowercase categorical fields
        for col in ["headline", "industry", "region"]:
            if col in df.columns:
                df[col] = df[col].str.lower()

        # Parse skills from comma-separated string to list
        if "skills" in df.columns:
            df["skills_list"] = df["skills"].fillna("").apply(
                lambda x: [s.strip().lower() for s in x.split(",") if s.strip()]
            )
            df["num_skills_listed"] = df["skills_list"].apply(len)

        # Fill numeric columns
        numeric_cols = ["connections", "num_endorsements", "num_positions",
                        "num_certifications", "activity_score"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Network strength score (heuristic composite)
        if "connections" in df.columns and "num_endorsements" in df.columns:
            df["network_strength"] = (
                (df["connections"] / df["connections"].max().clip(1)) * 0.6 +
                (df["num_endorsements"] / df["num_endorsements"].max().clip(1)) * 0.4
            ).round(4)

        logger.info(f"Cleaned LinkedIn data shape: {df.shape}")
        self.data = df
        return df

    # ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------

    def get_skill_coverage(self, target_skills: list) -> pd.DataFrame:
        """
        For each participant, compute what fraction of the target skill list
        they have listed on their profile.

        Args:
            target_skills: List of skills to check coverage for

        Returns:
            DataFrame with participant_id and skill_coverage_ratio
        """
        if self.data is None:
            raise RuntimeError("Data not loaded.")
        if "skills_list" not in self.data.columns:
            raise ValueError("skills_list not found. Call clean() first.")

        target_set = set(s.lower() for s in target_skills)

        def coverage(skills):
            if not target_set:
                return 0.0
            return round(len(set(skills) & target_set) / len(target_set), 4)

        result = self.data[[self.id_column, "skills_list"]].copy()
        result["skill_coverage_ratio"] = result["skills_list"].apply(coverage)
        return result

    def get_most_connected(self, n: int = 10) -> pd.DataFrame:
        """Return top N participants by connection count."""
        if self.data is None:
            raise RuntimeError("Data not loaded.")
        if "connections" not in self.data.columns:
            raise ValueError("'connections' column not found.")
        return (
            self.data[[self.id_column, "connections"]]
            .sort_values("connections", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def get_by_industry(self, industry: str) -> pd.DataFrame:
        """Return profiles filtered by industry."""
        if self.data is None:
            raise RuntimeError("Data not loaded.")
        return self.data[self.data["industry"] == industry.lower()]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return summary statistics of the LinkedIn dataset."""
        if self.data is None:
            raise RuntimeError("Data not loaded.")

        summary = {
            "total_profiles": len(self.data),
            "unique_industries": self.data["industry"].nunique() if "industry" in self.data.columns else "N/A",
            "regions": self.data["region"].unique().tolist() if "region" in self.data.columns else [],
        }
        for col in ["connections", "num_endorsements", "network_strength"]:
            if col in self.data.columns:
                summary[f"avg_{col}"] = round(self.data[col].mean(), 2)
        return summary

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_processed(self, output_path: str) -> None:
        """Save cleaned LinkedIn data to CSV."""
        if self.data is None:
            raise RuntimeError("No data to save.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.data.to_csv(output_path, index=False)
        logger.info(f"Saved LinkedIn data to {output_path}")


if __name__ == "__main__":
    scraper = LinkedInScraper(source_path="data/raw/linkedin_profiles.csv")
    scraper.load()
    scraper.validate()
    scraper.clean()
    print(scraper.get_summary())
    scraper.save_processed("data/processed/linkedin_clean.csv")