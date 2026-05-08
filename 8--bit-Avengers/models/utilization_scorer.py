# utilization_scorer.py
# TODO: implement
# models/utilization_scorer.py
# Computes a composite Skill Utilization Score for each participant

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from sklearn.preprocessing import MinMaxScaler
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UtilizationScorer:
    """
    Computes a composite Skill Utilization Score (SUS) for each participant.

    The SUS quantifies the gap between a participant's acquired skills and
    their actual economic/career utilization of those skills.

    Score Components (configurable weights):
    1. Employment Alignment Score  — is the job role using certified skills?
    2. Salary Adequacy Score       — is the salary commensurate with qualifications?
    3. Skill Depth Score           — skill count relative to role requirements
    4. Career Progression Score    — upward mobility over time
    5. Underemployment Penalty     — deduction for formal overqualification
    6. Temporal Recency Score      — skill freshness / recency of employment

    Final SUS: weighted sum, normalized to [0, 1]
    """

    DEFAULT_WEIGHTS = {
        "employment_alignment": 0.30,
        "salary_adequacy":      0.20,
        "skill_depth":          0.20,
        "career_progression":   0.15,
        "temporal_recency":     0.10,
        "underemployment":     -0.05   # negative = penalty
    }

    SCORE_BANDS = {
        "critically_underutilized": (0.0,  0.25),
        "underutilized":            (0.25, 0.50),
        "moderately_utilized":      (0.50, 0.70),
        "well_utilized":            (0.70, 0.85),
        "fully_utilized":           (0.85, 1.01)
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Custom component weights. Must sum to ~1.0 (excluding penalty)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.scaler = MinMaxScaler()
        self._component_columns = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Component Computations
    # ------------------------------------------------------------------

    def compute_employment_alignment(
        self,
        df: pd.DataFrame,
        skills_col: str = "skills_list",
        job_skills_col: str = "job_required_skills"
    ) -> pd.Series:
        """
        Fraction of participant skills that match current job requirements.
        """
        def overlap(row):
            p = set(str(row.get(skills_col, "")).lower().split(","))
            j = set(str(row.get(job_skills_col, "")).lower().split(","))
            if not j:
                return 0.0
            return len(p & j) / len(j)

        return df.apply(overlap, axis=1).clip(0, 1)

    def compute_salary_adequacy(
        self,
        df: pd.DataFrame,
        actual_salary_col: str = "salary_midpoint",
        expected_salary_col: str = "market_salary_midpoint"
    ) -> pd.Series:
        """
        Ratio of actual salary to expected market salary.
        """
        if actual_salary_col not in df.columns or expected_salary_col not in df.columns:
            logger.warning("Salary columns missing. Returning 0.5 default.")
            return pd.Series(np.full(len(df), 0.5))

        ratio = df[actual_salary_col] / (df[expected_salary_col].replace(0, np.nan))
        return ratio.fillna(0.5).clip(0, 1.5).apply(lambda x: min(x, 1.0))

    def compute_skill_depth(
        self,
        df: pd.DataFrame,
        skill_count_col: str = "skill_count",
        role_avg_skills: float = 8.0
    ) -> pd.Series:
        """
        Skill depth relative to average role requirement.
        """
        if skill_count_col not in df.columns:
            return pd.Series(np.full(len(df), 0.5))
        return (df[skill_count_col] / role_avg_skills).clip(0, 1)

    def compute_career_progression(
        self,
        df: pd.DataFrame,
        role_seniority_col: str = "seniority_level_encoded"
    ) -> pd.Series:
        """
        Normalized seniority level as a proxy for career progression.
        """
        if role_seniority_col not in df.columns:
            return pd.Series(np.full(len(df), 0.5))
        vals = df[role_seniority_col].fillna(0).values.astype(float)
        max_val = vals.max() if vals.max() > 0 else 1.0
        return pd.Series(vals / max_val)

    def compute_temporal_recency(
        self,
        df: pd.DataFrame,
        last_cert_date_col: str = "last_certification_date",
        max_decay_days: int = 730
    ) -> pd.Series:
        """
        Score based on how recently skills/certifications were updated.
        Decays from 1.0 (today) to 0.0 (>2 years ago).
        """
        if last_cert_date_col not in df.columns:
            return pd.Series(np.full(len(df), 0.5))

        today = pd.Timestamp.today()
        df = df.copy()
        df[last_cert_date_col] = pd.to_datetime(df[last_cert_date_col], errors="coerce")
        days_ago = (today - df[last_cert_date_col]).dt.days.fillna(max_decay_days)
        return (1 - days_ago.clip(0, max_decay_days) / max_decay_days).round(4)

    def compute_underemployment_flag(
        self,
        df: pd.DataFrame,
        status_col: str = "employment_status"
    ) -> pd.Series:
        """
        Binary flag: 1 if participant is underemployed (triggers penalty).
        """
        if status_col not in df.columns:
            return pd.Series(np.zeros(len(df)))
        return (df[status_col].str.lower() == "underemployed").astype(float)

    # ------------------------------------------------------------------
    # Composite Score
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute the full Skill Utilization Score for each participant.

        Args:
            df:      Input DataFrame
            kwargs:  Column name overrides for any component method

        Returns:
            DataFrame with all component scores and final SUS
        """
        result = df[["participant_id"]].copy() if "participant_id" in df.columns else pd.DataFrame()

        result["employment_alignment"] = self.compute_employment_alignment(df)
        result["salary_adequacy"]      = self.compute_salary_adequacy(df)
        result["skill_depth"]          = self.compute_skill_depth(df)
        result["career_progression"]   = self.compute_career_progression(df)
        result["temporal_recency"]     = self.compute_temporal_recency(df)
        result["underemployment_flag"] = self.compute_underemployment_flag(df)

        # Weighted composite
        result["utilization_score"] = (
            self.weights["employment_alignment"] * result["employment_alignment"] +
            self.weights["salary_adequacy"]      * result["salary_adequacy"] +
            self.weights["skill_depth"]          * result["skill_depth"] +
            self.weights["career_progression"]   * result["career_progression"] +
            self.weights["temporal_recency"]     * result["temporal_recency"] +
            self.weights["underemployment"]      * result["underemployment_flag"]
        ).clip(0, 1).round(4)

        result["utilization_band"] = result["utilization_score"].apply(self._assign_band)

        self._is_fitted = True
        logger.info(
            f"Utilization scores computed for {len(result)} participants. "
            f"Mean SUS: {result['utilization_score'].mean():.3f}"
        )
        return result

    def _assign_band(self, score: float) -> str:
        """Assign a named band to a score."""
        for band, (low, high) in self.SCORE_BANDS.items():
            if low <= score < high:
                return band
        return "unknown"

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_band_distribution(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """Return count and percentage per utilization band."""
        if "utilization_band" not in scored_df.columns:
            raise ValueError("Run compute() first.")
        counts = scored_df["utilization_band"].value_counts().reset_index()
        counts.columns = ["band", "count"]
        counts["pct"] = (counts["count"] / len(scored_df) * 100).round(1)
        return counts

    def get_bottom_n(self, scored_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Return N most underutilized participants."""
        return (
            scored_df.sort_values("utilization_score")
            .head(n)
            .reset_index(drop=True)
        )

    def get_component_summary(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """Return mean of each component score."""
        component_cols = [
            "employment_alignment", "salary_adequacy", "skill_depth",
            "career_progression", "temporal_recency", "underemployment_flag"
        ]
        available = [c for c in component_cols if c in scored_df.columns]
        return pd.DataFrame({
            "component": available,
            "mean_score": [round(scored_df[c].mean(), 4) for c in available],
            "weight": [self.weights.get(c, 0) for c in available]
        })

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.weights, os.path.join(output_dir, "utilization_weights.pkl"))
        logger.info(f"UtilizationScorer saved to {output_dir}")

    def load(self, output_dir: str) -> "UtilizationScorer":
        self.weights = joblib.load(os.path.join(output_dir, "utilization_weights.pkl"))
        self._is_fitted = True
        logger.info(f"UtilizationScorer loaded from {output_dir}")
        return self


if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    dummy = pd.DataFrame({
        "participant_id": [f"P{i:04d}" for i in range(n)],
        "skills_list": ["python,sql,ml,communication"] * n,
        "job_required_skills": ["python,sql,deep learning,communication"] * n,
        "salary_midpoint": np.random.uniform(20000, 100000, n),
        "market_salary_midpoint": np.random.uniform(30000, 90000, n),
        "skill_count": np.random.randint(3, 18, n),
        "seniority_level_encoded": np.random.randint(0, 4, n),
        "last_certification_date": pd.date_range("2022-01-01", periods=n, freq="3D"),
        "employment_status": np.random.choice(["employed", "underemployed", "unemployed"], n)
    })

    scorer = UtilizationScorer()
    results = scorer.compute(dummy)
    print(results[["participant_id", "utilization_score", "utilization_band"]].head(10))
    print(scorer.get_band_distribution(results))
    print(scorer.get_component_summary(results))
