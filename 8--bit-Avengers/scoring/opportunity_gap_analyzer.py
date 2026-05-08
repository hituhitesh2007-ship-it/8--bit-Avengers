# opportunity_gap_analyzer.py
# TODO: implement
# scoring/opportunity_gap_analyzer.py
# Identifies gaps between a participant's current profile and target job requirements

import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpportunityGapAnalyzer:
    """
    Compares each participant's current skill/certification/experience profile
    against the requirements of their target occupation (or closest matching
    job market demand cluster).

    Produces per-participant gap scores and actionable recommendations such as:
      - Skills to acquire
      - Certifications to pursue
      - Training hours needed

    Inputs
    ------
    participant_df : one row per participant with columns:
        participant_id, target_occupation, skills (list/set),
        certifications (list/set), years_experience, training_hours_completed

    job_requirements_df : one row per occupation with columns:
        occupation, required_skills (list/set), required_certifications (list/set),
        min_years_experience, min_training_hours
    """

    GAP_WEIGHT = {
        "skill_gap":         0.40,
        "cert_gap":          0.25,
        "experience_gap":    0.20,
        "training_gap":      0.15,
    }

    def __init__(
        self,
        participant_df: pd.DataFrame,
        job_requirements_df: pd.DataFrame,
    ):
        self.participants = participant_df.copy()
        self.requirements = job_requirements_df.copy()
        self.gaps_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self) -> pd.DataFrame:
        """
        Compute opportunity gaps for all participants.

        Returns:
            DataFrame with one row per participant containing gap scores,
            composite gap index, and top missing skills / certifications.
        """
        records = []
        req_index = self.requirements.set_index("occupation")

        for _, p in self.participants.iterrows():
            occupation = p.get("target_occupation")
            if occupation not in req_index.index:
                logger.warning(
                    f"No requirements found for occupation '{occupation}'. Skipping."
                )
                continue

            req = req_index.loc[occupation]
            record = self._compute_gap(p, req)
            records.append(record)

        self.gaps_ = pd.DataFrame(records).reset_index(drop=True)
        logger.info(
            f"Gap analysis complete for {len(self.gaps_)} participants. "
            f"Mean gap index = {self.gaps_['gap_index'].mean():.2f}"
        )
        return self.gaps_

    def _compute_gap(self, participant: pd.Series, requirement: pd.Series) -> dict:
        """Compute gap metrics for a single participant vs. their target occupation."""

        # --- Skills ---
        p_skills = set(participant.get("skills") or [])
        r_skills = set(requirement.get("required_skills") or [])
        missing_skills = r_skills - p_skills
        skill_gap = len(missing_skills) / max(len(r_skills), 1)

        # --- Certifications ---
        p_certs = set(participant.get("certifications") or [])
        r_certs = set(requirement.get("required_certifications") or [])
        missing_certs = r_certs - p_certs
        cert_gap = len(missing_certs) / max(len(r_certs), 1)

        # --- Experience ---
        p_exp = float(participant.get("years_experience") or 0)
        r_exp = float(requirement.get("min_years_experience") or 0)
        experience_gap = max(0.0, (r_exp - p_exp) / max(r_exp, 1))

        # --- Training ---
        p_train = float(participant.get("training_hours_completed") or 0)
        r_train = float(requirement.get("min_training_hours") or 0)
        training_gap = max(0.0, (r_train - p_train) / max(r_train, 1))

        # --- Composite gap index (0 = no gap, 100 = completely unqualified) ---
        gap_index = (
            skill_gap     * self.GAP_WEIGHT["skill_gap"]
            + cert_gap    * self.GAP_WEIGHT["cert_gap"]
            + experience_gap * self.GAP_WEIGHT["experience_gap"]
            + training_gap   * self.GAP_WEIGHT["training_gap"]
        ) * 100

        return {
            "participant_id":          participant.get("participant_id"),
            "target_occupation":       participant.get("target_occupation"),
            "skill_gap":               round(skill_gap, 4),
            "cert_gap":                round(cert_gap, 4),
            "experience_gap":          round(experience_gap, 4),
            "training_gap":            round(training_gap, 4),
            "gap_index":               round(gap_index, 2),
            "missing_skills":          list(missing_skills),
            "missing_certifications":  list(missing_certs),
            "extra_years_needed":      max(0.0, r_exp - p_exp),
            "extra_training_needed":   max(0.0, r_train - p_train),
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_high_gap_participants(self, threshold: float = 60.0) -> pd.DataFrame:
        """Return participants with gap_index above threshold (hardest to place)."""
        self._check_analyzed()
        result = self.gaps_[self.gaps_["gap_index"] > threshold]
        logger.info(f"{len(result)} participants with gap_index > {threshold}.")
        return result.reset_index(drop=True)

    def get_near_ready(self, threshold: float = 20.0) -> pd.DataFrame:
        """Return participants nearly ready for their target occupation."""
        self._check_analyzed()
        result = self.gaps_[self.gaps_["gap_index"] <= threshold]
        logger.info(f"{len(result)} participants nearly job-ready (gap ≤ {threshold}).")
        return result.reset_index(drop=True)

    def get_most_common_skill_gaps(self, top_n: int = 10) -> pd.DataFrame:
        """Return the most frequently missing skills across all participants."""
        self._check_analyzed()
        from collections import Counter
        all_missing = []
        for skills in self.gaps_["missing_skills"]:
            all_missing.extend(skills)
        freq = Counter(all_missing)
        df = pd.DataFrame(freq.items(), columns=["skill", "frequency"])
        return df.sort_values("frequency", ascending=False).head(top_n).reset_index(drop=True)

    def get_most_common_cert_gaps(self, top_n: int = 10) -> pd.DataFrame:
        """Return the most frequently missing certifications."""
        self._check_analyzed()
        from collections import Counter
        all_missing = []
        for certs in self.gaps_["missing_certifications"]:
            all_missing.extend(certs)
        freq = Counter(all_missing)
        df = pd.DataFrame(freq.items(), columns=["certification", "frequency"])
        return df.sort_values("frequency", ascending=False).head(top_n).reset_index(drop=True)

    def get_gap_by_occupation(self) -> pd.DataFrame:
        """Return average gap index broken down by target occupation."""
        self._check_analyzed()
        return (
            self.gaps_
            .groupby("target_occupation")["gap_index"]
            .agg(["mean", "median", "count"])
            .round(2)
            .rename(columns={"mean": "avg_gap", "median": "median_gap", "count": "n_participants"})
            .sort_values("avg_gap", ascending=False)
            .reset_index()
        )

    def get_recommendations(self, participant_id: str) -> Dict[str, Any]:
        """
        Return a structured set of recommendations for a specific participant.

        Returns:
            dict with keys: participant_id, target_occupation, gap_index,
            recommended_skills, recommended_certifications, extra_years_needed,
            extra_training_hours_needed
        """
        self._check_analyzed()
        row = self.gaps_[self.gaps_["participant_id"] == participant_id]
        if row.empty:
            raise ValueError(f"Participant '{participant_id}' not found in gap results.")
        row = row.iloc[0]
        return {
            "participant_id":               participant_id,
            "target_occupation":            row["target_occupation"],
            "gap_index":                    row["gap_index"],
            "recommended_skills":           row["missing_skills"],
            "recommended_certifications":   row["missing_certifications"],
            "extra_years_needed":           row["extra_years_needed"],
            "extra_training_hours_needed":  row["extra_training_needed"],
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(self, output_path: str) -> None:
        """Save gap analysis results to CSV."""
        self._check_analyzed()
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # Flatten list columns for CSV
        export = self.gaps_.copy()
        export["missing_skills"] = export["missing_skills"].apply(
            lambda x: "|".join(x) if x else ""
        )
        export["missing_certifications"] = export["missing_certifications"].apply(
            lambda x: "|".join(x) if x else ""
        )
        export.to_csv(output_path, index=False)
        logger.info(f"Gap analysis saved → {output_path}")

    def _check_analyzed(self) -> None:
        if self.gaps_ is None:
            raise RuntimeError("No analysis computed. Call analyze() first.")


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    participants = pd.DataFrame([
        {
            "participant_id": "P001",
            "target_occupation": "Data Analyst",
            "skills": ["python", "excel", "sql"],
            "certifications": ["Excel Cert"],
            "years_experience": 1,
            "training_hours_completed": 40,
        },
        {
            "participant_id": "P002",
            "target_occupation": "Logistics Coordinator",
            "skills": ["logistics", "communication"],
            "certifications": [],
            "years_experience": 3,
            "training_hours_completed": 20,
        },
    ])

    job_requirements = pd.DataFrame([
        {
            "occupation": "Data Analyst",
            "required_skills": ["python", "sql", "machine learning", "excel", "data analysis"],
            "required_certifications": ["Excel Cert", "SQL Cert"],
            "min_years_experience": 2,
            "min_training_hours": 80,
        },
        {
            "occupation": "Logistics Coordinator",
            "required_skills": ["logistics", "communication", "project management"],
            "required_certifications": ["Logistics Cert"],
            "min_years_experience": 2,
            "min_training_hours": 30,
        },
    ])

    analyzer = OpportunityGapAnalyzer(participants, job_requirements)
    gaps = analyzer.analyze()
    print(gaps[["participant_id", "gap_index", "missing_skills"]].to_string())
    print("\nRecommendations for P001:")
    print(analyzer.get_recommendations("P001"))
