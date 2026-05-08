# utilization_score.py
# TODO: implement
# scoring/utilization_score.py
# Measures how effectively a participant's skills and certifications are being utilised
# in their current or most recent employment role

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UtilizationScore:
    """
    Computes a Skill Utilization Score (0–100) for each participant reflecting
    whether their qualifications, certifications, and training are being used
    in their current employment.

    Key dimensions:
      - skill_utilization_rate   : fraction of participant's skills used in current role
      - cert_utilization_rate    : fraction of certifications relevant to current role
      - role_level_match         : whether role seniority matches experience level
      - salary_utilization       : current salary vs. market median for role (optional)

    Inputs
    ------
    participant_df : one row per participant
        Required columns:
            participant_id, skills (list), certifications (list),
            years_experience, current_job_role, current_industry

    role_requirements_df : one row per job role
        Required columns:
            job_role, required_skills (list), required_certifications (list),
            expected_min_years_exp, median_salary (optional)

    Optional participant columns:
        current_salary  — enables salary-based utilization sub-score
    """

    WEIGHT = {
        "skill_utilization_rate": 0.40,
        "cert_utilization_rate":  0.25,
        "role_level_match":       0.20,
        "salary_utilization":     0.15,
    }

    def __init__(
        self,
        participant_df: pd.DataFrame,
        role_requirements_df: pd.DataFrame,
        include_salary: bool = False,
    ):
        self.participants = participant_df.copy()
        self.role_reqs = role_requirements_df.set_index("job_role")
        self.include_salary = include_salary
        self.scores_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute(self) -> pd.DataFrame:
        """
        Compute utilization scores for all participants.

        Returns:
            DataFrame with utilization sub-scores, composite score, and band.
        """
        records = []
        for _, p in self.participants.iterrows():
            role = p.get("current_job_role")
            if role not in self.role_reqs.index:
                logger.warning(f"Role '{role}' not found in requirements. Skipping.")
                continue
            req = self.role_reqs.loc[role]
            records.append(self._score_participant(p, req))

        self.scores_ = pd.DataFrame(records).reset_index(drop=True)
        logger.info(
            f"Utilization scores computed for {len(self.scores_)} participants. "
            f"Mean = {self.scores_['utilization_score'].mean():.2f}"
        )
        return self.scores_

    def _score_participant(self, p: pd.Series, req: pd.Series) -> dict:
        p_skills = set(p.get("skills") or [])
        r_skills = set(req.get("required_skills") or [])
        skill_util = len(p_skills & r_skills) / max(len(p_skills), 1)

        p_certs = set(p.get("certifications") or [])
        r_certs = set(req.get("required_certifications") or [])
        cert_util = len(p_certs & r_certs) / max(len(p_certs), 1)

        # Role-level match: compare years_experience to expected minimum
        p_exp  = float(p.get("years_experience") or 0)
        r_exp  = float(req.get("expected_min_years_exp") or 0)
        role_match = min(1.0, p_exp / max(r_exp, 1))

        # Salary utilization (optional)
        salary_util = 0.5  # neutral default when not used
        if self.include_salary:
            p_salary = p.get("current_salary")
            r_salary = req.get("median_salary")
            if p_salary and r_salary and r_salary > 0:
                salary_util = min(1.0, float(p_salary) / float(r_salary))

        weights = self.WEIGHT
        if not self.include_salary:
            # Redistribute salary weight equally to other dimensions
            extra = weights["salary_utilization"] / 3
            adjusted = {
                "skill_utilization_rate": weights["skill_utilization_rate"] + extra,
                "cert_utilization_rate":  weights["cert_utilization_rate"]  + extra,
                "role_level_match":       weights["role_level_match"]       + extra,
                "salary_utilization":     0.0,
            }
        else:
            adjusted = weights

        composite = (
            skill_util   * adjusted["skill_utilization_rate"]
            + cert_util  * adjusted["cert_utilization_rate"]
            + role_match * adjusted["role_level_match"]
            + salary_util * adjusted["salary_utilization"]
        ) * 100

        return {
            "participant_id":         p.get("participant_id"),
            "current_job_role":       p.get("current_job_role"),
            "current_industry":       p.get("current_industry"),
            "skill_utilization_rate": round(skill_util, 4),
            "cert_utilization_rate":  round(cert_util, 4),
            "role_level_match":       round(role_match, 4),
            "salary_utilization":     round(salary_util, 4) if self.include_salary else None,
            "utilization_score":      round(composite, 2),
            "utilization_band":       self._band(composite),
            "unused_skills":          list(p_skills - r_skills),
            "unused_certifications":  list(p_certs - r_certs),
        }

    @staticmethod
    def _band(score: float) -> str:
        if score >= 75:
            return "Highly Utilised"
        elif score >= 50:
            return "Well Utilised"
        elif score >= 25:
            return "Partially Utilised"
        else:
            return "Under-utilised"

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return descriptive statistics for utilization scores."""
        self._check_computed()
        s = self.scores_["utilization_score"]
        return {
            "mean":   round(s.mean(), 2),
            "median": round(s.median(), 2),
            "std":    round(s.std(), 2),
            "min":    round(s.min(), 2),
            "max":    round(s.max(), 2),
            "band_distribution": self.scores_["utilization_band"].value_counts().to_dict(),
        }

    def get_underutilised(self, threshold: float = 25.0) -> pd.DataFrame:
        """Return participants with utilization_score below threshold."""
        self._check_computed()
        result = self.scores_[self.scores_["utilization_score"] < threshold]
        logger.info(f"{len(result)} participants under-utilised (score < {threshold}).")
        return result.reset_index(drop=True)

    def get_most_unused_skills(self, top_n: int = 10) -> pd.DataFrame:
        """Return the skills most frequently unused across all participants."""
        self._check_computed()
        from collections import Counter
        all_unused: List[str] = []
        for skills in self.scores_["unused_skills"]:
            all_unused.extend(skills)
        freq = Counter(all_unused)
        return (
            pd.DataFrame(freq.items(), columns=["skill", "unused_count"])
            .sort_values("unused_count", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def get_score_by_industry(self) -> pd.DataFrame:
        """Return average utilization score grouped by industry."""
        self._check_computed()
        return (
            self.scores_
            .groupby("current_industry")["utilization_score"]
            .agg(["mean", "median", "count"])
            .round(2)
            .rename(columns={"mean": "avg_util", "median": "median_util", "count": "n"})
            .sort_values("avg_util", ascending=False)
            .reset_index()
        )

    def get_score_by_role(self) -> pd.DataFrame:
        """Return average utilization score grouped by job role."""
        self._check_computed()
        return (
            self.scores_
            .groupby("current_job_role")["utilization_score"]
            .agg(["mean", "count"])
            .round(2)
            .rename(columns={"mean": "avg_util", "count": "n"})
            .sort_values("avg_util", ascending=False)
            .reset_index()
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(self, output_path: str) -> None:
        """Save utilization scores to CSV."""
        self._check_computed()
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        export = self.scores_.copy()
        export["unused_skills"] = export["unused_skills"].apply(
            lambda x: "|".join(x) if x else ""
        )
        export["unused_certifications"] = export["unused_certifications"].apply(
            lambda x: "|".join(x) if x else ""
        )
        export.to_csv(output_path, index=False)
        logger.info(f"Utilization scores saved → {output_path}")

    def _check_computed(self) -> None:
        if self.scores_ is None:
            raise RuntimeError("No scores computed. Call compute() first.")


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    participants = pd.DataFrame([
        {
            "participant_id": "P001",
            "skills": ["python", "sql", "machine learning", "excel"],
            "certifications": ["SQL Cert", "ML Cert"],
            "years_experience": 3,
            "current_job_role": "Data Entry Clerk",
            "current_industry": "finance",
            "current_salary": 28000,
        },
        {
            "participant_id": "P002",
            "skills": ["logistics", "communication", "project management"],
            "certifications": ["Logistics Cert"],
            "years_experience": 5,
            "current_job_role": "Logistics Coordinator",
            "current_industry": "supply chain",
            "current_salary": 45000,
        },
    ])

    role_reqs = pd.DataFrame([
        {
            "job_role": "Data Entry Clerk",
            "required_skills": ["excel", "typing"],
            "required_certifications": [],
            "expected_min_years_exp": 0,
            "median_salary": 25000,
        },
        {
            "job_role": "Logistics Coordinator",
            "required_skills": ["logistics", "communication", "project management"],
            "required_certifications": ["Logistics Cert"],
            "expected_min_years_exp": 3,
            "median_salary": 42000,
        },
    ])

    scorer = UtilizationScore(participants, role_reqs, include_salary=True)
    results = scorer.compute()

    print(results[["participant_id", "utilization_score", "utilization_band", "unused_skills"]])
    print("\nSummary:", scorer.get_summary())
    print("\nMost unused skills:\n", scorer.get_most_unused_skills())
