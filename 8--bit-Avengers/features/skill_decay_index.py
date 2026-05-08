# skill_decay_index.py
# TODO: implement
# features/skill_decay_index.py
# Computes a skill decay index — how stale are a participant's skills?

import pandas as pd
import numpy as np
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Half-life of skills in years (domain-specific decay rates)
# Based on workforce development and labor economics research
SKILL_HALF_LIFE_YEARS: Dict[str, float] = {
    # Technical skills decay fast
    "python": 2.5,
    "javascript": 2.0,
    "machine learning": 2.0,
    "data analysis": 3.0,
    "sql": 4.0,
    "java": 3.5,
    "excel": 5.0,
    # Domain skills decay moderately
    "accounting": 5.0,
    "logistics": 4.0,
    "graphic design": 3.0,
    "nursing": 3.0,
    "welding": 6.0,
    # Soft skills are relatively stable
    "communication": 10.0,
    "leadership": 10.0,
    "project management": 6.0,
    "customer service": 8.0,
    "research": 7.0,
    "teaching": 8.0,
    # Default for unknown skills
    "_default": 4.0,
}


class SkillDecayIndexEngineer:
    """
    Computes a Skill Decay Index (SDI) for each participant.

    Skills depreciate over time if not actively used or refreshed.
    The SDI estimates how much of a participant's skill portfolio
    remains current, given:
    - When each skill was acquired / last used
    - The inherent decay rate of each skill type
    - Whether the skill has been refreshed via recent training

    A high SDI (close to 1.0) means skills are current.
    A low SDI (close to 0.0) means skills are stale — intervention needed.

    Inputs:
        resume_df       : From ResumeParser (skills_mentioned, participant_id)
        certification_df: From CertificationLoader (skill, issue_date, expiry_date)
        employment_df   : From EmploymentConnector (last active role date)
    """

    def __init__(
        self,
        resume_df: pd.DataFrame,
        certification_df: pd.DataFrame = None,
        employment_df: pd.DataFrame = None,
        id_col: str = "participant_id",
    ):
        self.resume_df = resume_df.copy()
        self.certification_df = certification_df.copy() if certification_df is not None else None
        self.employment_df = employment_df.copy() if employment_df is not None else None
        self.id_col = id_col
        self.features = None
        self.today = pd.Timestamp.today()

    def build(self) -> pd.DataFrame:
        result = self._base_decay_from_resume()
        if self.certification_df is not None:
            result = result.merge(
                self._cert_refresh_bonus(), on=self.id_col, how="left"
            )
            result["skill_decay_index"] = (
                result["raw_decay_index"] + result["cert_refresh_bonus"].fillna(0)
            ).clip(0, 1)
        else:
            result["skill_decay_index"] = result["raw_decay_index"]

        result["skills_stale"] = (result["skill_decay_index"] < 0.5).astype(int)
        result["urgent_upskilling_needed"] = (result["skill_decay_index"] < 0.3).astype(int)

        self.features = result
        logger.info(f"Skill decay index features built: {result.shape}")
        return result

    def _base_decay_from_resume(self) -> pd.DataFrame:
        """
        Estimate skill freshness from last employment date or resume date.
        Uses exponential decay: retention = 0.5^(years_elapsed / half_life)
        """
        df = self.resume_df.copy()
        out_rows = []

        # Get last active date per participant from employment data if available
        last_active = {}
        if self.employment_df is not None and "employment_start_date" in self.employment_df.columns:
            self.employment_df["employment_start_date"] = pd.to_datetime(
                self.employment_df["employment_start_date"], errors="coerce"
            )
            last_active = (
                self.employment_df.groupby(self.id_col)["employment_start_date"]
                .max()
                .to_dict()
            )

        for _, row in df.iterrows():
            pid = row[self.id_col]
            skills = row.get("skills_mentioned", [])
            if not isinstance(skills, list):
                skills = []

            # Determine reference date for when skills were last used
            ref_date = last_active.get(pid, self.today - pd.DateOffset(years=2))
            years_elapsed = max((self.today - ref_date).days / 365.25, 0)

            if not skills:
                out_rows.append({self.id_col: pid, "raw_decay_index": 0.5,
                                  "num_skills_assessed": 0})
                continue

            decay_values = []
            for skill in skills:
                half_life = SKILL_HALF_LIFE_YEARS.get(
                    skill.lower(), SKILL_HALF_LIFE_YEARS["_default"]
                )
                retention = 0.5 ** (years_elapsed / half_life)
                decay_values.append(retention)

            out_rows.append({
                self.id_col: pid,
                "raw_decay_index": float(np.mean(decay_values)),
                "min_skill_retention": float(np.min(decay_values)),
                "max_skill_retention": float(np.max(decay_values)),
                "num_skills_assessed": len(decay_values),
                "years_since_last_role": round(years_elapsed, 2),
            })

        return pd.DataFrame(out_rows)

    def _cert_refresh_bonus(self) -> pd.DataFrame:
        """
        Recent certifications partially restore skill freshness.
        A cert issued within 1 year gives full bonus; bonus decays over 3 years.
        """
        cert_df = self.certification_df.copy()
        if "issue_date" not in cert_df.columns:
            return pd.DataFrame({self.id_col: cert_df[self.id_col].unique(),
                                  "cert_refresh_bonus": 0.0})

        cert_df["issue_date"] = pd.to_datetime(cert_df["issue_date"], errors="coerce")
        cert_df["years_since_cert"] = (
            (self.today - cert_df["issue_date"]).dt.days / 365.25
        ).clip(lower=0)

        # Bonus decays from 0.2 at issue to 0 at 3 years
        cert_df["cert_bonus"] = (0.2 * np.exp(-cert_df["years_since_cert"] / 3)).clip(0, 0.2)

        bonus = (
            cert_df.groupby(self.id_col)["cert_bonus"]
            .sum()
            .reset_index()
            .rename(columns={"cert_bonus": "cert_refresh_bonus"})
        )
        bonus["cert_refresh_bonus"] = bonus["cert_refresh_bonus"].clip(0, 0.3)
        return bonus

    def get_stale_skills_report(self) -> pd.DataFrame:
        """Return participants with stale skills ranked by urgency."""
        if self.features is None:
            raise RuntimeError("Call build() first.")
        return (
            self.features[self.features["skills_stale"] == 1]
            .sort_values("skill_decay_index")
            .reset_index(drop=True)
        )

    def get_features(self) -> pd.DataFrame:
        if self.features is None:
            raise RuntimeError("Call build() first.")
        return self.features

    def save(self, output_path: str) -> None:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.features.to_csv(output_path, index=False)
        logger.info(f"Saved skill decay index features to {output_path}")


if __name__ == "__main__":
    resumes = pd.read_csv("data/processed/resumes_parsed.csv")
    certs = pd.read_csv("data/processed/certifications_clean.csv")
    emp = pd.read_csv("data/processed/employment_clean.csv")

    engineer = SkillDecayIndexEngineer(resumes, certs, emp)
    df = engineer.build()
    print(df[["participant_id", "skill_decay_index", "skills_stale",
              "urgent_upskilling_needed"]].head(10))
    engineer.save("data/features/skill_decay_index.csv")
