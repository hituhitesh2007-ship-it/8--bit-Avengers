# internship_finder.py
# TODO: implement
# recommendations/internship_finder.py
# Finds and ranks internship opportunities for participants based on skill fit

import pandas as pd
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTERNSHIP_DATABASE = [
    {"internship_id": "I001", "title": "Data Science Intern",
     "required_skills": ["python", "sql", "statistics"],
     "company": "Analytics Corp", "industry": "data science",
     "region": "karnataka", "stipend_monthly_inr": 15000,
     "duration_months": 3, "mode": "hybrid"},

    {"internship_id": "I002", "title": "Software Development Intern",
     "required_skills": ["python", "javascript", "sql"],
     "company": "TechStart India", "industry": "software engineering",
     "region": "karnataka", "stipend_monthly_inr": 12000,
     "duration_months": 6, "mode": "remote"},

    {"internship_id": "I003", "title": "Business Analyst Intern",
     "required_skills": ["excel", "communication", "research"],
     "company": "Deloitte", "industry": "consulting",
     "region": "maharashtra", "stipend_monthly_inr": 20000,
     "duration_months": 3, "mode": "onsite"},

    {"internship_id": "I004", "title": "Digital Marketing Intern",
     "required_skills": ["communication", "research", "excel"],
     "company": "GrowthLabs", "industry": "marketing",
     "region": "karnataka", "stipend_monthly_inr": 8000,
     "duration_months": 2, "mode": "remote"},

    {"internship_id": "I005", "title": "ML Research Intern",
     "required_skills": ["machine learning", "python", "deep learning"],
     "company": "AI Research Lab", "industry": "data science",
     "region": "tamil nadu", "stipend_monthly_inr": 25000,
     "duration_months": 6, "mode": "onsite"},

    {"internship_id": "I006", "title": "Project Coordinator Intern",
     "required_skills": ["project management", "communication", "excel"],
     "company": "BuildCo", "industry": "operations",
     "region": "karnataka", "stipend_monthly_inr": 10000,
     "duration_months": 3, "mode": "hybrid"},
]


class InternshipFinder:
    """
    Finds internship opportunities for participants based on:
    - Skill match
    - Region preference
    - Industry alignment
    - Stipend and duration constraints
    - Work mode preference
    """

    def __init__(self, internships: list = None, top_n: int = 3):
        self.internships = internships or INTERNSHIP_DATABASE
        self.top_n = top_n
        self.internship_df = pd.DataFrame(self.internships)
        self.mlb = MultiLabelBinarizer()
        self.internship_matrix = self.mlb.fit_transform(
            self.internship_df["required_skills"]
        )

    def find(
        self,
        participant_id: str,
        current_skills: list,
        preferred_region: str = None,
        preferred_industry: str = None,
        preferred_mode: str = None,
        min_stipend: float = None,
        max_duration_months: int = None
    ) -> pd.DataFrame:
        """
        Find top N internships for a participant.

        Args:
            participant_id:       Unique participant ID
            current_skills:       List of participant's skills
            preferred_region:     Region preference
            preferred_industry:   Industry preference
            preferred_mode:       'remote', 'onsite', or 'hybrid'
            min_stipend:          Minimum monthly stipend in INR
            max_duration_months:  Maximum internship duration in months
        """
        current_skills = [s.strip().lower() for s in current_skills]
        df = self.internship_df.copy()

        # Apply hard filters
        if min_stipend:
            df = df[df["stipend_monthly_inr"] >= min_stipend]
        if max_duration_months:
            df = df[df["duration_months"] <= max_duration_months]
        if preferred_mode:
            df = df[df["mode"].str.lower() == preferred_mode.lower()]

        if df.empty:
            logger.warning("No internships match the given constraints.")
            return pd.DataFrame()

        filtered_matrix = self.mlb.transform(df["required_skills"])
        participant_vector = self.mlb.transform([current_skills])

        # Skill similarity
        skill_scores = cosine_similarity(participant_vector, filtered_matrix)[0]

        # Region match bonus
        region_bonus = np.zeros(len(df)) if preferred_region is None else (
            (df["region"].str.lower() == preferred_region.lower()).astype(float) * 0.15
        ).values

        # Industry match bonus
        industry_bonus = np.zeros(len(df)) if preferred_industry is None else (
            (df["industry"].str.lower() == preferred_industry.lower()).astype(float) * 0.15
        ).values

        # Stipend score (normalized)
        max_stipend = df["stipend_monthly_inr"].max() or 1
        stipend_scores = (df["stipend_monthly_inr"] / max_stipend).values * 0.10

        composite = skill_scores * 0.55 + region_bonus + industry_bonus + stipend_scores

        result = df.copy()
        result["participant_id"] = participant_id
        result["skill_match"] = skill_scores.round(4)
        result["composite_score"] = composite.round(4)
        result["missing_skills"] = result["required_skills"].apply(
            lambda req: list(set(req) - set(current_skills))
        )

        top = result.sort_values(
            "composite_score", ascending=False
        ).head(self.top_n).reset_index(drop=True)

        logger.info(
            f"Found {len(top)} internships for participant {participant_id}"
        )
        return top[[
            "participant_id", "internship_id", "title", "company",
            "industry", "region", "stipend_monthly_inr",
            "duration_months", "mode", "skill_match",
            "composite_score", "missing_skills"
        ]]

    def find_batch(self, participants_df: pd.DataFrame) -> pd.DataFrame:
        """Batch internship finding for all participants."""
        all_results = []
        for _, row in participants_df.iterrows():
            skills = row.get("skills", [])
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",")]

            result = self.find(
                participant_id=row["participant_id"],
                current_skills=skills,
                preferred_region=row.get("region"),
                preferred_industry=row.get("industry")
            )
            all_results.append(result)

        return pd.concat(all_results, ignore_index=True)

    def get_remote_internships(self) -> pd.DataFrame:
        """Return all remote internships."""
        return self.internship_df[
            self.internship_df["mode"] == "remote"
        ].reset_index(drop=True)

    def get_summary(self) -> dict:
        return {
            "total_internships": len(self.internship_df),
            "regions": self.internship_df["region"].unique().tolist(),
            "industries": self.internship_df["industry"].unique().tolist(),
            "avg_stipend_monthly_inr": round(
                self.internship_df["stipend_monthly_inr"].mean(), 0
            ),
            "modes": self.internship_df["mode"].value_counts().to_dict()
        }


import numpy as np

if __name__ == "__main__":
    finder = InternshipFinder(top_n=3)
    results = finder.find(
        participant_id="P001",
        current_skills=["python", "sql", "excel"],
        preferred_region="karnataka",
        preferred_industry="data science"
    )
    print(results[["title", "company", "stipend_monthly_inr",
                    "composite_score", "missing_skills"]])
    print(finder.get_summary())
