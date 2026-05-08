# career_path_recommender.py
# TODO: implement
# recommendations/career_path_recommender.py
# Recommends career pathways based on current skills, barriers, and market demand

import pandas as pd
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Static career pathway definitions
CAREER_PATHS = [
    {"path_id": "CP001", "title": "Data Analyst",
     "required_skills": ["python", "sql", "excel", "statistics"],
     "industry": "data science", "avg_salary": 600000, "demand_level": "high",
     "entry_difficulty": "medium"},

    {"path_id": "CP002", "title": "ML Engineer",
     "required_skills": ["python", "machine learning", "deep learning", "sql"],
     "industry": "data science", "avg_salary": 1200000, "demand_level": "very high",
     "entry_difficulty": "high"},

    {"path_id": "CP003", "title": "Business Analyst",
     "required_skills": ["excel", "communication", "sql", "project management"],
     "industry": "consulting", "avg_salary": 700000, "demand_level": "high",
     "entry_difficulty": "low"},

    {"path_id": "CP004", "title": "Full Stack Developer",
     "required_skills": ["javascript", "python", "sql", "react"],
     "industry": "software engineering", "avg_salary": 900000, "demand_level": "very high",
     "entry_difficulty": "medium"},

    {"path_id": "CP005", "title": "Digital Marketing Specialist",
     "required_skills": ["communication", "excel", "research", "leadership"],
     "industry": "marketing", "avg_salary": 450000, "demand_level": "medium",
     "entry_difficulty": "low"},

    {"path_id": "CP006", "title": "Project Manager",
     "required_skills": ["project management", "communication", "leadership", "excel"],
     "industry": "operations", "avg_salary": 800000, "demand_level": "high",
     "entry_difficulty": "medium"},

    {"path_id": "CP007", "title": "Data Engineer",
     "required_skills": ["python", "sql", "java", "statistics"],
     "industry": "data science", "avg_salary": 1000000, "demand_level": "very high",
     "entry_difficulty": "high"},

    {"path_id": "CP008", "title": "UX Researcher",
     "required_skills": ["research", "communication", "teaching"],
     "industry": "design", "avg_salary": 550000, "demand_level": "medium",
     "entry_difficulty": "low"},
]


class CareerPathRecommender:
    """
    Recommends career pathways to participants based on:
    - Current skill match (cosine similarity)
    - Market demand level
    - Entry difficulty relative to participant profile
    - Salary potential
    - Industry alignment
    """

    DEMAND_WEIGHTS = {"very high": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}
    DIFFICULTY_WEIGHTS = {"low": 1.0, "medium": 0.7, "high": 0.4}

    def __init__(self, career_paths: list = None, top_n: int = 3):
        self.career_paths = career_paths or CAREER_PATHS
        self.top_n = top_n
        self.paths_df = pd.DataFrame(self.career_paths)
        self.mlb = MultiLabelBinarizer()

        # Fit on all career path required skills
        self.skill_matrix = self.mlb.fit_transform(
            self.paths_df["required_skills"]
        )

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        participant_id: str,
        current_skills: list,
        preferred_industry: str = None,
        barrier_label: str = None
    ) -> pd.DataFrame:
        """
        Recommend top N career paths for a participant.

        Args:
            participant_id:      Unique participant ID
            current_skills:      List of current skills
            preferred_industry:  Optional industry preference
            barrier_label:       Detected barrier (adjusts difficulty weight)

        Returns:
            DataFrame of top N recommended paths with scores
        """
        current_skills = [s.strip().lower() for s in current_skills]

        # Vectorize current skills
        participant_vector = self.mlb.transform([current_skills])

        # Skill similarity
        skill_scores = cosine_similarity(participant_vector, self.skill_matrix)[0]

        # Skill gap (% of required skills already held)
        skill_gap_ratios = []
        for path in self.career_paths:
            required = set(path["required_skills"])
            held = set(current_skills)
            ratio = len(held & required) / len(required) if required else 0
            skill_gap_ratios.append(ratio)

        skill_gap_ratios = np.array(skill_gap_ratios)

        # Demand score
        demand_scores = self.paths_df["demand_level"].map(
            self.DEMAND_WEIGHTS
        ).fillna(0.5).values

        # Difficulty score — for high-barrier participants favour easier paths
        difficulty_modifier = 1.0
        if barrier_label in ["confidence_deficit", "economic_instability"]:
            difficulty_modifier = 1.3  # boost easy paths

        difficulty_scores = self.paths_df["entry_difficulty"].map(
            self.DIFFICULTY_WEIGHTS
        ).fillna(0.5).values * difficulty_modifier

        # Industry alignment bonus
        industry_bonus = np.zeros(len(self.paths_df))
        if preferred_industry:
            industry_bonus = (
                self.paths_df["industry"].str.lower() == preferred_industry.lower()
            ).astype(float) * 0.15

        # Composite score
        composite = (
            skill_scores * 0.40 +
            skill_gap_ratios * 0.25 +
            demand_scores * 0.20 +
            difficulty_scores * 0.10 +
            industry_bonus * 0.05
        )

        results = self.paths_df.copy()
        results["participant_id"] = participant_id
        results["skill_similarity"] = skill_scores.round(4)
        results["skill_coverage"] = skill_gap_ratios.round(4)
        results["composite_score"] = composite.round(4)

        # Compute missing skills per path
        results["missing_skills"] = results["required_skills"].apply(
            lambda req: list(set(req) - set(current_skills))
        )

        top = results.sort_values(
            "composite_score", ascending=False
        ).head(self.top_n).reset_index(drop=True)

        logger.info(
            f"Recommended {len(top)} career paths for participant {participant_id}"
        )
        return top[[
            "participant_id", "path_id", "title", "industry",
            "demand_level", "entry_difficulty", "avg_salary",
            "skill_similarity", "skill_coverage",
            "composite_score", "missing_skills"
        ]]

    def recommend_batch(self, participants_df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch recommendations for all participants.

        Required columns: participant_id, skills
        Optional columns: industry, predicted_barrier
        """
        all_recs = []
        for _, row in participants_df.iterrows():
            skills = row["skills"]
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",")]

            recs = self.recommend(
                participant_id=row["participant_id"],
                current_skills=skills,
                preferred_industry=row.get("industry"),
                barrier_label=row.get("predicted_barrier")
            )
            all_recs.append(recs)

        return pd.concat(all_recs, ignore_index=True)

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------

    def get_most_recommended_paths(
        self, results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Return frequency of each recommended path across all participants."""
        return (
            results_df["title"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "path", "title": "count"})
        )

    def get_high_demand_paths(self) -> pd.DataFrame:
        """Return all paths with very high or high market demand."""
        return self.paths_df[
            self.paths_df["demand_level"].isin(["very high", "high"])
        ].reset_index(drop=True)


if __name__ == "__main__":
    recommender = CareerPathRecommender(top_n=3)
    recs = recommender.recommend(
        participant_id="P001",
        current_skills=["python", "sql", "excel"],
        preferred_industry="data science",
        barrier_label="confidence_deficit"
    )
    print(recs[["title", "composite_score", "missing_skills"]])
