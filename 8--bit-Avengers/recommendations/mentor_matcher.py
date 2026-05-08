# mentor_matcher.py
# TODO: implement
# recommendations/mentor_matcher.py
# Matches participants to mentors based on skill alignment, region, and industry

import pandas as pd
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MentorMatcher:
    """
    Matches participants to the most compatible mentors using:
    - Skill overlap (cosine similarity on skill vectors)
    - Region proximity
    - Industry alignment
    - Mentor availability and capacity
    """

    def __init__(self, top_n: int = 3):
        """
        Args:
            top_n: Number of mentor recommendations per participant
        """
        self.top_n = top_n
        self.mentor_df = None
        self.mlb = MultiLabelBinarizer()
        self.skill_matrix = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_mentors(self, mentor_df: pd.DataFrame) -> None:
        """
        Load mentor profiles.

        Required columns:
        - mentor_id, name, industry, region, skills (list or comma-sep string),
          available_slots, years_experience
        """
        required = ["mentor_id", "industry", "region", "skills"]
        missing = [c for c in required if c not in mentor_df.columns]
        if missing:
            raise ValueError(f"Mentor DataFrame missing columns: {missing}")

        self.mentor_df = mentor_df.copy()

        # Parse skills to list if string
        self.mentor_df["skills"] = self.mentor_df["skills"].apply(
            lambda x: [s.strip().lower() for s in x.split(",")]
            if isinstance(x, str) else x
        )

        # Build skill matrix
        self.skill_matrix = self.mlb.fit_transform(self.mentor_df["skills"])
        logger.info(f"Loaded {len(self.mentor_df)} mentor profiles.")

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(
        self,
        participant_id: str,
        participant_skills: list,
        participant_industry: str,
        participant_region: str
    ) -> pd.DataFrame:
        """
        Find the top N mentors for a participant.

        Args:
            participant_id:       Unique ID
            participant_skills:   List of participant's skills
            participant_industry: Participant's target industry
            participant_region:   Participant's region

        Returns:
            DataFrame of top N mentor matches with scores
        """
        if self.mentor_df is None:
            raise RuntimeError("Mentors not loaded. Call load_mentors() first.")

        # Vectorize participant skills
        participant_vector = self.mlb.transform([[
            s.strip().lower() for s in participant_skills
        ]])

        # Cosine similarity on skills
        skill_scores = cosine_similarity(participant_vector, self.skill_matrix)[0]

        # Region and industry bonus
        region_bonus = (
            self.mentor_df["region"].str.lower() == participant_region.lower()
        ).astype(float) * 0.15

        industry_bonus = (
            self.mentor_df["industry"].str.lower() == participant_industry.lower()
        ).astype(float) * 0.15

        # Availability bonus
        if "available_slots" in self.mentor_df.columns:
            avail_bonus = (
                self.mentor_df["available_slots"] > 0
            ).astype(float) * 0.10
        else:
            avail_bonus = 0.0

        # Experience bonus (normalized)
        if "years_experience" in self.mentor_df.columns:
            max_exp = self.mentor_df["years_experience"].max()
            exp_bonus = (self.mentor_df["years_experience"] / max_exp) * 0.10
        else:
            exp_bonus = 0.0

        # Final composite score
        final_scores = skill_scores + region_bonus + industry_bonus + avail_bonus + exp_bonus

        results = self.mentor_df.copy()
        results["skill_similarity"] = skill_scores.round(4)
        results["match_score"] = final_scores.round(4)
        results["participant_id"] = participant_id

        # Filter only available mentors
        if "available_slots" in results.columns:
            results = results[results["available_slots"] > 0]

        top_matches = results.sort_values(
            "match_score", ascending=False
        ).head(self.top_n).reset_index(drop=True)

        logger.info(
            f"Matched participant {participant_id} to "
            f"{len(top_matches)} mentors."
        )
        return top_matches[
            ["participant_id", "mentor_id", "industry", "region",
             "skill_similarity", "match_score"]
            + (["name"] if "name" in top_matches.columns else [])
            + (["available_slots"] if "available_slots" in top_matches.columns else [])
        ]

    def match_batch(
        self,
        participants_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Match mentors for a full batch of participants.

        Required columns in participants_df:
        - participant_id, skills, industry, region
        """
        required = ["participant_id", "skills", "industry", "region"]
        missing = [c for c in required if c not in participants_df.columns]
        if missing:
            raise ValueError(f"Participants DataFrame missing: {missing}")

        all_matches = []
        for _, row in participants_df.iterrows():
            skills = row["skills"]
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",")]

            matches = self.match(
                participant_id=row["participant_id"],
                participant_skills=skills,
                participant_industry=row["industry"],
                participant_region=row["region"]
            )
            all_matches.append(matches)

        return pd.concat(all_matches, ignore_index=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_mentor_load(self) -> pd.DataFrame:
        """Return how many participants each mentor has been matched to."""
        if self.mentor_df is None:
            raise RuntimeError("Mentors not loaded.")
        return self.mentor_df[["mentor_id"]].copy()

    def get_summary(self) -> dict:
        """Return mentor pool summary."""
        if self.mentor_df is None:
            raise RuntimeError("Mentors not loaded.")
        return {
            "total_mentors": len(self.mentor_df),
            "industries_covered": self.mentor_df["industry"].nunique(),
            "regions_covered": self.mentor_df["region"].nunique(),
            "avg_available_slots": round(
                self.mentor_df["available_slots"].mean(), 1
            ) if "available_slots" in self.mentor_df.columns else "N/A"
        }


if __name__ == "__main__":
    mentors = pd.DataFrame([
        {"mentor_id": "M001", "name": "Anjali R", "industry": "data science",
         "region": "karnataka", "skills": "python, ml, statistics", "available_slots": 3,
         "years_experience": 8},
        {"mentor_id": "M002", "name": "Ravi K", "industry": "software engineering",
         "region": "karnataka", "skills": "java, python, sql", "available_slots": 2,
         "years_experience": 5},
        {"mentor_id": "M003", "name": "Priya S", "industry": "data science",
         "region": "tamil nadu", "skills": "python, deep learning, nlp", "available_slots": 1,
         "years_experience": 10},
    ])

    matcher = MentorMatcher(top_n=2)
    matcher.load_mentors(mentors)

    matches = matcher.match(
        participant_id="P001",
        participant_skills=["python", "ml"],
        participant_industry="data science",
        participant_region="karnataka"
    )
    print(matches)
    print(matcher.get_summary())
