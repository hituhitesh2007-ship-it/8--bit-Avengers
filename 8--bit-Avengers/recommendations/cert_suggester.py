# cert_suggester.py
# TODO: implement
# recommendations/cert_suggester.py
# Suggests certifications based on skill gaps, barrier type, and market demand

import pandas as pd
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Static certification catalogue
CERTIFICATION_CATALOGUE = [
    {"cert_id": "C001", "name": "Google Data Analytics Certificate",
     "skills_covered": ["data analysis", "sql", "excel", "statistics"],
     "cost_inr": 3000, "duration_weeks": 6, "demand_level": "very high",
     "provider": "Google / Coursera", "industry": "data science"},

    {"cert_id": "C002", "name": "AWS Cloud Practitioner",
     "skills_covered": ["cloud computing", "aws", "networking"],
     "cost_inr": 8000, "duration_weeks": 4, "demand_level": "very high",
     "provider": "Amazon", "industry": "software engineering"},

    {"cert_id": "C003", "name": "PMP - Project Management Professional",
     "skills_covered": ["project management", "leadership", "communication"],
     "cost_inr": 25000, "duration_weeks": 12, "demand_level": "high",
     "provider": "PMI", "industry": "operations"},

    {"cert_id": "C004", "name": "TensorFlow Developer Certificate",
     "skills_covered": ["python", "deep learning", "machine learning"],
     "cost_inr": 10000, "duration_weeks": 8, "demand_level": "very high",
     "provider": "Google", "industry": "data science"},

    {"cert_id": "C005", "name": "HubSpot Digital Marketing",
     "skills_covered": ["communication", "research", "digital marketing"],
     "cost_inr": 0, "duration_weeks": 3, "demand_level": "high",
     "provider": "HubSpot Academy", "industry": "marketing"},

    {"cert_id": "C006", "name": "SQL for Data Science",
     "skills_covered": ["sql", "data analysis", "statistics"],
     "cost_inr": 1500, "duration_weeks": 4, "demand_level": "high",
     "provider": "UC Davis / Coursera", "industry": "data science"},

    {"cert_id": "C007", "name": "Python for Everybody",
     "skills_covered": ["python", "programming", "data analysis"],
     "cost_inr": 2000, "duration_weeks": 8, "demand_level": "very high",
     "provider": "University of Michigan / Coursera", "industry": "software engineering"},

    {"cert_id": "C008", "name": "Scrum Master Certification",
     "skills_covered": ["project management", "agile", "leadership"],
     "cost_inr": 15000, "duration_weeks": 2, "demand_level": "high",
     "provider": "Scrum Alliance", "industry": "operations"},
]

DEMAND_SCORE_MAP = {"very high": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}


class CertSuggester:
    """
    Suggests certifications to participants based on:
    - Skill gap between current skills and career path requirements
    - Market demand for the certification
    - Cost and duration constraints
    - Barrier type (e.g. credential_mismatch → high-recognition certs)
    """

    def __init__(self, catalogue: list = None, top_n: int = 3):
        self.catalogue = catalogue or CERTIFICATION_CATALOGUE
        self.top_n = top_n
        self.catalogue_df = pd.DataFrame(self.catalogue)
        self.mlb = MultiLabelBinarizer()
        self.cert_matrix = self.mlb.fit_transform(
            self.catalogue_df["skills_covered"]
        )

    # ------------------------------------------------------------------
    # Suggestion
    # ------------------------------------------------------------------

    def suggest(
        self,
        participant_id: str,
        current_skills: list,
        target_skills: list = None,
        barrier_label: str = None,
        max_cost_inr: float = None,
        max_duration_weeks: int = None
    ) -> pd.DataFrame:
        """
        Suggest top N certifications for a participant.

        Args:
            participant_id:     Unique participant ID
            current_skills:     Skills the participant already has
            target_skills:      Skills they need (from career path gap)
            barrier_label:      Detected barrier label
            max_cost_inr:       Optional budget constraint
            max_duration_weeks: Optional time constraint
        """
        current_skills = [s.strip().lower() for s in current_skills]
        target_skills = [s.strip().lower() for s in (target_skills or [])]

        df = self.catalogue_df.copy()

        # Apply filters
        if max_cost_inr is not None:
            df = df[df["cost_inr"] <= max_cost_inr]
        if max_duration_weeks is not None:
            df = df[df["duration_weeks"] <= max_duration_weeks]

        if df.empty:
            logger.warning("No certifications match the given constraints.")
            return pd.DataFrame()

        # Recompute matrix for filtered certs
        filtered_matrix = self.mlb.transform(df["skills_covered"])

        # Skill gap vector — target skills not yet held
        gap_skills = list(set(target_skills) - set(current_skills)) if target_skills \
            else current_skills
        gap_vector = self.mlb.transform([gap_skills]) if gap_skills \
            else self.mlb.transform([[]])

        # Cosine similarity to gap
        gap_scores = cosine_similarity(gap_vector, filtered_matrix)[0]

        # Demand score
        demand_scores = df["demand_level"].map(
            DEMAND_SCORE_MAP
        ).fillna(0.5).values

        # Cost efficiency (cheaper = higher score, normalized)
        max_cost = df["cost_inr"].max() or 1
        cost_scores = 1 - (df["cost_inr"] / max_cost).values

        # Duration efficiency (shorter = higher score)
        max_dur = df["duration_weeks"].max() or 1
        duration_scores = 1 - (df["duration_weeks"] / max_dur).values

        # Barrier-based weight adjustment
        barrier_boost = self._get_barrier_boost(df, barrier_label)

        # Composite score
        composite = (
            gap_scores * 0.40 +
            demand_scores * 0.25 +
            cost_scores * 0.15 +
            duration_scores * 0.10 +
            barrier_boost * 0.10
        )

        result = df.copy()
        result["participant_id"] = participant_id
        result["gap_match_score"] = gap_scores.round(4)
        result["composite_score"] = composite.round(4)

        # Skills this cert adds to the participant
        result["new_skills_gained"] = result["skills_covered"].apply(
            lambda covered: list(set(covered) - set(current_skills))
        )

        top = result.sort_values(
            "composite_score", ascending=False
        ).head(self.top_n).reset_index(drop=True)

        logger.info(
            f"Suggested {len(top)} certifications for participant {participant_id}"
        )
        return top[[
            "participant_id", "cert_id", "name", "provider",
            "industry", "cost_inr", "duration_weeks",
            "demand_level", "gap_match_score",
            "composite_score", "new_skills_gained"
        ]]

    def suggest_batch(self, participants_df: pd.DataFrame) -> pd.DataFrame:
        """Batch certification suggestions for multiple participants."""
        all_suggestions = []
        for _, row in participants_df.iterrows():
            skills = row.get("skills", [])
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",")]

            target = row.get("target_skills", [])
            if isinstance(target, str):
                target = [s.strip() for s in target.split(",")]

            suggestions = self.suggest(
                participant_id=row["participant_id"],
                current_skills=skills,
                target_skills=target,
                barrier_label=row.get("predicted_barrier"),
                max_cost_inr=row.get("max_budget"),
                max_duration_weeks=row.get("max_weeks")
            )
            all_suggestions.append(suggestions)

        return pd.concat(all_suggestions, ignore_index=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_barrier_boost(self, df: pd.DataFrame, barrier_label: str) -> float:
        """Boost certain certs based on barrier type."""
        boost = pd.Series([0.0] * len(df))

        if barrier_label == "credential_mismatch":
            # Prefer certs from well-known providers
            known = ["google", "amazon", "microsoft", "pmi"]
            boost = df["provider"].str.lower().apply(
                lambda p: 0.2 if any(k in p for k in known) else 0.0
            ).values

        elif barrier_label == "skill_decay":
            # Prefer shorter, refresher-style certs
            boost = (df["duration_weeks"] <= 4).astype(float).values * 0.15

        elif barrier_label == "economic_instability":
            # Prefer free or very cheap certs
            boost = (df["cost_inr"] <= 2000).astype(float).values * 0.20

        return boost

    def get_free_certs(self) -> pd.DataFrame:
        """Return all free certifications in the catalogue."""
        return self.catalogue_df[self.catalogue_df["cost_inr"] == 0].reset_index(drop=True)

    def get_summary(self) -> dict:
        """Return summary of the certification catalogue."""
        return {
            "total_certs": len(self.catalogue_df),
            "free_certs": int((self.catalogue_df["cost_inr"] == 0).sum()),
            "avg_cost_inr": round(self.catalogue_df["cost_inr"].mean(), 2),
            "avg_duration_weeks": round(self.catalogue_df["duration_weeks"].mean(), 1),
            "industries_covered": self.catalogue_df["industry"].nunique()
        }


if __name__ == "__main__":
    suggester = CertSuggester(top_n=3)
    suggestions = suggester.suggest(
        participant_id="P001",
        current_skills=["excel", "communication"],
        target_skills=["python", "sql", "data analysis"],
        barrier_label="credential_mismatch",
        max_cost_inr=10000
    )
    print(suggestions[["name", "provider", "cost_inr", "composite_score", "new_skills_gained"]])
    print(suggester.get_summary())
