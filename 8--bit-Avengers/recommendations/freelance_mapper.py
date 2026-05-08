# freelance_mapper.py
# TODO: implement
# recommendations/freelance_mapper.py
# Maps participant skills to freelance opportunities and platforms

import pandas as pd
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FREELANCE_OPPORTUNITIES = [
    {"opp_id": "F001", "title": "Data Entry & Analysis",
     "required_skills": ["excel", "data analysis"],
     "platform": "Upwork", "avg_hourly_rate_inr": 500,
     "difficulty": "low", "demand": "high"},

    {"opp_id": "F002", "title": "Python Automation Scripts",
     "required_skills": ["python", "programming"],
     "platform": "Fiverr", "avg_hourly_rate_inr": 1200,
     "difficulty": "medium", "demand": "very high"},

    {"opp_id": "F003", "title": "SQL Database Queries",
     "required_skills": ["sql", "data analysis"],
     "platform": "Toptal", "avg_hourly_rate_inr": 1500,
     "difficulty": "medium", "demand": "high"},

    {"opp_id": "F004", "title": "Content Research & Writing",
     "required_skills": ["research", "communication"],
     "platform": "Fiverr", "avg_hourly_rate_inr": 400,
     "difficulty": "low", "demand": "high"},

    {"opp_id": "F005", "title": "Social Media Management",
     "required_skills": ["communication", "digital marketing"],
     "platform": "PeoplePerHour", "avg_hourly_rate_inr": 600,
     "difficulty": "low", "demand": "medium"},

    {"opp_id": "F006", "title": "ML Model Consulting",
     "required_skills": ["machine learning", "python", "statistics"],
     "platform": "Toptal", "avg_hourly_rate_inr": 3000,
     "difficulty": "high", "demand": "very high"},

    {"opp_id": "F007", "title": "Excel Dashboard Creation",
     "required_skills": ["excel", "data analysis", "communication"],
     "platform": "Upwork", "avg_hourly_rate_inr": 700,
     "difficulty": "low", "demand": "high"},

    {"opp_id": "F008", "title": "Online Teaching / Tutoring",
     "required_skills": ["teaching", "communication"],
     "platform": "Chegg / Vedantu", "avg_hourly_rate_inr": 500,
     "difficulty": "low", "demand": "high"},
]

DEMAND_MAP = {"very high": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}
DIFFICULTY_MAP = {"low": 1.0, "medium": 0.7, "high": 0.4}


class FreelanceMapper:
    """
    Maps participants to freelance opportunities they can pursue
    immediately based on existing skills, with income potential scoring.
    """

    def __init__(self, opportunities: list = None, top_n: int = 3):
        self.opportunities = opportunities or FREELANCE_OPPORTUNITIES
        self.top_n = top_n
        self.opp_df = pd.DataFrame(self.opportunities)
        self.mlb = MultiLabelBinarizer()
        self.opp_matrix = self.mlb.fit_transform(self.opp_df["required_skills"])

    def map(
        self,
        participant_id: str,
        current_skills: list,
        barrier_label: str = None,
        min_hourly_rate: float = None
    ) -> pd.DataFrame:
        """
        Map participant to freelance opportunities.

        Args:
            participant_id:   Unique participant ID
            current_skills:   List of participant's current skills
            barrier_label:    Detected barrier (adjusts difficulty preference)
            min_hourly_rate:  Minimum acceptable hourly rate in INR
        """
        current_skills = [s.strip().lower() for s in current_skills]

        df = self.opp_df.copy()

        if min_hourly_rate:
            df = df[df["avg_hourly_rate_inr"] >= min_hourly_rate]

        if df.empty:
            logger.warning("No freelance opportunities match the given constraints.")
            return pd.DataFrame()

        filtered_matrix = self.mlb.transform(df["required_skills"])
        participant_vector = self.mlb.transform([current_skills])

        skill_scores = cosine_similarity(participant_vector, filtered_matrix)[0]
        demand_scores = df["demand"].map(DEMAND_MAP).fillna(0.5).values

        # For economic/confidence barriers, prefer easy entry gigs
        diff_modifier = 1.3 if barrier_label in [
            "economic_instability", "confidence_deficit"
        ] else 1.0
        difficulty_scores = df["difficulty"].map(
            DIFFICULTY_MAP
        ).fillna(0.5).values * diff_modifier

        # Income potential (normalized)
        max_rate = df["avg_hourly_rate_inr"].max() or 1
        income_scores = (df["avg_hourly_rate_inr"] / max_rate).values

        composite = (
            skill_scores * 0.45 +
            demand_scores * 0.25 +
            difficulty_scores * 0.15 +
            income_scores * 0.15
        )

        result = df.copy()
        result["participant_id"] = participant_id
        result["skill_match"] = skill_scores.round(4)
        result["composite_score"] = composite.round(4)
        result["missing_skills"] = result["required_skills"].apply(
            lambda req: list(set(req) - set(current_skills))
        )
        result["ready_to_apply"] = result["missing_skills"].apply(
            lambda m: len(m) == 0
        )

        top = result.sort_values(
            "composite_score", ascending=False
        ).head(self.top_n).reset_index(drop=True)

        logger.info(
            f"Mapped {len(top)} freelance opportunities for participant {participant_id}"
        )
        return top[[
            "participant_id", "opp_id", "title", "platform",
            "avg_hourly_rate_inr", "difficulty", "demand",
            "skill_match", "composite_score",
            "missing_skills", "ready_to_apply"
        ]]

    def map_batch(self, participants_df: pd.DataFrame) -> pd.DataFrame:
        """Batch freelance mapping for all participants."""
        all_results = []
        for _, row in participants_df.iterrows():
            skills = row.get("skills", [])
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",")]

            result = self.map(
                participant_id=row["participant_id"],
                current_skills=skills,
                barrier_label=row.get("predicted_barrier")
            )
            all_results.append(result)

        return pd.concat(all_results, ignore_index=True)

    def get_immediately_applicable(
        self, participant_id: str, current_skills: list
    ) -> pd.DataFrame:
        """Return only opportunities the participant can apply to right now."""
        all_opps = self.map(participant_id, current_skills)
        return all_opps[all_opps["ready_to_apply"] == True].reset_index(drop=True)

    def get_summary(self) -> dict:
        return {
            "total_opportunities": len(self.opp_df),
            "platforms": self.opp_df["platform"].unique().tolist(),
            "avg_hourly_rate_inr": round(self.opp_df["avg_hourly_rate_inr"].mean(), 0),
            "high_demand_count": int(
                self.opp_df["demand"].isin(["very high", "high"]).sum()
            )
        }


if __name__ == "__main__":
    mapper = FreelanceMapper(top_n=3)
    results = mapper.map(
        participant_id="P001",
        current_skills=["python", "excel", "communication"],
        barrier_label="economic_instability"
    )
    print(results[["title", "platform", "avg_hourly_rate_inr",
                    "composite_score", "ready_to_apply"]])
