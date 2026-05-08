# recommendations/intervention_engine.py
# Core engine that decides which intervention to recommend based on detected barrier

import pandas as pd
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Barrier-to-intervention mapping
INTERVENTION_MAP = {
    "no_barrier": [],
    "opportunity_gap": ["internship_finder", "freelance_mapper", "career_path_recommender"],
    "confidence_deficit": ["mentor_matcher", "cert_suggester", "career_path_recommender"],
    "credential_mismatch": ["cert_suggester", "career_path_recommender"],
    "network_isolation": ["mentor_matcher", "freelance_mapper"],
    "economic_instability": ["freelance_mapper", "internship_finder", "cert_suggester"],
    "skill_decay": ["cert_suggester", "career_path_recommender", "mentor_matcher"],
}

PRIORITY_WEIGHTS = {
    "mentor_matcher": 0.85,
    "career_path_recommender": 0.80,
    "cert_suggester": 0.75,
    "internship_finder": 0.70,
    "freelance_mapper": 0.65,
}


class InterventionEngine:
    """
    Master intervention router.
    Takes a participant's barrier label and profile,
    returns a ranked list of recommended interventions
    with confidence scores and reasoning.
    """

    def __init__(self, intervention_map: dict = None, priority_weights: dict = None):
        self.intervention_map = intervention_map or INTERVENTION_MAP
        self.priority_weights = priority_weights or PRIORITY_WEIGHTS

    # ------------------------------------------------------------------
    # Core Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        participant_id: str,
        barrier_label: str,
        profile: Optional[dict] = None
    ) -> list:
        """
        Generate ranked intervention recommendations for a participant.

        Args:
            participant_id: Unique participant ID
            barrier_label:  Detected barrier (from BarrierDetector)
            profile:        Optional dict of participant attributes
                            (engagement_score, region, num_certifications, etc.)

        Returns:
            List of dicts with intervention name, priority score, and reason
        """
        barrier_label = barrier_label.lower().strip()

        if barrier_label not in self.intervention_map:
            logger.warning(
                f"Unknown barrier label '{barrier_label}'. "
                f"Defaulting to general recommendations."
            )
            barrier_label = "opportunity_gap"

        interventions = self.intervention_map[barrier_label]

        if not interventions:
            logger.info(f"No interventions needed for participant {participant_id}.")
            return []

        ranked = []
        for intervention in interventions:
            base_score = self.priority_weights.get(intervention, 0.5)
            adjusted_score = self._adjust_score(base_score, intervention, profile)

            ranked.append({
                "participant_id": participant_id,
                "intervention": intervention,
                "priority_score": round(adjusted_score, 4),
                "barrier": barrier_label,
                "reason": self._get_reason(barrier_label, intervention)
            })

        # Sort by priority score descending
        ranked.sort(key=lambda x: x["priority_score"], reverse=True)

        logger.info(
            f"Generated {len(ranked)} interventions for "
            f"participant {participant_id} (barrier: {barrier_label})"
        )
        return ranked

    def recommend_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply intervention recommendations to a full DataFrame.

        Args:
            df: DataFrame with columns participant_id and predicted_barrier

        Returns:
            DataFrame with top intervention per participant appended
        """
        if "participant_id" not in df.columns or "predicted_barrier" not in df.columns:
            raise ValueError(
                "DataFrame must have 'participant_id' and 'predicted_barrier' columns."
            )

        results = []
        for _, row in df.iterrows():
            profile = row.to_dict()
            recommendations = self.recommend(
                participant_id=row["participant_id"],
                barrier_label=row["predicted_barrier"],
                profile=profile
            )
            if recommendations:
                top = recommendations[0]
                results.append({
                    "participant_id": row["participant_id"],
                    "top_intervention": top["intervention"],
                    "priority_score": top["priority_score"],
                    "barrier": top["barrier"],
                    "reason": top["reason"],
                    "all_interventions": [r["intervention"] for r in recommendations]
                })
            else:
                results.append({
                    "participant_id": row["participant_id"],
                    "top_intervention": "none",
                    "priority_score": 0.0,
                    "barrier": row["predicted_barrier"],
                    "reason": "No barriers detected.",
                    "all_interventions": []
                })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Score Adjustment
    # ------------------------------------------------------------------

    def _adjust_score(
        self,
        base_score: float,
        intervention: str,
        profile: Optional[dict]
    ) -> float:
        """
        Adjust base priority score using participant profile signals.

        Rules:
        - Low engagement → boost mentor_matcher
        - Low certifications → boost cert_suggester
        - High days_to_employment → boost internship_finder
        - Low network_strength → boost freelance_mapper
        """
        if not profile:
            return base_score

        score = base_score

        engagement = profile.get("avg_engagement_score", 5.0)
        num_certs = profile.get("num_certifications", 1)
        days_to_emp = profile.get("days_to_employment", 90)
        network = profile.get("network_strength", 0.5)

        if intervention == "mentor_matcher" and engagement < 4.0:
            score += 0.10
        if intervention == "cert_suggester" and num_certs < 2:
            score += 0.08
        if intervention == "internship_finder" and days_to_emp > 120:
            score += 0.07
        if intervention == "freelance_mapper" and network < 0.3:
            score += 0.06
        if intervention == "career_path_recommender" and num_certs >= 2:
            score += 0.05

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------

    def _get_reason(self, barrier: str, intervention: str) -> str:
        """Return a human-readable reason for the recommendation."""
        reasons = {
            ("opportunity_gap", "internship_finder"):
                "Local job opportunities are limited — internships bridge immediate access.",
            ("opportunity_gap", "freelance_mapper"):
                "Freelancing can generate income while formal opportunities are scarce.",
            ("opportunity_gap", "career_path_recommender"):
                "Redirecting to adjacent roles where demand exists.",
            ("confidence_deficit", "mentor_matcher"):
                "Low engagement signals confidence issues — mentorship provides structured support.",
            ("confidence_deficit", "cert_suggester"):
                "Additional certifications can rebuild confidence and signal readiness.",
            ("confidence_deficit", "career_path_recommender"):
                "Exploring alternative paths may reveal more accessible entry points.",
            ("credential_mismatch", "cert_suggester"):
                "Employer-recognized certifications needed to bridge the credential gap.",
            ("credential_mismatch", "career_path_recommender"):
                "Some career paths accept current credentials without additional training.",
            ("network_isolation", "mentor_matcher"):
                "Building a mentor relationship directly expands professional network.",
            ("network_isolation", "freelance_mapper"):
                "Freelance projects generate professional contacts organically.",
            ("economic_instability", "freelance_mapper"):
                "Flexible income through freelancing buffers against market instability.",
            ("economic_instability", "internship_finder"):
                "Paid internships provide income and build employability simultaneously.",
            ("economic_instability", "cert_suggester"):
                "Low-cost certifications in high-demand areas improve job market position.",
            ("skill_decay", "cert_suggester"):
                "Skills have decayed — refresher or updated certifications are recommended.",
            ("skill_decay", "career_path_recommender"):
                "Pivoting to roles where current skills still apply prevents full retraining.",
            ("skill_decay", "mentor_matcher"):
                "Mentors can provide rapid skill refreshment through guided practice.",
        }
        return reasons.get(
            (barrier, intervention),
            f"Recommended based on {barrier} barrier profile."
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_intervention_distribution(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Return frequency of each top intervention across all participants."""
        if "top_intervention" not in results_df.columns:
            raise ValueError("DataFrame must have 'top_intervention' column.")
        return (
            results_df["top_intervention"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "intervention", "top_intervention": "count"})
        )


# ------------------------------------------------------------------
# Quick usage example
# ------------------------------------------------------------------
if __name__ == "__main__":
    engine = InterventionEngine()

    # Single participant
    result = engine.recommend(
        participant_id="P001",
        barrier_label="confidence_deficit",
        profile={
            "avg_engagement_score": 3.1,
            "num_certifications": 1,
            "days_to_employment": 200,
            "network_strength": 0.2
        }
    )
    for r in result:
        print(r)

    # Batch
    df = pd.DataFrame([
        {"participant_id": "P001", "predicted_barrier": "confidence_deficit",
         "avg_engagement_score": 3.1, "num_certifications": 1},
        {"participant_id": "P002", "predicted_barrier": "skill_decay",
         "avg_engagement_score": 6.0, "num_certifications": 3},
        {"participant_id": "P003", "predicted_barrier": "network_isolation",
         "avg_engagement_score": 5.5, "num_certifications": 2},
    ])
    print(engine.recommend_batch(df))﻿# intervention_engine.py
# TODO: implement
