# opportunity_score.py
# TODO: implement
# features/opportunity_score.py
# Computes a composite opportunity score for each participant
# Combines skill readiness, market demand, network strength, and confidence

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpportunityScoreEngineer:
    """
    Produces a single composite Opportunity Score (0–100) per participant
    by fusing all feature modules.

    The score answers: "How ready is this participant to achieve
    a positive employment outcome in the current market?"

    Components:
    ┌─────────────────────────────────┬────────┐
    │ Component                       │ Weight │
    ├─────────────────────────────────┼────────┤
    │ Skill-demand alignment          │  25%   │
    │ Employment readiness            │  20%   │
    │ Network strength                │  20%   │
    │ Behavioral engagement           │  15%   │
    │ Confidence proxy                │  10%   │
    │ Regional demand                 │  10%   │
    └─────────────────────────────────┴────────┘

    Inputs: Merged participant-level feature DataFrame containing
            outputs from all feature engineers.
    """

    COMPONENT_WEIGHTS = {
        "demand_alignment_score": 0.25,
        "employment_status_encoded": 0.20,
        "network_strength_score": 0.20,
        "activity_recency_score": 0.15,
        "confidence_proxy_score": 0.10,
        "regional_demand_score": 0.10,
    }

    def __init__(self, df: pd.DataFrame, id_col: str = "participant_id"):
        self.df = df.copy()
        self.id_col = id_col
        self.features = None

    def build(self) -> pd.DataFrame:
        """Compute opportunity scores for all participants."""
        result = self.df[[self.id_col]].drop_duplicates().copy()
        result["opportunity_score_raw"] = self._compute_raw_score()
        result["opportunity_score"] = (result["opportunity_score_raw"] * 100).round(1)
        result["opportunity_tier"] = result["opportunity_score"].apply(self._tier_label)
        result["priority_intervention"] = (result["opportunity_score"] < 40).astype(int)

        # Attach component scores for explainability
        for col in self.COMPONENT_WEIGHTS:
            if col in self.df.columns:
                result[f"component_{col}"] = self._normalize_col(self.df[col]).values

        self.features = result
        logger.info(f"Opportunity scores computed: {result.shape}")
        return result

    def _compute_raw_score(self) -> pd.Series:
        """Weighted sum of normalized components → 0–1 raw score."""
        score = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        total_weight = 0.0

        for col, weight in self.COMPONENT_WEIGHTS.items():
            if col in self.df.columns:
                normalized = self._normalize_col(self.df[col])
                score += weight * normalized
                total_weight += weight

        if total_weight > 0:
            score = score / total_weight

        return score.clip(0, 1).values

    def _normalize_col(self, series: pd.Series) -> pd.Series:
        """Min-max normalize a series to 0–1."""
        s = pd.to_numeric(series, errors="coerce").fillna(0)
        s_min, s_max = s.min(), s.max()
        if s_max > s_min:
            return (s - s_min) / (s_max - s_min)
        return s * 0  # all same value → 0

    @staticmethod
    def _tier_label(score: float) -> str:
        if score >= 70:
            return "high"
        elif score >= 40:
            return "medium"
        return "low"

    def get_top_opportunities(self, n: int = 20) -> pd.DataFrame:
        """Return top N participants by opportunity score (for placement focus)."""
        if self.features is None:
            raise RuntimeError("Call build() first.")
        return (
            self.features.nlargest(n, "opportunity_score")
            .reset_index(drop=True)
        )

    def get_priority_interventions(self) -> pd.DataFrame:
        """Return participants flagged for priority intervention (score < 40)."""
        if self.features is None:
            raise RuntimeError("Call build() first.")
        return self.features[self.features["priority_intervention"] == 1].reset_index(drop=True)

    def score_distribution(self) -> pd.DataFrame:
        """Return tier-level summary."""
        if self.features is None:
            raise RuntimeError("Call build() first.")
        return (
            self.features["opportunity_tier"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "tier", "opportunity_tier": "count"})
        )

    def get_features(self) -> pd.DataFrame:
        if self.features is None:
            raise RuntimeError("Call build() first.")
        return self.features

    def save(self, output_path: str) -> None:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.features.to_csv(output_path, index=False)
        logger.info(f"Saved opportunity scores to {output_path}")


if __name__ == "__main__":
    # Example usage with a merged feature DataFrame
    df = pd.read_csv("data/features/participant_features.csv")
    scorer = OpportunityScoreEngineer(df)
    result = scorer.build()
    print(result[["participant_id", "opportunity_score", "opportunity_tier"]].head(10))
    print(scorer.score_distribution())
    scorer.save("data/features/opportunity_scores.csv")
