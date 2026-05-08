# employability_index.py
# TODO: implement
# scoring/employability_index.py
# Computes a composite Employability Index score for each participant

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmployabilityIndex:
    """
    Computes a normalised Employability Index (0–100) for each participant
    based on a weighted combination of sub-scores:

        - skill_match_score     : How well participant skills match job demand
        - certification_score   : Number and relevance of certifications held
        - experience_score      : Years and recency of work experience
        - training_score        : Hours of training / courses completed
        - market_demand_score   : Demand score for participant's target occupation

    All sub-scores are expected in [0, 1] range before weighting.
    Final index is scaled to [0, 100].
    """

    DEFAULT_WEIGHTS = {
        "skill_match_score":   0.30,
        "certification_score": 0.20,
        "experience_score":    0.20,
        "training_score":      0.15,
        "market_demand_score": 0.15,
    }

    SCORE_COLUMNS = list(DEFAULT_WEIGHTS.keys())

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Optional custom weight dict. Must sum to 1.0.
                     Keys must match DEFAULT_WEIGHTS.
        """
        if weights is not None:
            self._validate_weights(weights)
            self.weights = weights
        else:
            self.weights = self.DEFAULT_WEIGHTS.copy()

        self.scores_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_weights(self, weights: Dict[str, float]) -> None:
        missing = set(self.DEFAULT_WEIGHTS) - set(weights)
        if missing:
            raise ValueError(f"Missing weight keys: {missing}")
        total = sum(weights.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing_cols = [c for c in self.SCORE_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame missing columns: {missing_cols}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the Employability Index for all rows in df.

        Args:
            df: DataFrame containing one row per participant.
                Must include all SCORE_COLUMNS (values in [0, 1]).

        Returns:
            Copy of df with additional columns:
              - 'employability_index'  : final score in [0, 100]
              - 'employability_band'   : categorical band (Low / Medium / High / Very High)
        """
        self._validate_input(df)
        result = df.copy()

        # Clip sub-scores to [0, 1]
        for col in self.SCORE_COLUMNS:
            result[col] = result[col].clip(0.0, 1.0)

        # Weighted sum → scale to 100
        weighted = sum(result[col] * w for col, w in self.weights.items())
        result["employability_index"] = (weighted * 100).round(2)

        # Band classification
        result["employability_band"] = result["employability_index"].apply(
            self._assign_band
        )

        self.scores_ = result
        logger.info(
            f"Employability Index computed for {len(result)} participants. "
            f"Mean = {result['employability_index'].mean():.2f}"
        )
        return result

    @staticmethod
    def _assign_band(score: float) -> str:
        if score >= 75:
            return "Very High"
        elif score >= 50:
            return "High"
        elif score >= 25:
            return "Medium"
        else:
            return "Low"

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return descriptive statistics for the computed index."""
        if self.scores_ is None:
            raise RuntimeError("No scores computed yet. Call compute() first.")
        idx = self.scores_["employability_index"]
        return {
            "mean":   round(idx.mean(), 2),
            "median": round(idx.median(), 2),
            "std":    round(idx.std(), 2),
            "min":    round(idx.min(), 2),
            "max":    round(idx.max(), 2),
            "band_distribution": self.scores_["employability_band"].value_counts().to_dict(),
        }

    def get_top_participants(self, n: int = 10) -> pd.DataFrame:
        """Return the top-n participants by Employability Index."""
        if self.scores_ is None:
            raise RuntimeError("Call compute() first.")
        return (
            self.scores_
            .sort_values("employability_index", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def get_at_risk(self, threshold: float = 25.0) -> pd.DataFrame:
        """Return participants with Employability Index below threshold."""
        if self.scores_ is None:
            raise RuntimeError("Call compute() first.")
        at_risk = self.scores_[self.scores_["employability_index"] < threshold]
        logger.info(f"{len(at_risk)} participants below threshold {threshold}.")
        return at_risk.reset_index(drop=True)

    def get_band_breakdown(self) -> pd.DataFrame:
        """Return count and percentage per band."""
        if self.scores_ is None:
            raise RuntimeError("Call compute() first.")
        counts = self.scores_["employability_band"].value_counts().reset_index()
        counts.columns = ["band", "count"]
        counts["pct"] = (counts["count"] / len(self.scores_) * 100).round(1)
        return counts

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(self, output_path: str) -> None:
        """Save scored DataFrame to CSV."""
        if self.scores_ is None:
            raise RuntimeError("Call compute() first.")
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.scores_.to_csv(output_path, index=False)
        logger.info(f"Employability scores saved → {output_path}")


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        "participant_id":     [f"P{i:04d}" for i in range(n)],
        "skill_match_score":   np.random.beta(2, 5, n),
        "certification_score": np.random.beta(2, 3, n),
        "experience_score":    np.random.beta(3, 4, n),
        "training_score":      np.random.beta(2, 4, n),
        "market_demand_score": np.random.beta(4, 3, n),
    })

    scorer = EmployabilityIndex()
    result = scorer.compute(data)

    print(result[["participant_id", "employability_index", "employability_band"]].head(10))
    print("\nSummary:", scorer.get_summary())
    print("\nBand breakdown:\n", scorer.get_band_breakdown())
