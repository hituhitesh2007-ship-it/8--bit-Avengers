# skill_relevance_model.py
# TODO: implement
# models/skill_relevance_model.py
# Scores and ranks skills by current and projected market relevance

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from sklearn.preprocessing import MinMaxScaler
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkillRelevanceModel:
    """
    Scores and ranks skills by market relevance using:

    1. Demand Signal       — current frequency in job postings
    2. Growth Trend        — rising/declining demand over time
    3. Salary Premium      — salary uplift associated with the skill
    4. Versatility Score   — number of industries/roles using the skill
    5. Decay Risk          — risk of automation or obsolescence
    6. Supply-Demand Gap   — scarcity of certified professionals vs openings

    Output: Relevance score [0, 1] per skill with trend direction
    """

    DECAY_RISK_MAP = {
        "data entry": 0.9, "telemarketing": 0.85, "bookkeeping": 0.75,
        "basic excel": 0.7, "manual testing": 0.65,
        "machine learning": 0.05, "python": 0.10, "nursing": 0.08,
        "cloud computing": 0.05, "cybersecurity": 0.04,
        "welding": 0.3, "plumbing": 0.25, "teaching": 0.2
    }

    COMPONENT_WEIGHTS = {
        "demand_score":       0.30,
        "growth_score":       0.25,
        "salary_premium":     0.20,
        "versatility_score":  0.15,
        "supply_gap_score":   0.10
    }

    def __init__(
        self,
        decay_risk_map: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.decay_risk_map = decay_risk_map or self.DECAY_RISK_MAP
        self.weights = weights or self.COMPONENT_WEIGHTS
        self.scaler = MinMaxScaler()
        self._skill_scores: Optional[pd.DataFrame] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Component Computation
    # ------------------------------------------------------------------

    def compute_demand_score(
        self,
        df: pd.DataFrame,
        skill_col: str = "skill",
        demand_col: str = "demand_count"
    ) -> pd.DataFrame:
        """Normalize demand count to [0, 1]."""
        df = df[[skill_col, demand_col]].copy()
        max_demand = df[demand_col].max()
        df["demand_score"] = (df[demand_col] / max_demand).round(4) if max_demand > 0 else 0.0
        return df

    def compute_growth_score(
        self,
        df: pd.DataFrame,
        skill_col: str = "skill",
        change_pct_col: str = "demand_change_pct"
    ) -> pd.DataFrame:
        """
        Normalize demand growth percentage to [0, 1].
        Negative growth maps to low score.
        """
        df = df[[skill_col, change_pct_col]].copy()
        df["growth_score"] = (
            (df[change_pct_col].clip(-100, 200) + 100) / 300
        ).round(4)
        return df

    def compute_salary_premium(
        self,
        df: pd.DataFrame,
        skill_col: str = "skill",
        avg_salary_col: str = "avg_salary_with_skill",
        baseline_salary: float = 40000.0
    ) -> pd.DataFrame:
        """Salary premium score relative to baseline."""
        df = df[[skill_col, avg_salary_col]].copy()
        df["salary_premium"] = (
            (df[avg_salary_col] - baseline_salary) / baseline_salary
        ).clip(0, 2).round(4) / 2
        return df

    def compute_versatility_score(
        self,
        df: pd.DataFrame,
        skill_col: str = "skill",
        num_industries_col: str = "num_industries",
        max_industries: float = 10.0
    ) -> pd.DataFrame:
        """Fraction of industries where the skill appears."""
        df = df[[skill_col, num_industries_col]].copy()
        df["versatility_score"] = (
            df[num_industries_col].clip(0, max_industries) / max_industries
        ).round(4)
        return df

    def compute_supply_gap_score(
        self,
        df: pd.DataFrame,
        skill_col: str = "skill",
        demand_col: str = "demand_count",
        supply_col: str = "certified_count"
    ) -> pd.DataFrame:
        """
        Score based on demand-supply gap: higher gap = higher score.
        gap_ratio = (demand - supply) / demand
        """
        df = df[[skill_col, demand_col, supply_col]].copy()
        df["gap_ratio"] = (
            (df[demand_col] - df[supply_col]) / (df[demand_col] + 1e-9)
        ).clip(0, 1).round(4)
        df["supply_gap_score"] = df["gap_ratio"]
        return df

    # ------------------------------------------------------------------
    # Composite Score
    # ------------------------------------------------------------------

    def fit(
        self,
        demand_df: pd.DataFrame,
        growth_df: Optional[pd.DataFrame] = None,
        salary_df: Optional[pd.DataFrame] = None,
        versatility_df: Optional[pd.DataFrame] = None,
        supply_df: Optional[pd.DataFrame] = None,
        skill_col: str = "skill"
    ) -> "SkillRelevanceModel":
        """
        Compute composite relevance scores from available component data.

        At minimum, demand_df must be provided.
        Other component DataFrames are optional and joined on skill_col.

        Returns:
            self
        """
        result = self.compute_demand_score(demand_df, skill_col=skill_col)

        if growth_df is not None:
            g = self.compute_growth_score(growth_df, skill_col=skill_col)
            result = result.merge(g[[skill_col, "growth_score"]], on=skill_col, how="left")
        else:
            result["growth_score"] = 0.5

        if salary_df is not None:
            s = self.compute_salary_premium(salary_df, skill_col=skill_col)
            result = result.merge(s[[skill_col, "salary_premium"]], on=skill_col, how="left")
        else:
            result["salary_premium"] = 0.5

        if versatility_df is not None:
            v = self.compute_versatility_score(versatility_df, skill_col=skill_col)
            result = result.merge(v[[skill_col, "versatility_score"]], on=skill_col, how="left")
        else:
            result["versatility_score"] = 0.5

        if supply_df is not None:
            sg = self.compute_supply_gap_score(supply_df, skill_col=skill_col)
            result = result.merge(sg[[skill_col, "supply_gap_score"]], on=skill_col, how="left")
        else:
            result["supply_gap_score"] = 0.5

        # Fill missing with neutral 0.5
        for col in ["growth_score", "salary_premium", "versatility_score", "supply_gap_score"]:
            result[col] = result[col].fillna(0.5)

        # Apply decay risk penalty
        result["decay_risk"] = result[skill_col].map(
            lambda s: self.decay_risk_map.get(s.lower(), 0.2)
        )
        result["decay_penalty"] = 1 - result["decay_risk"]

        # Weighted composite
        result["relevance_score"] = (
            self.weights["demand_score"]      * result["demand_score"] +
            self.weights["growth_score"]      * result["growth_score"] +
            self.weights["salary_premium"]    * result["salary_premium"] +
            self.weights["versatility_score"] * result["versatility_score"] +
            self.weights["supply_gap_score"]  * result["supply_gap_score"]
        ) * result["decay_penalty"]

        result["relevance_score"] = result["relevance_score"].clip(0, 1).round(4)

        result["relevance_tier"] = pd.cut(
            result["relevance_score"],
            bins=[0, 0.25, 0.50, 0.75, 1.0],
            labels=["low", "medium", "high", "critical"]
        )

        self._skill_scores = result.sort_values("relevance_score", ascending=False).reset_index(drop=True)
        self._is_fitted = True
        logger.info(f"Skill relevance computed for {len(self._skill_scores)} skills.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score_skill(self, skill: str) -> Optional[dict]:
        """Return relevance score for a single skill."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        row = self._skill_scores[self._skill_scores["skill"] == skill.lower()]
        return row.iloc[0].to_dict() if not row.empty else None

    def score_skill_list(self, skills: List[str]) -> pd.DataFrame:
        """Return scores for a list of skills."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        return self._skill_scores[
            self._skill_scores["skill"].isin([s.lower() for s in skills])
        ].reset_index(drop=True)

    def get_top_skills(self, n: int = 20, tier: Optional[str] = None) -> pd.DataFrame:
        """Return top N skills, optionally filtered by tier."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        df = self._skill_scores.copy()
        if tier:
            df = df[df["relevance_tier"] == tier]
        return df.head(n)

    def get_at_risk_skills(self, decay_threshold: float = 0.6) -> pd.DataFrame:
        """Return skills with high decay risk."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        return self._skill_scores[
            self._skill_scores["decay_risk"] >= decay_threshold
        ].reset_index(drop=True)

    def get_emerging_skills(self, growth_threshold: float = 0.7) -> pd.DataFrame:
        """Return skills with high growth scores."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        return self._skill_scores[
            self._skill_scores["growth_score"] >= growth_threshold
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        if self._skill_scores is not None:
            self._skill_scores.to_csv(
                os.path.join(output_dir, "skill_relevance_scores.csv"), index=False
            )
        joblib.dump(self.weights, os.path.join(output_dir, "skill_relevance_weights.pkl"))
        logger.info(f"SkillRelevanceModel saved to {output_dir}")

    def load(self, output_dir: str) -> "SkillRelevanceModel":
        scores_path = os.path.join(output_dir, "skill_relevance_scores.csv")
        if os.path.exists(scores_path):
            self._skill_scores = pd.read_csv(scores_path)
        self.weights = joblib.load(os.path.join(output_dir, "skill_relevance_weights.pkl"))
        self._is_fitted = True
        logger.info(f"SkillRelevanceModel loaded from {output_dir}")
        return self


if __name__ == "__main__":
    np.random.seed(42)
    skills = ["python", "machine learning", "data entry", "sql", "bookkeeping",
              "cloud computing", "nursing", "welding", "deep learning", "cybersecurity"]

    demand_df = pd.DataFrame({
        "skill": skills,
        "demand_count": np.random.randint(50, 5000, len(skills))
    })
    growth_df = pd.DataFrame({
        "skill": skills,
        "demand_change_pct": np.random.uniform(-40, 120, len(skills))
    })

    model = SkillRelevanceModel()
    model.fit(demand_df, growth_df=growth_df)
    print(model.get_top_skills(10)[["skill", "relevance_score", "relevance_tier", "decay_risk"]])
    print("\nAt-Risk Skills:")
    print(model.get_at_risk_skills())
