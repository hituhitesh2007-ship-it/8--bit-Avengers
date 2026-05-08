# skill_heatmap.py
# TODO: implement
# scoring/skill_heatmap.py
# Builds skill supply vs. demand heatmaps across regions and occupations

import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkillHeatmap:
    """
    Generates skill-level heatmap data comparing:
      - Participant skill supply  (from parsed resumes / profiles)
      - Job market skill demand   (from job postings / ONET / Lightcast)

    Outputs pivot tables, gap matrices, and optional matplotlib/seaborn visuals
    suited for dashboards and the explainability report.

    Inputs
    ------
    supply_df : one row per (participant, skill) or wide format
        Columns: participant_id, skill, region  [long format]
                 OR participant_id, region, <skill_col_1>, <skill_col_2>, … [wide]

    demand_df : one row per (occupation, skill) or wide format
        Columns: occupation, skill, demand_score  [long format]
                 OR occupation, <skill_col_1>, … [wide]
    """

    def __init__(
        self,
        supply_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        supply_format: str = "long",   # 'long' | 'wide'
        demand_format: str = "long",
    ):
        if supply_format not in ("long", "wide"):
            raise ValueError("supply_format must be 'long' or 'wide'.")
        if demand_format not in ("long", "wide"):
            raise ValueError("demand_format must be 'long' or 'wide'.")

        self.supply_format = supply_format
        self.demand_format = demand_format

        self._supply_long = self._normalise_supply(supply_df)
        self._demand_long = self._normalise_demand(demand_df)

        self.supply_pivot_: Optional[pd.DataFrame] = None   # rows=region,  cols=skill
        self.demand_pivot_: Optional[pd.DataFrame] = None   # rows=occupation, cols=skill
        self.gap_matrix_:   Optional[pd.DataFrame] = None   # supply − demand (region × skill)

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _normalise_supply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return long-format supply with columns [participant_id, region, skill]."""
        if self.supply_format == "long":
            required = {"participant_id", "skill", "region"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"supply_df missing columns: {missing}")
            return df[["participant_id", "region", "skill"]].copy()

        # wide → long
        id_vars = ["participant_id", "region"]
        skill_cols = [c for c in df.columns if c not in id_vars]
        melted = df.melt(id_vars=id_vars, value_vars=skill_cols,
                         var_name="skill", value_name="has_skill")
        return melted[melted["has_skill"] == 1][["participant_id", "region", "skill"]].copy()

    def _normalise_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return long-format demand with columns [occupation, skill, demand_score]."""
        if self.demand_format == "long":
            required = {"occupation", "skill", "demand_score"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"demand_df missing columns: {missing}")
            return df[["occupation", "skill", "demand_score"]].copy()

        # wide → long (treat numeric values as demand scores)
        id_vars = ["occupation"]
        skill_cols = [c for c in df.columns if c not in id_vars]
        melted = df.melt(id_vars=id_vars, value_vars=skill_cols,
                         var_name="skill", value_name="demand_score")
        return melted.dropna(subset=["demand_score"]).copy()

    # ------------------------------------------------------------------
    # Build pivots
    # ------------------------------------------------------------------

    def build_supply_pivot(self, normalise: bool = True) -> pd.DataFrame:
        """
        Build a region × skill pivot table of participant counts (or rates).

        Args:
            normalise: If True, divide each cell by total participants in that region
                       (supply rate 0–1). If False, return raw counts.

        Returns:
            DataFrame with regions as index and skills as columns.
        """
        pivot = (
            self._supply_long
            .groupby(["region", "skill"])
            .size()
            .reset_index(name="count")
            .pivot(index="region", columns="skill", values="count")
            .fillna(0)
        )

        if normalise:
            region_totals = (
                self._supply_long[["participant_id", "region"]]
                .drop_duplicates()
                .groupby("region")
                .size()
            )
            pivot = pivot.div(region_totals, axis=0).round(4)

        self.supply_pivot_ = pivot
        logger.info(f"Supply pivot built: {pivot.shape[0]} regions × {pivot.shape[1]} skills.")
        return pivot

    def build_demand_pivot(self) -> pd.DataFrame:
        """
        Build an occupation × skill pivot of average demand scores.

        Returns:
            DataFrame with occupations as index and skills as columns.
        """
        pivot = (
            self._demand_long
            .pivot_table(index="occupation", columns="skill",
                         values="demand_score", aggfunc="mean")
            .fillna(0)
            .round(4)
        )

        self.demand_pivot_ = pivot
        logger.info(f"Demand pivot built: {pivot.shape[0]} occupations × {pivot.shape[1]} skills.")
        return pivot

    # ------------------------------------------------------------------
    # Gap matrix
    # ------------------------------------------------------------------

    def build_gap_matrix(self) -> pd.DataFrame:
        """
        Compute supply − demand gap per skill aggregated across all regions
        and occupations.

        Positive values = surplus (more supply than demand).
        Negative values = deficit (demand exceeds supply).

        Returns:
            DataFrame indexed by skill with columns:
              supply_rate, avg_demand, gap, gap_direction
        """
        if self.supply_pivot_ is None:
            self.build_supply_pivot(normalise=True)
        if self.demand_pivot_ is None:
            self.build_demand_pivot()

        supply_mean = self.supply_pivot_.mean(axis=0).rename("supply_rate")
        demand_mean = self.demand_pivot_.mean(axis=0).rename("avg_demand")

        combined = pd.concat([supply_mean, demand_mean], axis=1).fillna(0)
        combined["gap"] = (combined["supply_rate"] - combined["avg_demand"]).round(4)
        combined["gap_direction"] = combined["gap"].apply(
            lambda g: "Surplus" if g > 0.05 else ("Deficit" if g < -0.05 else "Balanced")
        )

        self.gap_matrix_ = combined.sort_values("gap").reset_index()
        self.gap_matrix_.rename(columns={"index": "skill"}, inplace=True)
        logger.info(
            f"Gap matrix built. "
            f"Deficits: {(combined['gap_direction'] == 'Deficit').sum()}, "
            f"Surpluses: {(combined['gap_direction'] == 'Surplus').sum()}"
        )
        return self.gap_matrix_

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_top_deficit_skills(self, n: int = 10) -> pd.DataFrame:
        """Return top-n skills with highest demand–supply deficit."""
        if self.gap_matrix_ is None:
            self.build_gap_matrix()
        return (
            self.gap_matrix_[self.gap_matrix_["gap_direction"] == "Deficit"]
            .sort_values("gap")
            .head(n)
            .reset_index(drop=True)
        )

    def get_top_surplus_skills(self, n: int = 10) -> pd.DataFrame:
        """Return top-n skills where supply exceeds demand."""
        if self.gap_matrix_ is None:
            self.build_gap_matrix()
        return (
            self.gap_matrix_[self.gap_matrix_["gap_direction"] == "Surplus"]
            .sort_values("gap", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def get_skill_supply_by_region(self, skill: str) -> pd.Series:
        """Return supply rate for a specific skill across all regions."""
        if self.supply_pivot_ is None:
            self.build_supply_pivot()
        if skill not in self.supply_pivot_.columns:
            raise ValueError(f"Skill '{skill}' not found in supply pivot.")
        return self.supply_pivot_[skill].sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_supply_heatmap(
        self,
        top_n_skills: int = 20,
        figsize: Tuple[int, int] = (16, 8),
        title: str = "Skill Supply Rate by Region",
    ) -> None:
        """Render a seaborn heatmap of skill supply rates per region."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not installed. Cannot plot.")
            return

        if self.supply_pivot_ is None:
            self.build_supply_pivot()

        top_skills = (
            self.supply_pivot_.mean(axis=0)
            .sort_values(ascending=False)
            .head(top_n_skills)
            .index
        )
        data = self.supply_pivot_[top_skills]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            data, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.4, ax=ax, cbar_kws={"label": "Supply Rate"}
        )
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel("Skill")
        ax.set_ylabel("Region")
        plt.tight_layout()
        plt.show()

    def plot_gap_bar(
        self,
        top_n: int = 15,
        figsize: Tuple[int, int] = (12, 6),
    ) -> None:
        """Render a horizontal bar chart of skill gaps (supply − demand)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed. Cannot plot.")
            return

        if self.gap_matrix_ is None:
            self.build_gap_matrix()

        df = self.gap_matrix_.set_index("skill")["gap"].sort_values().head(top_n)
        colors = ["#d73027" if g < 0 else "#1a9850" for g in df]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df.index, df.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Supply Rate − Avg Demand Score")
        ax.set_title(f"Top {top_n} Skill Gaps (red = deficit, green = surplus)")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_all(self, output_dir: str) -> None:
        """Save supply pivot, demand pivot, and gap matrix to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        if self.supply_pivot_ is not None:
            path = os.path.join(output_dir, "supply_pivot.csv")
            self.supply_pivot_.to_csv(path)
            logger.info(f"Supply pivot saved → {path}")

        if self.demand_pivot_ is not None:
            path = os.path.join(output_dir, "demand_pivot.csv")
            self.demand_pivot_.to_csv(path)
            logger.info(f"Demand pivot saved → {path}")

        if self.gap_matrix_ is not None:
            path = os.path.join(output_dir, "gap_matrix.csv")
            self.gap_matrix_.to_csv(path, index=False)
            logger.info(f"Gap matrix saved → {path}")


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    skills = ["python", "sql", "excel", "communication", "logistics",
              "machine learning", "project management", "accounting"]
    regions = ["north", "south", "east", "west"]

    supply_rows = []
    for i in range(150):
        pid = f"P{i:04d}"
        region = regions[i % len(regions)]
        for skill in np.random.choice(skills, size=np.random.randint(2, 5), replace=False):
            supply_rows.append({"participant_id": pid, "region": region, "skill": skill})

    supply_df = pd.DataFrame(supply_rows)

    demand_rows = []
    occupations = ["Data Analyst", "Logistics Coordinator", "Accountant"]
    for occ in occupations:
        for skill in np.random.choice(skills, size=4, replace=False):
            demand_rows.append({"occupation": occ, "skill": skill,
                                 "demand_score": round(np.random.uniform(0.3, 1.0), 2)})
    demand_df = pd.DataFrame(demand_rows)

    heatmap = SkillHeatmap(supply_df, demand_df)
    heatmap.build_supply_pivot()
    heatmap.build_demand_pivot()
    gap = heatmap.build_gap_matrix()

    print("Top deficit skills:")
    print(heatmap.get_top_deficit_skills())
    heatmap.save_all("data/processed/heatmaps")
