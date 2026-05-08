# regional_demand.py
# TODO: implement
# features/regional_demand.py
# Derives region-level labor market demand features for each participant

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegionalDemandEngineer:
    """
    Maps each participant to their regional labor market conditions.

    Combines:
    - Job posting volume and growth by region
    - Sector-specific demand in participant's target field
    - Unemployment rate and trend
    - Economic indicators (GDP growth, CPI, hiring index)
    - Skill-supply/demand ratio for participant's skills

    Inputs:
        participant_df  : Participant feature DataFrame (with region + target_role)
        job_df          : Job market connector output
        economic_df     : Economic indicators connector output
    """

    def __init__(
        self,
        participant_df: pd.DataFrame,
        job_df: pd.DataFrame,
        economic_df: pd.DataFrame = None,
        id_col: str = "participant_id",
        region_col: str = "region",
    ):
        self.participant_df = participant_df.copy()
        self.job_df = job_df.copy()
        self.economic_df = economic_df.copy() if economic_df is not None else None
        self.id_col = id_col
        self.region_col = region_col
        self.features = None

    def build(self) -> pd.DataFrame:
        result = self.participant_df[[self.id_col]].drop_duplicates().copy()

        if self.region_col in self.participant_df.columns:
            result[self.region_col] = self.participant_df[self.region_col].values

        result = result.merge(self._job_volume_by_region(), on=self.region_col, how="left")
        result = result.merge(self._sector_demand_match(), on=self.id_col, how="left")

        if self.economic_df is not None:
            result = result.merge(self._economic_conditions(), on=self.region_col, how="left")

        result["regional_demand_score"] = self._composite_demand_score(result)

        self.features = result
        logger.info(f"Regional demand features built: {result.shape}")
        return result

    def _job_volume_by_region(self) -> pd.DataFrame:
        """Aggregate job posting counts and growth by region."""
        if self.region_col not in self.job_df.columns:
            logger.warning("job_df missing 'region' column.")
            # Return empty frame with region col
            regions = self.participant_df[self.region_col].unique()
            return pd.DataFrame({self.region_col: regions, "job_postings_count": np.nan})

        region_agg = (
            self.job_df.groupby(self.region_col)
            .agg(
                job_postings_count=("job_title", "count") if "job_title" in self.job_df.columns
                else (self.job_df.columns[1], "count"),
            )
            .reset_index()
        )

        # Month-over-month growth in postings if date column exists
        if "posted_date" in self.job_df.columns:
            self.job_df["posted_date"] = pd.to_datetime(self.job_df["posted_date"], errors="coerce")
            self.job_df["month"] = self.job_df["posted_date"].dt.to_period("M")

            monthly = (
                self.job_df.groupby([self.region_col, "month"])
                .size().reset_index(name="monthly_postings")
            )
            monthly = monthly.sort_values([self.region_col, "month"])
            monthly["mom_growth"] = monthly.groupby(self.region_col)["monthly_postings"].pct_change()

            growth = (
                monthly.groupby(self.region_col)["mom_growth"]
                .mean().reset_index()
                .rename(columns={"mom_growth": "avg_posting_growth_rate"})
            )
            region_agg = region_agg.merge(growth, on=self.region_col, how="left")

        return region_agg

    def _sector_demand_match(self) -> pd.DataFrame:
        """
        Score how well job demand in participant's region matches their target sector.
        """
        out = self.participant_df[[self.id_col]].drop_duplicates().copy()

        if "target_industry" not in self.participant_df.columns:
            out["sector_demand_match"] = np.nan
            return out

        if "industry" not in self.job_df.columns:
            out["sector_demand_match"] = np.nan
            return out

        industry_region_demand = (
            self.job_df.groupby(
                [col for col in [self.region_col, "industry"] if col in self.job_df.columns]
            )
            .size()
            .reset_index(name="sector_postings")
        )

        participant_lookup = self.participant_df[
            [self.id_col, self.region_col, "target_industry"]
        ].copy() if self.region_col in self.participant_df.columns else \
            self.participant_df[[self.id_col, "target_industry"]].copy()

        merged = participant_lookup.merge(
            industry_region_demand,
            left_on=[c for c in [self.region_col, "target_industry"] if c in participant_lookup.columns],
            right_on=[c for c in [self.region_col, "industry"] if c in industry_region_demand.columns],
            how="left",
        )
        merged["sector_demand_match"] = np.log1p(merged["sector_postings"].fillna(0))
        out = out.merge(merged[[self.id_col, "sector_demand_match"]], on=self.id_col, how="left")
        return out

    def _economic_conditions(self) -> pd.DataFrame:
        """Attach regional economic indicator averages."""
        econ = self.economic_df.copy()

        if self.region_col not in econ.columns:
            # Broadcast latest national-level indicators
            numeric_cols = econ.select_dtypes(include=[np.number]).columns
            context = econ[numeric_cols].tail(12).mean().to_dict()
            regions = self.participant_df[self.region_col].unique()
            rows = []
            for r in regions:
                row = {self.region_col: r}
                row.update({f"econ_{k}": v for k, v in context.items()})
                rows.append(row)
            return pd.DataFrame(rows)

        numeric_econ = econ.select_dtypes(include=[np.number]).columns.tolist()
        return (
            econ.groupby(self.region_col)[numeric_econ]
            .mean()
            .reset_index()
            .rename(columns={c: f"econ_{c}" for c in numeric_econ})
        )

    def _composite_demand_score(self, df: pd.DataFrame) -> pd.Series:
        """Normalize and combine demand signals into a 0–1 score."""
        SIGNAL_COLS = [
            "job_postings_count", "avg_posting_growth_rate",
            "sector_demand_match",
        ]

        score = pd.Series(np.zeros(len(df)), index=df.index)
        weights = [0.40, 0.30, 0.30]
        total_weight = 0.0

        for col, weight in zip(SIGNAL_COLS, weights):
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
                col_min, col_max = vals.min(), vals.max()
                if col_max > col_min:
                    normalized = (vals - col_min) / (col_max - col_min)
                else:
                    normalized = vals * 0
                score += weight * normalized
                total_weight += weight

        if total_weight > 0:
            score = score / total_weight

        return score.clip(0, 1)

    def get_features(self) -> pd.DataFrame:
        if self.features is None:
            raise RuntimeError("Call build() first.")
        return self.features

    def save(self, output_path: str) -> None:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.features.to_csv(output_path, index=False)
        logger.info(f"Saved regional demand features to {output_path}")
