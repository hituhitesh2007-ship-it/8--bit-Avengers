import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehavioralSignalEngineer:
    """
    Generates behavioral engagement features per participant.

    Captures:
    - Platform login frequency and recency
    - Content consumption (courses viewed, resources downloaded)
    - Peer interaction signals (posts, replies, reactions)
    - Event attendance (workshops, webinars, job fairs)
    - Drop-off and re-engagement patterns

    Input: DataFrame from CommunityScraper / platform activity logs
    """

    ENGAGEMENT_COLS = [
        "logins", "courses_viewed", "resources_downloaded",
        "posts_made", "replies_given", "events_attended",
        "messages_sent", "profile_updates",
    ]

    def __init__(self, df: pd.DataFrame, date_col: str = "activity_date",
                 id_col: str = "participant_id"):
        self.df = df.copy()
        self.date_col = date_col
        self.id_col = id_col
        self.features = None

    def build(self) -> pd.DataFrame:
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce")
        result = self._aggregate_activity()
        result = result.merge(self._recency_signals(), on=self.id_col, how="left")
        result = result.merge(self._engagement_tier(), on=self.id_col, how="left")
        result = result.merge(self._dropout_signal(), on=self.id_col, how="left")
        self.features = result
        logger.info(f"Behavioral signal features built: {result.shape}")
        return result

    def _aggregate_activity(self) -> pd.DataFrame:
        """Sum all activity columns per participant."""
        agg_cols = [c for c in self.ENGAGEMENT_COLS if c in self.df.columns]
        if not agg_cols:
            logger.warning("No engagement columns found in dataframe.")
            return self.df[[self.id_col]].drop_duplicates()

        agg = self.df.groupby(self.id_col)[agg_cols].sum().reset_index()

        # Total activity composite score
        agg["total_activity_score"] = agg[agg_cols].sum(axis=1)

        # Active weeks (distinct weeks with any activity)
        self.df["week"] = self.df[self.date_col].dt.to_period("W")
        active_weeks = (
            self.df.groupby(self.id_col)["week"].nunique().reset_index()
            .rename(columns={"week": "active_weeks"})
        )
        agg = agg.merge(active_weeks, on=self.id_col, how="left")
        return agg

    def _recency_signals(self) -> pd.DataFrame:
        """Days since last activity and last login."""
        today = pd.Timestamp.today()
        recency = (
            self.df.groupby(self.id_col)[self.date_col]
            .max().reset_index()
            .rename(columns={self.date_col: "last_activity_date"})
        )
        recency["days_since_last_activity"] = (today - recency["last_activity_date"]).dt.days
        recency["activity_recency_score"] = np.exp(-recency["days_since_last_activity"] / 30)
        return recency[[self.id_col, "days_since_last_activity", "activity_recency_score"]]

    def _engagement_tier(self) -> pd.DataFrame:
        """Classify participants into engagement tiers: high / medium / low."""
        agg_cols = [c for c in self.ENGAGEMENT_COLS if c in self.df.columns]
        totals = self.df.groupby(self.id_col)[agg_cols].sum().sum(axis=1).reset_index()
        totals.columns = [self.id_col, "total_score"]

        p33 = totals["total_score"].quantile(0.33)
        p66 = totals["total_score"].quantile(0.66)

        def tier(score):
            if score >= p66:
                return 2  # high
            elif score >= p33:
                return 1  # medium
            return 0       # low

        totals["engagement_tier"] = totals["total_score"].apply(tier)
        return totals[[self.id_col, "engagement_tier"]]

    def _dropout_signal(self) -> pd.DataFrame:
        """Flag participants with >60 days inactivity gap as at-risk."""
        self.df_sorted = self.df.sort_values([self.id_col, self.date_col])

        def max_gap(group):
            dates = group[self.date_col].dropna().sort_values()
            if len(dates) < 2:
                return pd.Series({"max_inactivity_gap_days": np.nan, "dropout_risk": 0})
            gaps = dates.diff().dt.days.dropna()
            max_g = gaps.max()
            return pd.Series({
                "max_inactivity_gap_days": max_g,
                "dropout_risk": int(max_g > 60),
            })

        return (
            self.df.groupby(self.id_col)
            .apply(max_gap)
            .reset_index()
        )

    def get_features(self) -> pd.DataFrame:
        if self.features is None:
            raise RuntimeError("Call build() first.")
        return self.features

    def save(self, output_path: str) -> None:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.features.to_csv(output_path, index=False)
        logger.info(f"Saved behavioral signal features to {output_path}")
