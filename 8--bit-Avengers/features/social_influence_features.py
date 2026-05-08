# social_influence_features.py
# TODO: implement
# features/social_influence_features.py
# Measures a participant's influence within the community/cohort network

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SocialInfluenceFeatureEngineer:
    """
    Quantifies each participant's social influence within their cohort
    and broader community network.

    High social influence correlates with:
    - Peer referral potential (they can bring others to opportunities)
    - Leadership readiness
    - Soft skill proficiency (communication, networking)
    - Sustained program engagement

    Measures:
    - Content engagement received (reactions, replies to their posts)
    - Peer-initiated contact rate (how often others message them)
    - Referral activity (did they refer others to the program?)
    - Alumni engagement (do they mentor / give back post-placement?)
    - Community standing score (aggregate forum/channel reputation)

    Input: Community / platform interaction DataFrame
    """

    def __init__(self, df: pd.DataFrame, id_col: str = "participant_id"):
        self.df = df.copy()
        self.id_col = id_col
        self.features = None

    def build(self) -> pd.DataFrame:
        result = self.df[[self.id_col]].drop_duplicates().copy()
        result = result.merge(self._content_influence(), on=self.id_col, how="left")
        result = result.merge(self._peer_initiated_contact(), on=self.id_col, how="left")
        result = result.merge(self._referral_activity(), on=self.id_col, how="left")
        result = result.merge(self._alumni_engagement(), on=self.id_col, how="left")
        result["social_influence_score"] = self._composite_influence_score(result)
        result["influence_tier"] = result["social_influence_score"].apply(self._tier)
        self.features = result
        logger.info(f"Social influence features built: {result.shape}")
        return result

    def _content_influence(self) -> pd.DataFrame:
        """How much engagement do this participant's posts/content receive?"""
        out = self.df[[self.id_col]].drop_duplicates().copy()

        REACTION_COLS = ["reactions_received", "replies_received", "post_shares"]
        present = [c for c in REACTION_COLS if c in self.df.columns]

        if present:
            out["total_content_reactions"] = (
                self.df[present].fillna(0).astype(float).sum(axis=1).values
            )
            out["avg_reactions_per_post"] = np.where(
                self.df.get("posts_made", pd.Series(np.zeros(len(self.df)))).values > 0,
                out["total_content_reactions"] /
                self.df.get("posts_made", pd.Series(np.ones(len(self.df)))).values,
                0.0,
            )
        else:
            out["total_content_reactions"] = 0.0
            out["avg_reactions_per_post"] = 0.0

        return out

    def _peer_initiated_contact(self) -> pd.DataFrame:
        """How often do other participants reach out to this person?"""
        out = self.df[[self.id_col]].drop_duplicates().copy()

        if "messages_received" in self.df.columns:
            out["messages_received"] = self.df["messages_received"].fillna(0).values
            out["is_sought_out"] = (out["messages_received"] > 5).astype(int)
        else:
            out["messages_received"] = 0.0
            out["is_sought_out"] = 0

        if "connection_requests_received" in self.df.columns:
            out["connection_requests_received"] = (
                self.df["connection_requests_received"].fillna(0).values
            )

        return out

    def _referral_activity(self) -> pd.DataFrame:
        """Did this participant refer others to the program?"""
        out = self.df[[self.id_col]].drop_duplicates().copy()

        if "referrals_made" in self.df.columns:
            out["referrals_made"] = self.df["referrals_made"].fillna(0).values
            out["is_referrer"] = (out["referrals_made"] > 0).astype(int)
        else:
            out["referrals_made"] = 0
            out["is_referrer"] = 0

        if "referral_conversion_rate" in self.df.columns:
            out["referral_conversion_rate"] = (
                self.df["referral_conversion_rate"].fillna(0).values
            )

        return out

    def _alumni_engagement(self) -> pd.DataFrame:
        """Post-placement: do they mentor, speak, or contribute back?"""
        out = self.df[[self.id_col]].drop_duplicates().copy()

        ALUMNI_COLS = [
            "mentorship_sessions_given", "alumni_events_attended",
            "guest_speaker", "success_story_shared",
        ]
        present = [c for c in ALUMNI_COLS if c in self.df.columns]

        if present:
            out["alumni_engagement_score"] = (
                self.df[present].fillna(0).astype(float).sum(axis=1).values
            )
            out["is_alumni_contributor"] = (out["alumni_engagement_score"] > 0).astype(int)
        else:
            out["alumni_engagement_score"] = 0.0
            out["is_alumni_contributor"] = 0

        return out

    def _composite_influence_score(self, df: pd.DataFrame) -> pd.Series:
        """Weighted composite → normalized 0–1 social influence score."""
        WEIGHTS = {
            "total_content_reactions": 0.25,
            "avg_reactions_per_post": 0.15,
            "messages_received": 0.20,
            "referrals_made": 0.20,
            "alumni_engagement_score": 0.20,
        }

        score = pd.Series(np.zeros(len(df)), index=df.index)
        total_weight = 0.0

        for col, weight in WEIGHTS.items():
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

    @staticmethod
    def _tier(score: float) -> str:
        if score >= 0.66:
            return "influencer"
        elif score >= 0.33:
            return "active"
        return "passive"

    def get_top_influencers(self, n: int = 10) -> pd.DataFrame:
        """Return top N participants by social influence (peer referral candidates)."""
        if self.features is None:
            raise RuntimeError("Call build() first.")
        return (
            self.features.nlargest(n, "social_influence_score")
            .reset_index(drop=True)
        )

    def get_features(self) -> pd.DataFrame:
        if self.features is None:
            raise RuntimeError("Call build() first.")
        return self.features

    def save(self, output_path: str) -> None:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.features.to_csv(output_path, index=False)
        logger.info(f"Saved social influence features to {output_path}")


if __name__ == "__main__":
    community_df = pd.read_csv("data/processed/community_clean.csv")
    engineer = SocialInfluenceFeatureEngineer(community_df)
    df = engineer.build()
    print(df[["participant_id", "social_influence_score", "influence_tier"]].head(10))
    print(engineer.get_top_influencers())
    engineer.save("data/features/social_influence_features.csv")
