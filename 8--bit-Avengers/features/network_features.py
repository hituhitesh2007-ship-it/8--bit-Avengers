# network_features.py
# TODO: implement
# features/network_features.py
# Derives social/professional network strength features per participant

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkFeatureEngineer:
    """
    Quantifies each participant's professional and peer network strength.

    Research shows network quality is one of the strongest predictors
    of employment outcomes for workforce development participants.

    Captures:
    - Connections to employed peers (warm referral potential)
    - Mentor/coach relationship depth
    - LinkedIn / professional profile connections
    - Alumni network engagement
    - Employer contact events (job fairs, info sessions)

    Inputs:
        activity_df   : Platform activity / community data
        linkedin_df   : Optional — from LinkedInScraper
    """

    def __init__(self, activity_df: pd.DataFrame,
                 linkedin_df: pd.DataFrame = None,
                 id_col: str = "participant_id"):
        self.activity_df = activity_df.copy()
        self.linkedin_df = linkedin_df.copy() if linkedin_df is not None else None
        self.id_col = id_col
        self.features = None

    def build(self) -> pd.DataFrame:
        result = self.activity_df[[self.id_col]].drop_duplicates().copy()
        result = result.merge(self._peer_connections(), on=self.id_col, how="left")
        result = result.merge(self._mentor_depth(), on=self.id_col, how="left")
        result = result.merge(self._employer_touchpoints(), on=self.id_col, how="left")
        if self.linkedin_df is not None:
            result = result.merge(self._linkedin_signals(), on=self.id_col, how="left")
        result["network_strength_score"] = self._composite_network_score(result)
        self.features = result
        logger.info(f"Network features built: {result.shape}")
        return result

    def _peer_connections(self) -> pd.DataFrame:
        """Count active peer connections and proportion who are employed."""
        out = self.activity_df[[self.id_col]].drop_duplicates().copy()

        if "peer_connections" in self.activity_df.columns:
            out["num_peer_connections"] = self.activity_df["peer_connections"].fillna(0).values

        if "employed_peer_connections" in self.activity_df.columns:
            out["employed_peer_connections"] = (
                self.activity_df["employed_peer_connections"].fillna(0).values
            )
            if "num_peer_connections" in out.columns:
                out["pct_employed_peers"] = (
                    out["employed_peer_connections"] /
                    out["num_peer_connections"].replace(0, np.nan)
                )
        return out

    def _mentor_depth(self) -> pd.DataFrame:
        """Mentor engagement depth: sessions, duration, quality rating."""
        out = self.activity_df[[self.id_col]].drop_duplicates().copy()

        if "mentor_sessions" in self.activity_df.columns:
            out["mentor_sessions"] = self.activity_df["mentor_sessions"].fillna(0).values
            out["has_mentor"] = (out["mentor_sessions"] > 0).astype(int)

        if "mentor_rating" in self.activity_df.columns:
            out["mentor_quality"] = self.activity_df["mentor_rating"].fillna(0).values

        if "coach_sessions" in self.activity_df.columns:
            out["coach_sessions"] = self.activity_df["coach_sessions"].fillna(0).values

        # Combined coaching depth
        coach_cols = [c for c in ["mentor_sessions", "coach_sessions"] if c in out.columns]
        if coach_cols:
            out["total_coaching_sessions"] = out[coach_cols].sum(axis=1)

        return out

    def _employer_touchpoints(self) -> pd.DataFrame:
        """Direct employer engagement events attended."""
        out = self.activity_df[[self.id_col]].drop_duplicates().copy()

        EMPLOYER_EVENTS = [
            "job_fairs_attended", "info_sessions_attended",
            "employer_site_visits", "mock_interviews_completed",
        ]
        present = [c for c in EMPLOYER_EVENTS if c in self.activity_df.columns]

        if present:
            out["employer_touchpoints"] = (
                self.activity_df[present].fillna(0).astype(int).sum(axis=1).values
            )
            out["any_employer_contact"] = (out["employer_touchpoints"] > 0).astype(int)
        else:
            out["employer_touchpoints"] = 0
            out["any_employer_contact"] = 0

        return out

    def _linkedin_signals(self) -> pd.DataFrame:
        """Extract network size and profile strength from LinkedIn data."""
        out = self.linkedin_df[[self.id_col]].drop_duplicates().copy()

        if "connections_count" in self.linkedin_df.columns:
            out["linkedin_connections"] = self.linkedin_df["connections_count"].fillna(0).values
            out["linkedin_500plus"] = (
                self.linkedin_df["connections_count"] >= 500
            ).astype(int).values

        if "profile_strength" in self.linkedin_df.columns:
            strength_map = {
                "all-star": 5, "expert": 4, "advanced": 3,
                "intermediate": 2, "beginner": 1,
            }
            out["linkedin_profile_strength"] = (
                self.linkedin_df["profile_strength"]
                .str.lower().map(strength_map).fillna(1).astype(int).values
            )

        if "recommendations_received" in self.linkedin_df.columns:
            out["linkedin_recommendations"] = (
                self.linkedin_df["recommendations_received"].fillna(0).values
            )

        return out

    def _composite_network_score(self, df: pd.DataFrame) -> pd.Series:
        """Weighted composite → normalized 0–1 network strength score."""
        WEIGHTS = {
            "num_peer_connections": 0.15,
            "pct_employed_peers": 0.20,
            "total_coaching_sessions": 0.20,
            "employer_touchpoints": 0.25,
            "linkedin_connections": 0.10,
            "linkedin_profile_strength": 0.10,
        }

        score = pd.Series(np.zeros(len(df)), index=df.index)
        total_weight = 0.0

        for col, weight in WEIGHTS.items():
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                col_min, col_max = vals.min(), vals.max()
                if col_max > col_min:
                    normalized = (vals - col_min) / (col_max - col_min)
                else:
                    normalized = vals.fillna(0)
                score += weight * normalized.fillna(0)
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
        logger.info(f"Saved network features to {output_path}")
