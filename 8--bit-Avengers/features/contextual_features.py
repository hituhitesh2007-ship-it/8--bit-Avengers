# contextual_features.py
# TODO: implement
# features/confidence_proxy.py
# Estimates participant self-efficacy / confidence from indirect behavioral proxies

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceProxyEngineer:
    """
    Estimates participant confidence and self-efficacy using observable proxies.

    Workforce development research shows confidence is a strong predictor
    of job-seeking behavior and employment outcomes. Since we can't directly
    measure it, we proxy it through:

    - Profile completeness (willingness to present oneself)
    - Application attempt rate (willingness to try)
    - Interview acceptance rate (follow-through)
    - Peer help behavior (giving back signals security)
    - Self-assessment scores (if collected)
    - Re-enrollment after failure (resilience signal)

    Input: Merged DataFrame with activity, application, and profile data
    """

    def __init__(self, df: pd.DataFrame, id_col: str = "participant_id"):
        self.df = df.copy()
        self.id_col = id_col
        self.features = None

    def build(self) -> pd.DataFrame:
        result = self.df[[self.id_col]].drop_duplicates().copy()
        result = result.merge(self._profile_completeness(), on=self.id_col, how="left")
        result = result.merge(self._application_behavior(), on=self.id_col, how="left")
        result = result.merge(self._peer_help_signal(), on=self.id_col, how="left")
        result = result.merge(self._self_assessment_score(), on=self.id_col, how="left")
        result = result.merge(self._resilience_signal(), on=self.id_col, how="left")
        result["confidence_proxy_score"] = self._composite_score(result)
        self.features = result
        logger.info(f"Confidence proxy features built: {result.shape}")
        return result

    def _profile_completeness(self) -> pd.DataFrame:
        """Score profile fill rate as a proxy for willingness to present oneself."""
        PROFILE_FIELDS = [
            "has_photo", "has_bio", "has_resume", "has_skills_listed",
            "has_certifications", "has_work_history", "has_education",
        ]
        present = [c for c in PROFILE_FIELDS if c in self.df.columns]
        out = self.df[[self.id_col]].drop_duplicates().copy()

        if present:
            out["profile_completeness"] = (
                self.df[present].fillna(0).astype(int).sum(axis=1) / len(PROFILE_FIELDS)
            )
        else:
            out["profile_completeness"] = np.nan

        return out

    def _application_behavior(self) -> pd.DataFrame:
        """Application rate and interview conversion as confidence signals."""
        out = self.df[[self.id_col]].drop_duplicates().copy()

        if "num_applications" in self.df.columns:
            out["num_applications"] = self.df["num_applications"].values
            out["applied_at_all"] = (self.df["num_applications"] > 0).astype(int).values

        if "num_interviews" in self.df.columns and "num_applications" in self.df.columns:
            out["interview_conversion_rate"] = (
                self.df["num_interviews"] /
                (self.df["num_applications"].replace(0, np.nan))
            ).values

        return out

    def _peer_help_signal(self) -> pd.DataFrame:
        """Participants who help others signal higher confidence."""
        out = self.df[[self.id_col]].drop_duplicates().copy()

        if "replies_given" in self.df.columns:
            out["peer_help_score"] = np.log1p(self.df["replies_given"].fillna(0)).values
        elif "mentorship_sessions" in self.df.columns:
            out["peer_help_score"] = self.df["mentorship_sessions"].fillna(0).values
        else:
            out["peer_help_score"] = np.nan

        return out

    def _self_assessment_score(self) -> pd.DataFrame:
        """If platform collects self-assessment data, normalize it."""
        out = self.df[[self.id_col]].drop_duplicates().copy()

        if "self_assessment_score" in self.df.columns:
            scores = self.df["self_assessment_score"].fillna(
                self.df["self_assessment_score"].mean()
            )
            out["self_assessment_normalized"] = (
                (scores - scores.mean()) / (scores.std() + 1e-9)
            ).values
        else:
            out["self_assessment_normalized"] = np.nan

        return out

    def _resilience_signal(self) -> pd.DataFrame:
        """Re-engagement after dropout or failure signals resilience."""
        out = self.df[[self.id_col]].drop_duplicates().copy()

        if "re_enrolled" in self.df.columns:
            out["resilience_signal"] = self.df["re_enrolled"].fillna(0).astype(int).values
        elif "dropout_risk" in self.df.columns and "recently_active" in self.df.columns:
            # Was at dropout risk but came back
            out["resilience_signal"] = (
                (self.df["dropout_risk"] == 1) & (self.df["recently_active"] == 1)
            ).astype(int).values
        else:
            out["resilience_signal"] = np.nan

        return out

    def _composite_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Weighted composite of all proxy signals → normalized 0–1 confidence score.
        Weights reflect relative predictive value based on workforce dev literature.
        """
        WEIGHTS = {
            "profile_completeness": 0.20,
            "applied_at_all": 0.20,
            "interview_conversion_rate": 0.25,
            "peer_help_score": 0.15,
            "self_assessment_normalized": 0.10,
            "resilience_signal": 0.10,
        }

        score = pd.Series(np.zeros(len(df)), index=df.index)
        total_weight = 0.0

        for col, weight in WEIGHTS.items():
            if col in df.columns:
                col_vals = pd.to_numeric(df[col], errors="coerce")
                # Min-max normalize each component
                col_min, col_max = col_vals.min(), col_vals.max()
                if col_max > col_min:
                    normalized = (col_vals - col_min) / (col_max - col_min)
                else:
                    normalized = col_vals.fillna(0)
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
        logger.info(f"Saved confidence proxy features to {output_path}")
