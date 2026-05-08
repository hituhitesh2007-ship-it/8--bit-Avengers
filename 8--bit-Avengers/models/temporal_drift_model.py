# temporal_drift_model.py
# TODO: implement
# models/temporal_drift_model.py
# Detects and adapts to temporal drift in skills, job markets, and participant outcomes

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from scipy import stats
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalDriftModel:
    """
    Monitors and handles temporal drift in:
    - Skill relevance (skills becoming obsolete or newly valued)
    - Job market demand shifts
    - Model prediction distribution drift (data drift / concept drift)
    - Participant outcome pattern changes over time

    Methods:
    - Statistical drift detection (KS test, PSI)
    - Rolling window feature statistics
    - Drift alerts and retraining triggers
    - Temporal feature engineering (recency weighting)
    """

    PSI_BUCKETS = 10
    PSI_THRESHOLD = 0.2        # >0.2 = significant drift
    KS_P_THRESHOLD = 0.05      # p < 0.05 = drift detected
    WINDOW_DAYS = 90           # Rolling window size

    def __init__(self, reference_window_days: int = 180):
        """
        Args:
            reference_window_days: Days of data to use as baseline distribution
        """
        self.reference_window_days = reference_window_days
        self.reference_stats: Dict[str, dict] = {}
        self.drift_log: List[dict] = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Baseline Fitting
    # ------------------------------------------------------------------

    def fit_reference(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        date_col: str = "date"
    ) -> "TemporalDriftModel":
        """
        Fit the reference distribution from the most recent N days of data.

        Args:
            df:           DataFrame with temporal data
            feature_cols: Numerical feature columns to monitor
            date_col:     Date column for filtering

        Returns:
            self
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        cutoff = df[date_col].max() - pd.Timedelta(days=self.reference_window_days)
        reference = df[df[date_col] >= cutoff]

        for col in feature_cols:
            if col not in df.columns:
                continue
            vals = reference[col].dropna().values
            self.reference_stats[col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "percentiles": np.percentile(vals, np.linspace(0, 100, self.PSI_BUCKETS + 1)).tolist(),
                "n": len(vals)
            }

        self._is_fitted = True
        logger.info(
            f"Reference distribution fitted on {len(reference)} records "
            f"across {len(self.reference_stats)} features."
        )
        return self

    # ------------------------------------------------------------------
    # Drift Detection
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        date_col: str = "date"
    ) -> pd.DataFrame:
        """
        Detect drift in current data vs reference distribution.

        Args:
            df:           Incoming DataFrame
            feature_cols: Columns to check (defaults to fitted features)
            date_col:     Date column

        Returns:
            DataFrame with drift metrics per feature
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_reference() first.")

        feature_cols = feature_cols or list(self.reference_stats.keys())
        results = []

        for col in feature_cols:
            if col not in df.columns or col not in self.reference_stats:
                continue

            current_vals = df[col].dropna().values
            ref_stats = self.reference_stats[col]

            ks_stat, ks_p = stats.ks_2samp(
                np.random.normal(
                    ref_stats["mean"], ref_stats["std"] + 1e-9,
                    ref_stats["n"]
                ),
                current_vals
            )

            psi = self._compute_psi(
                ref_stats["percentiles"],
                current_vals
            )

            drift_detected = (ks_p < self.KS_P_THRESHOLD) or (psi > self.PSI_THRESHOLD)

            result = {
                "feature": col,
                "ks_statistic": round(ks_stat, 4),
                "ks_p_value": round(ks_p, 4),
                "psi_score": round(psi, 4),
                "drift_detected": drift_detected,
                "current_mean": round(float(np.mean(current_vals)), 4),
                "reference_mean": round(ref_stats["mean"], 4),
                "mean_shift": round(float(np.mean(current_vals)) - ref_stats["mean"], 4)
            }

            results.append(result)

            if drift_detected:
                self.drift_log.append({
                    "feature": col,
                    "psi": psi,
                    "ks_p": ks_p,
                    "timestamp": pd.Timestamp.now().isoformat()
                })
                logger.warning(f"DRIFT DETECTED: {col} | PSI={psi:.3f} | KS_p={ks_p:.4f}")

        return pd.DataFrame(results)

    def _compute_psi(self, reference_percentiles: list, current_vals: np.ndarray) -> float:
        """
        Compute Population Stability Index (PSI).

        PSI < 0.1: No drift
        0.1-0.2:   Minor drift
        > 0.2:     Significant drift
        """
        psi = 0.0
        bins = reference_percentiles
        current_hist, _ = np.histogram(current_vals, bins=bins)
        current_pct = current_hist / (len(current_vals) + 1e-9)

        ref_pct = np.full(len(current_pct), 1.0 / self.PSI_BUCKETS)

        for curr, ref in zip(current_pct, ref_pct):
            curr = max(curr, 1e-9)
            ref = max(ref, 1e-9)
            psi += (curr - ref) * np.log(curr / ref)

        return float(psi)

    # ------------------------------------------------------------------
    # Skill Drift
    # ------------------------------------------------------------------

    def compute_skill_relevance_drift(
        self,
        historical_demand: pd.DataFrame,
        current_demand: pd.DataFrame,
        skill_col: str = "skill",
        demand_col: str = "demand_count"
    ) -> pd.DataFrame:
        """
        Compare skill demand between historical and current job market data.

        Returns:
            DataFrame with drift metrics per skill (rising/declining/stable)
        """
        hist = historical_demand.groupby(skill_col)[demand_col].sum().reset_index()
        curr = current_demand.groupby(skill_col)[demand_col].sum().reset_index()

        merged = pd.merge(hist, curr, on=skill_col, suffixes=("_hist", "_curr"), how="outer").fillna(0)
        total_hist = merged[f"{demand_col}_hist"].sum() + 1e-9
        total_curr = merged[f"{demand_col}_curr"].sum() + 1e-9

        merged["hist_share"] = merged[f"{demand_col}_hist"] / total_hist
        merged["curr_share"] = merged[f"{demand_col}_curr"] / total_curr
        merged["demand_change_pct"] = (
            (merged["curr_share"] - merged["hist_share"]) / (merged["hist_share"] + 1e-9) * 100
        ).round(2)

        merged["trend"] = merged["demand_change_pct"].apply(
            lambda x: "rising" if x > 15 else "declining" if x < -15 else "stable"
        )

        return merged.sort_values("demand_change_pct", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Recency Weighting
    # ------------------------------------------------------------------

    def apply_recency_weights(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        decay_rate: float = 0.005
    ) -> pd.DataFrame:
        """
        Apply exponential recency weighting to training data.
        More recent records receive higher weights for model training.

        Args:
            df:         DataFrame with date column
            date_col:   Date column name
            decay_rate: Controls how fast weight decays with age

        Returns:
            DataFrame with appended 'recency_weight' column
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        max_date = df[date_col].max()
        days_ago = (max_date - df[date_col]).dt.days.fillna(365)
        df["recency_weight"] = np.exp(-decay_rate * days_ago).round(6)
        return df

    # ------------------------------------------------------------------
    # Rolling Statistics
    # ------------------------------------------------------------------

    def rolling_feature_stats(
        self,
        df: pd.DataFrame,
        feature_col: str,
        date_col: str = "date",
        window_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute rolling mean, std, and trend for a feature over time.

        Returns:
            DataFrame indexed by date with rolling statistics
        """
        window = window_days or self.WINDOW_DAYS
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
        df = df.set_index(date_col)

        df[f"{feature_col}_rolling_mean"] = (
            df[feature_col].rolling(f"{window}D").mean().round(4)
        )
        df[f"{feature_col}_rolling_std"] = (
            df[feature_col].rolling(f"{window}D").std().round(4)
        )
        df[f"{feature_col}_rolling_min"] = (
            df[feature_col].rolling(f"{window}D").min().round(4)
        )
        df[f"{feature_col}_rolling_max"] = (
            df[feature_col].rolling(f"{window}D").max().round(4)
        )
        return df.reset_index()

    # ------------------------------------------------------------------
    # Retraining Trigger
    # ------------------------------------------------------------------

    def should_retrain(
        self,
        drift_df: pd.DataFrame,
        max_drifted_features: int = 3
    ) -> Tuple[bool, List[str]]:
        """
        Determine if models should be retrained based on drift results.

        Args:
            drift_df:              Output of detect_drift()
            max_drifted_features:  Trigger retraining if more than N features drifted

        Returns:
            Tuple of (retrain_flag, list_of_drifted_features)
        """
        drifted = drift_df[drift_df["drift_detected"] == True]["feature"].tolist()
        retrain = len(drifted) >= max_drifted_features
        if retrain:
            logger.warning(
                f"RETRAINING RECOMMENDED: {len(drifted)} features drifted: {drifted}"
            )
        return retrain, drifted

    def get_drift_log(self) -> pd.DataFrame:
        """Return full drift event log as a DataFrame."""
        return pd.DataFrame(self.drift_log) if self.drift_log else pd.DataFrame()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.reference_stats, os.path.join(output_dir, "reference_stats.pkl"))
        joblib.dump(self.drift_log, os.path.join(output_dir, "drift_log.pkl"))
        logger.info(f"TemporalDriftModel saved to {output_dir}")

    def load(self, output_dir: str) -> "TemporalDriftModel":
        self.reference_stats = joblib.load(os.path.join(output_dir, "reference_stats.pkl"))
        log_path = os.path.join(output_dir, "drift_log.pkl")
        if os.path.exists(log_path):
            self.drift_log = joblib.load(log_path)
        self._is_fitted = True
        logger.info(f"TemporalDriftModel loaded from {output_dir}")
        return self


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df_hist = pd.DataFrame({
        "date": dates[:300],
        "skill_count": np.random.randint(3, 15, 300),
        "gap_score": np.random.uniform(0, 0.6, 300),
        "days_to_employment": np.random.randint(30, 180, 300)
    })
    df_curr = pd.DataFrame({
        "date": dates[300:],
        "skill_count": np.random.randint(5, 20, 200),
        "gap_score": np.random.uniform(0.4, 1.0, 200),
        "days_to_employment": np.random.randint(60, 300, 200)
    })

    drift_model = TemporalDriftModel()
    drift_model.fit_reference(df_hist, feature_cols=["skill_count", "gap_score", "days_to_employment"])
    drift_report = drift_model.detect_drift(df_curr)
    print(drift_report)

    retrain, features = drift_model.should_retrain(drift_report)
    print(f"\nRetrain needed: {retrain} | Drifted features: {features}")
