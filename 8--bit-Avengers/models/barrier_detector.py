# barrier_detector.py
# TODO: implement
# models/barrier_detector.py
# Detects structural, personal, and systemic barriers preventing skill utilization

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BarrierDetector:
    """
    Multi-label classifier that detects which barriers a participant faces:

    Barrier Categories:
    - geographic:    Distance to employment hubs, lack of transport
    - financial:     Cannot afford further training, family dependency
    - social:        Lack of networks, low community engagement
    - psychological: Low confidence, fear of failure, imposter syndrome
    - structural:    No formal credential recognition, systemic discrimination
    - informational: Unaware of available opportunities or programs
    - temporal:      Time constraints (caregiving, multiple jobs)

    Input features: employment history, certification data, community signals,
                    sentiment scores, regional economic indicators
    """

    BARRIER_TYPES = [
        "geographic", "financial", "social",
        "psychological", "structural", "informational", "temporal"
    ]

    FEATURE_COLUMNS = [
        "days_to_employment", "num_certifications", "employment_status_encoded",
        "community_engagement_score", "sentiment_score", "network_strength",
        "region_unemployment_rate", "distance_to_hub_km", "num_dependents",
        "years_experience", "education_level_encoded", "skill_count",
        "monthly_income_band", "transport_access_score"
    ]

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        threshold: float = 0.4,
        random_state: int = 42
    ):
        """
        Args:
            model_type:   'gradient_boosting' or 'random_forest'
            threshold:    Probability threshold for barrier flag (multi-label)
            random_state: Seed for reproducibility
        """
        self.model_type = model_type
        self.threshold = threshold
        self.random_state = random_state

        self.models: Dict[str, object] = {}
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        cat_cols = ["employment_status_encoded", "education_level_encoded"]

        for col in cat_cols:
            raw_col = col.replace("_encoded", "")
            if raw_col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[raw_col].astype(str).fillna("unknown"))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        df[col] = df[raw_col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        return df

    def _prepare_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        df = self._encode_categoricals(df, fit=fit)
        available = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        X = df[available].fillna(0).values

        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        return X

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, label_cols: Optional[List[str]] = None) -> "BarrierDetector":
        """
        Train one binary classifier per barrier type.

        Args:
            df:         Training DataFrame with features and barrier label columns
            label_cols: List of binary label columns (default: BARRIER_TYPES)

        Returns:
            self
        """
        label_cols = label_cols or self.BARRIER_TYPES
        X = self._prepare_features(df, fit=True)

        for barrier in label_cols:
            if barrier not in df.columns:
                logger.warning(f"Label column '{barrier}' not found. Skipping.")
                continue

            y = df[barrier].fillna(0).astype(int).values

            if self.model_type == "gradient_boosting":
                clf = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4,
                    random_state=self.random_state
                )
            else:
                clf = RandomForestClassifier(
                    n_estimators=100, max_depth=6,
                    random_state=self.random_state, n_jobs=-1
                )

            clf.fit(X, y)
            self.models[barrier] = clf
            logger.info(f"Trained barrier detector for: {barrier}")

        self._is_fitted = True
        return self

    def evaluate(self, df: pd.DataFrame, label_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Evaluate all barrier classifiers and return metrics DataFrame.

        Returns:
            DataFrame with precision, recall, f1, roc_auc per barrier
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        label_cols = label_cols or self.BARRIER_TYPES
        X = self._prepare_features(df, fit=False)
        results = []

        for barrier in label_cols:
            if barrier not in df.columns or barrier not in self.models:
                continue

            y_true = df[barrier].fillna(0).astype(int).values
            clf = self.models[barrier]
            y_pred = clf.predict(X)
            y_proba = clf.predict_proba(X)[:, 1]

            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5

            results.append({
                "barrier": barrier,
                "precision": round(report["1"]["precision"], 3),
                "recall": round(report["1"]["recall"], 3),
                "f1_score": round(report["1"]["f1-score"], 3),
                "roc_auc": round(auc, 3),
                "support": int(report["1"]["support"])
            })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict barrier flags for new participants.

        Returns:
            DataFrame with binary barrier flags per participant
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._prepare_features(df, fit=False)
        results = pd.DataFrame()

        if "participant_id" in df.columns:
            results["participant_id"] = df["participant_id"].values

        for barrier, clf in self.models.items():
            proba = clf.predict_proba(X)[:, 1]
            results[f"{barrier}_flag"] = (proba >= self.threshold).astype(int)
            results[f"{barrier}_probability"] = np.round(proba, 4)

        results["total_barriers"] = results[
            [c for c in results.columns if c.endswith("_flag")]
        ].sum(axis=1)

        results["barrier_severity"] = pd.cut(
            results["total_barriers"],
            bins=[-1, 0, 2, 4, 7],
            labels=["none", "low", "moderate", "high"]
        )

        return results

    def predict_proba_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return raw probability scores for all barriers."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        X = self._prepare_features(df, fit=False)
        results = pd.DataFrame()

        if "participant_id" in df.columns:
            results["participant_id"] = df["participant_id"].values

        for barrier, clf in self.models.items():
            results[f"{barrier}_prob"] = np.round(clf.predict_proba(X)[:, 1], 4)

        return results

    def get_feature_importance(self, barrier: str) -> pd.DataFrame:
        """
        Return feature importances for a specific barrier model.

        Args:
            barrier: One of BARRIER_TYPES

        Returns:
            DataFrame sorted by importance descending
        """
        if barrier not in self.models:
            raise ValueError(f"No model found for barrier: {barrier}")

        clf = self.models[barrier]
        available = [c for c in self.FEATURE_COLUMNS]
        importances = clf.feature_importances_

        return (
            pd.DataFrame({
                "feature": available[:len(importances)],
                "importance": importances
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        """Save all barrier models, scaler, and encoders."""
        os.makedirs(output_dir, exist_ok=True)
        for barrier, clf in self.models.items():
            joblib.dump(clf, os.path.join(output_dir, f"{barrier}_model.pkl"))
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(output_dir, "label_encoders.pkl"))
        logger.info(f"BarrierDetector saved to {output_dir}")

    def load(self, output_dir: str) -> "BarrierDetector":
        """Load all barrier models, scaler, and encoders."""
        for barrier in self.BARRIER_TYPES:
            path = os.path.join(output_dir, f"{barrier}_model.pkl")
            if os.path.exists(path):
                self.models[barrier] = joblib.load(path)
        self.scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))
        self.label_encoders = joblib.load(os.path.join(output_dir, "label_encoders.pkl"))
        self._is_fitted = True
        logger.info(f"BarrierDetector loaded from {output_dir}")
        return self


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    dummy = pd.DataFrame({
        "participant_id": [f"P{i:04d}" for i in range(n)],
        "days_to_employment": np.random.randint(0, 365, n),
        "num_certifications": np.random.randint(0, 5, n),
        "employment_status": np.random.choice(["employed", "unemployed", "underemployed"], n),
        "community_engagement_score": np.random.uniform(0, 1, n),
        "sentiment_score": np.random.uniform(-1, 1, n),
        "network_strength": np.random.uniform(0, 1, n),
        "region_unemployment_rate": np.random.uniform(3, 20, n),
        "distance_to_hub_km": np.random.uniform(0, 100, n),
        "num_dependents": np.random.randint(0, 5, n),
        "years_experience": np.random.randint(0, 15, n),
        "education_level": np.random.choice(["diploma", "bachelor", "master"], n),
        "skill_count": np.random.randint(1, 20, n),
        "monthly_income_band": np.random.randint(1, 5, n),
        "transport_access_score": np.random.uniform(0, 1, n),
        # barrier labels
        "geographic": np.random.randint(0, 2, n),
        "financial": np.random.randint(0, 2, n),
        "social": np.random.randint(0, 2, n),
        "psychological": np.random.randint(0, 2, n),
        "structural": np.random.randint(0, 2, n),
        "informational": np.random.randint(0, 2, n),
        "temporal": np.random.randint(0, 2, n),
    })

    detector = BarrierDetector()
    detector.fit(dummy)
    preds = detector.predict(dummy.head(10))
    print(preds[["participant_id", "total_barriers", "barrier_severity"]])
    print(detector.evaluate(dummy))
