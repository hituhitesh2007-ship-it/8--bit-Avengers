# outcome_predictor.py
# TODO: implement
# models/outcome_predictor.py
# Predicts employment outcomes: job placement success, salary band, time-to-employment

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    mean_absolute_error, r2_score
)
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutcomePredictor:
    """
    Predicts multiple employment outcomes for trained participants:

    1. Employment Success (binary classifier):
       Will a participant secure employment within 6 months post-certification?

    2. Salary Band (multi-class classifier):
       Which salary tier will the participant land in?
       Bands: low / mid_low / mid / mid_high / high

    3. Time-to-Employment (regressor):
       How many days will it take to find employment after certification?

    All three models share the same feature set and preprocessing pipeline.
    """

    SALARY_BANDS = ["low", "mid_low", "mid", "mid_high", "high"]

    FEATURE_COLUMNS = [
        "skill_count", "years_experience", "num_certifications",
        "education_level_encoded", "employment_status_encoded",
        "community_engagement_score", "sentiment_score",
        "network_strength", "gap_score", "coverage_ratio",
        "num_barriers", "region_unemployment_rate",
        "distance_to_hub_km", "transport_access_score",
        "monthly_income_band", "num_dependents"
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.employment_clf = None
        self.salary_clf = None
        self.tte_regressor = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.salary_encoder = LabelEncoder()
        self._fitted = {"employment": False, "salary": False, "tte": False}

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _encode(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        for col in ["employment_status", "education_level"]:
            enc_col = f"{col}_encoded"
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[enc_col] = le.fit_transform(df[col].astype(str).fillna("unknown"))
                    self.label_encoders[col] = le
                elif col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[enc_col] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        return df

    def _prepare(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        df = self._encode(df, fit=fit)
        available = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        X = df[available].fillna(0).values
        return self.scaler.fit_transform(X) if fit else self.scaler.transform(X)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_employment_classifier(
        self,
        df: pd.DataFrame,
        target_col: str = "employed_within_6mo"
    ) -> "OutcomePredictor":
        """Train binary employment success classifier."""
        X = self._prepare(df, fit=True)
        y = df[target_col].fillna(0).astype(int).values

        self.employment_clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            random_state=self.random_state
        )
        self.employment_clf.fit(X, y)
        self._fitted["employment"] = True
        logger.info("Employment success classifier trained.")
        return self

    def fit_salary_classifier(
        self,
        df: pd.DataFrame,
        target_col: str = "salary_band"
    ) -> "OutcomePredictor":
        """Train salary band multi-class classifier."""
        X = self._prepare(df, fit=not self._fitted["employment"])
        y = self.salary_encoder.fit_transform(df[target_col].astype(str))

        self.salary_clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            random_state=self.random_state
        )
        self.salary_clf.fit(X, y)
        self._fitted["salary"] = True
        logger.info("Salary band classifier trained.")
        return self

    def fit_tte_regressor(
        self,
        df: pd.DataFrame,
        target_col: str = "days_to_employment"
    ) -> "OutcomePredictor":
        """Train time-to-employment regressor."""
        X = self._prepare(df, fit=not any(self._fitted.values()))
        y = df[target_col].fillna(df[target_col].median()).values

        self.tte_regressor = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            random_state=self.random_state
        )
        self.tte_regressor.fit(X, y)
        self._fitted["tte"] = True
        logger.info("Time-to-employment regressor trained.")
        return self

    def fit_all(
        self,
        df: pd.DataFrame,
        employment_col: str = "employed_within_6mo",
        salary_col: str = "salary_band",
        tte_col: str = "days_to_employment"
    ) -> "OutcomePredictor":
        """Train all three models sequentially."""
        X = self._prepare(df, fit=True)

        if employment_col in df.columns:
            y_emp = df[employment_col].fillna(0).astype(int).values
            self.employment_clf = GradientBoostingClassifier(
                n_estimators=150, max_depth=4, random_state=self.random_state
            )
            self.employment_clf.fit(X, y_emp)
            self._fitted["employment"] = True

        if salary_col in df.columns:
            y_sal = self.salary_encoder.fit_transform(df[salary_col].astype(str))
            self.salary_clf = GradientBoostingClassifier(
                n_estimators=150, max_depth=4, random_state=self.random_state
            )
            self.salary_clf.fit(X, y_sal)
            self._fitted["salary"] = True

        if tte_col in df.columns:
            y_tte = df[tte_col].fillna(df[tte_col].median()).values
            self.tte_regressor = GradientBoostingRegressor(
                n_estimators=150, max_depth=4, random_state=self.random_state
            )
            self.tte_regressor.fit(X, y_tte)
            self._fitted["tte"] = True

        logger.info("All outcome models trained.")
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        employment_col: str = "employed_within_6mo",
        salary_col: str = "salary_band",
        tte_col: str = "days_to_employment"
    ) -> dict:
        """Evaluate all fitted models and return metrics."""
        X = self._prepare(df, fit=False)
        results = {}

        if self._fitted["employment"] and employment_col in df.columns:
            y_true = df[employment_col].astype(int).values
            y_pred = self.employment_clf.predict(X)
            y_proba = self.employment_clf.predict_proba(X)[:, 1]
            results["employment"] = {
                "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
                "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            }

        if self._fitted["salary"] and salary_col in df.columns:
            y_true = self.salary_encoder.transform(df[salary_col].astype(str))
            y_pred = self.salary_clf.predict(X)
            results["salary"] = {
                "report": classification_report(
                    y_true, y_pred,
                    target_names=self.salary_encoder.classes_,
                    output_dict=True, zero_division=0
                )
            }

        if self._fitted["tte"] and tte_col in df.columns:
            y_true = df[tte_col].fillna(df[tte_col].median()).values
            y_pred = self.tte_regressor.predict(X)
            results["time_to_employment"] = {
                "mae_days": round(mean_absolute_error(y_true, y_pred), 1),
                "r2_score": round(r2_score(y_true, y_pred), 4)
            }

        return results

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all predictions for a DataFrame of participants.

        Returns:
            DataFrame with predicted outcomes appended
        """
        X = self._prepare(df, fit=False)
        results = pd.DataFrame()

        if "participant_id" in df.columns:
            results["participant_id"] = df["participant_id"].values

        if self._fitted["employment"] and self.employment_clf:
            proba = self.employment_clf.predict_proba(X)[:, 1]
            results["employment_success_prob"] = np.round(proba, 4)
            results["predicted_employed_6mo"] = (proba >= 0.5).astype(int)

        if self._fitted["salary"] and self.salary_clf:
            y_pred = self.salary_clf.predict(X)
            results["predicted_salary_band"] = self.salary_encoder.inverse_transform(y_pred)
            salary_proba = self.salary_clf.predict_proba(X)
            for i, band in enumerate(self.salary_encoder.classes_):
                results[f"salary_prob_{band}"] = np.round(salary_proba[:, i], 4)

        if self._fitted["tte"] and self.tte_regressor:
            tte_pred = self.tte_regressor.predict(X)
            results["predicted_days_to_employment"] = np.round(np.clip(tte_pred, 0, None), 1)

        return results

    def predict_single(self, record: dict) -> dict:
        """Predict outcomes for a single participant record (dict)."""
        df = pd.DataFrame([record])
        result_df = self.predict(df)
        return result_df.iloc[0].to_dict() if not result_df.empty else {}

    # ------------------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------------------

    def get_feature_importance(self, model: str = "employment") -> pd.DataFrame:
        """
        Return feature importances for a specific outcome model.

        Args:
            model: 'employment', 'salary', or 'tte'
        """
        clf_map = {
            "employment": self.employment_clf,
            "salary": self.salary_clf,
            "tte": self.tte_regressor
        }
        clf = clf_map.get(model)
        if clf is None:
            raise ValueError(f"Model '{model}' not fitted.")

        available = [c for c in self.FEATURE_COLUMNS]
        importances = clf.feature_importances_
        return (
            pd.DataFrame({
                "feature": available[:len(importances)],
                "importance": np.round(importances, 5)
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        if self.employment_clf:
            joblib.dump(self.employment_clf, os.path.join(output_dir, "employment_clf.pkl"))
        if self.salary_clf:
            joblib.dump(self.salary_clf, os.path.join(output_dir, "salary_clf.pkl"))
        if self.tte_regressor:
            joblib.dump(self.tte_regressor, os.path.join(output_dir, "tte_regressor.pkl"))
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        joblib.dump(self.salary_encoder, os.path.join(output_dir, "salary_encoder.pkl"))
        joblib.dump(self.label_encoders, os.path.join(output_dir, "label_encoders.pkl"))
        logger.info(f"OutcomePredictor saved to {output_dir}")

    def load(self, output_dir: str) -> "OutcomePredictor":
        for fname, attr in [
            ("employment_clf.pkl", "employment_clf"),
            ("salary_clf.pkl", "salary_clf"),
            ("tte_regressor.pkl", "tte_regressor")
        ]:
            path = os.path.join(output_dir, fname)
            if os.path.exists(path):
                setattr(self, attr, joblib.load(path))
                self._fitted[fname.split("_")[0]] = True
        self.scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))
        self.salary_encoder = joblib.load(os.path.join(output_dir, "salary_encoder.pkl"))
        self.label_encoders = joblib.load(os.path.join(output_dir, "label_encoders.pkl"))
        logger.info(f"OutcomePredictor loaded from {output_dir}")
        return self


if __name__ == "__main__":
    np.random.seed(42)
    n = 600
    dummy = pd.DataFrame({
        "participant_id": [f"P{i:04d}" for i in range(n)],
        "skill_count": np.random.randint(2, 20, n),
        "years_experience": np.random.randint(0, 15, n),
        "num_certifications": np.random.randint(0, 6, n),
        "education_level": np.random.choice(["diploma", "bachelor", "master"], n),
        "employment_status": np.random.choice(["employed", "unemployed"], n),
        "community_engagement_score": np.random.uniform(0, 1, n),
        "sentiment_score": np.random.uniform(-1, 1, n),
        "network_strength": np.random.uniform(0, 1, n),
        "gap_score": np.random.uniform(0, 1, n),
        "coverage_ratio": np.random.uniform(0, 1, n),
        "num_barriers": np.random.randint(0, 5, n),
        "region_unemployment_rate": np.random.uniform(3, 20, n),
        "distance_to_hub_km": np.random.uniform(0, 100, n),
        "transport_access_score": np.random.uniform(0, 1, n),
        "monthly_income_band": np.random.randint(1, 5, n),
        "num_dependents": np.random.randint(0, 5, n),
        "employed_within_6mo": np.random.randint(0, 2, n),
        "salary_band": np.random.choice(["low", "mid_low", "mid", "mid_high", "high"], n),
        "days_to_employment": np.random.randint(0, 365, n)
    })

    predictor = OutcomePredictor()
    predictor.fit_all(dummy)
    preds = predictor.predict(dummy.head(5))
    print(preds.T)
    print(predictor.evaluate(dummy))
