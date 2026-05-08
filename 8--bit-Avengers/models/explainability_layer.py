# explainability_layer.py
# TODO: implement
# models/explainability_layer.py
# Unified explainability interface using SHAP, LIME, and custom rule-based explanations

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Any
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplainabilityLayer:
    """
    Unified XAI (Explainable AI) interface for all models in the pipeline.

    Provides:
    - SHAP values (global and local)
    - LIME explanations for individual predictions
    - Rule-based natural language explanations
    - Counterfactual 'what-if' statements
    - Feature contribution summaries per participant

    Designed to wrap any scikit-learn compatible model.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_type: str = "tree",
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            model:         Fitted sklearn-compatible model
            feature_names: Feature column names (in order)
            model_type:    'tree' (GBM/RF), 'linear', or 'generic'
            class_names:   Class labels for classification models
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.class_names = class_names
        self._shap_explainer = None
        self._lime_explainer = None

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def _get_shap_explainer(self, X_background: Optional[np.ndarray] = None):
        """Lazy-load SHAP explainer."""
        if self._shap_explainer is not None:
            return self._shap_explainer
        try:
            import shap
            if self.model_type == "tree":
                self._shap_explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "linear":
                self._shap_explainer = shap.LinearExplainer(
                    self.model, X_background if X_background is not None else np.zeros((1, len(self.feature_names)))
                )
            else:
                if X_background is None:
                    raise ValueError("X_background required for generic SHAP explainer.")
                self._shap_explainer = shap.KernelExplainer(
                    self.model.predict, shap.sample(X_background, 100)
                )
            return self._shap_explainer
        except ImportError:
            logger.error("SHAP not installed. Run: pip install shap")
            raise

    def get_shap_values(
        self,
        X: np.ndarray,
        X_background: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute SHAP values for a set of instances.

        Args:
            X:            Feature matrix (N, num_features)
            X_background: Background dataset for KernelExplainer

        Returns:
            SHAP values array
        """
        explainer = self._get_shap_explainer(X_background)
        shap_values = explainer.shap_values(X)
        logger.info(f"SHAP values computed for {len(X)} instances.")
        return shap_values

    def get_shap_summary(
        self,
        X: np.ndarray,
        X_background: Optional[np.ndarray] = None,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Return mean absolute SHAP values per feature (global importance).

        Returns:
            DataFrame sorted by mean |SHAP| descending
        """
        shap_vals = self.get_shap_values(X, X_background)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

        mean_abs = np.abs(shap_vals).mean(axis=0)
        return (
            pd.DataFrame({
                "feature": self.feature_names[:len(mean_abs)],
                "mean_abs_shap": np.round(mean_abs, 5)
            })
            .sort_values("mean_abs_shap", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def explain_instance_shap(
        self,
        x: np.ndarray,
        X_background: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Explain a single instance prediction using SHAP.

        Returns:
            DataFrame with feature, value, and SHAP contribution
        """
        shap_vals = self.get_shap_values(x.reshape(1, -1), X_background)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

        contributions = shap_vals[0]
        return (
            pd.DataFrame({
                "feature": self.feature_names[:len(contributions)],
                "feature_value": x[:len(contributions)],
                "shap_contribution": np.round(contributions, 5)
            })
            .sort_values("shap_contribution", key=abs, ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # LIME
    # ------------------------------------------------------------------

    def _get_lime_explainer(self, X_train: np.ndarray):
        """Lazy-load LIME explainer."""
        if self._lime_explainer is not None:
            return self._lime_explainer
        try:
            from lime.lime_tabular import LimeTabularExplainer
            self._lime_explainer = LimeTabularExplainer(
                training_data=X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode="classification" if self.class_names else "regression"
            )
            return self._lime_explainer
        except ImportError:
            logger.error("LIME not installed. Run: pip install lime")
            raise

    def explain_instance_lime(
        self,
        x: np.ndarray,
        X_train: np.ndarray,
        num_features: int = 10,
        num_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Generate LIME explanation for a single instance.

        Returns:
            DataFrame with feature label and LIME weight
        """
        explainer = self._get_lime_explainer(X_train)

        predict_fn = (
            self.model.predict_proba
            if hasattr(self.model, "predict_proba")
            else self.model.predict
        )

        exp = explainer.explain_instance(
            x,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )

        items = exp.as_list()
        return pd.DataFrame(items, columns=["feature_condition", "lime_weight"])

    # ------------------------------------------------------------------
    # Rule-based Natural Language Explanation
    # ------------------------------------------------------------------

    def generate_nl_explanation(
        self,
        shap_df: pd.DataFrame,
        participant_id: Optional[str] = None,
        outcome_label: str = "employment success",
        top_n: int = 3
    ) -> str:
        """
        Generate a human-readable explanation from SHAP contributions.

        Args:
            shap_df:        Output of explain_instance_shap()
            participant_id: Optional participant identifier
            outcome_label:  Name of the predicted outcome
            top_n:          Number of top factors to mention

        Returns:
            Plain English explanation string
        """
        top_positive = shap_df[shap_df["shap_contribution"] > 0].head(top_n)
        top_negative = shap_df[shap_df["shap_contribution"] < 0].head(top_n)

        lines = []
        if participant_id:
            lines.append(f"Explanation for participant {participant_id}:")

        lines.append(f"\nFactors INCREASING {outcome_label}:")
        if top_positive.empty:
            lines.append("  No strong positive contributors found.")
        else:
            for _, row in top_positive.iterrows():
                lines.append(
                    f"  + {row['feature']} = {row['feature_value']:.2f} "
                    f"(contribution: +{row['shap_contribution']:.4f})"
                )

        lines.append(f"\nFactors DECREASING {outcome_label}:")
        if top_negative.empty:
            lines.append("  No strong negative contributors found.")
        else:
            for _, row in top_negative.iterrows():
                lines.append(
                    f"  - {row['feature']} = {row['feature_value']:.2f} "
                    f"(contribution: {row['shap_contribution']:.4f})"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Counterfactual Explanation
    # ------------------------------------------------------------------

    def generate_counterfactual(
        self,
        x: np.ndarray,
        feature_deltas: Dict[str, float],
        outcome_label: str = "employment success"
    ) -> str:
        """
        Generate a what-if counterfactual explanation.

        Args:
            x:              Original feature vector
            feature_deltas: Dict of {feature_name: new_value}
            outcome_label:  Outcome being predicted

        Returns:
            Plain English counterfactual statement
        """
        x_cf = x.copy()
        changes = []

        for feat, new_val in feature_deltas.items():
            if feat in self.feature_names:
                idx = self.feature_names.index(feat)
                old_val = x_cf[idx]
                x_cf[idx] = new_val
                changes.append(f"{feat}: {old_val:.2f} → {new_val:.2f}")

        original_pred = self.model.predict(x.reshape(1, -1))[0]
        cf_pred = self.model.predict(x_cf.reshape(1, -1))[0]

        if hasattr(self.model, "predict_proba"):
            orig_prob = self.model.predict_proba(x.reshape(1, -1))[0].max()
            cf_prob = self.model.predict_proba(x_cf.reshape(1, -1))[0].max()
            prob_str = f"Probability changes from {orig_prob:.2%} to {cf_prob:.2%}."
        else:
            prob_str = f"Predicted value changes from {original_pred:.2f} to {cf_pred:.2f}."

        lines = [
            f"Counterfactual Analysis for '{outcome_label}':",
            f"If the following changes were made:",
            *[f"  • {c}" for c in changes],
            f"Then: {prob_str}"
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Batch Explanations
    # ------------------------------------------------------------------

    def explain_batch(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        id_col: str = "participant_id",
        X_background: Optional[np.ndarray] = None,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Generate SHAP-based explanations for all records in a DataFrame.

        Returns:
            Long-format DataFrame: participant_id, feature, shap_contribution
        """
        shap_vals = self.get_shap_values(X, X_background)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

        records = []
        for i in range(len(X)):
            pid = df[id_col].iloc[i] if id_col in df.columns else i
            contribs = shap_vals[i]
            top_idx = np.argsort(np.abs(contribs))[::-1][:top_n]

            for rank, idx in enumerate(top_idx):
                records.append({
                    id_col: pid,
                    "rank": rank + 1,
                    "feature": self.feature_names[idx] if idx < len(self.feature_names) else f"f{idx}",
                    "feature_value": round(float(X[i, idx]), 4),
                    "shap_contribution": round(float(contribs[idx]), 5)
                })

        return pd.DataFrame(records)


if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingClassifier

    np.random.seed(42)
    n = 300
    features = ["skill_count", "years_experience", "gap_score",
                "network_strength", "sentiment_score"]
    X = np.random.rand(n, len(features))
    y = (X[:, 0] + X[:, 2] > 1.0).astype(int)

    clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)

    xai = ExplainabilityLayer(clf, feature_names=features, model_type="tree",
                              class_names=["not_employed", "employed"])

    shap_df = xai.explain_instance_shap(X[0])
    print(shap_df)
    print(xai.generate_nl_explanation(shap_df, participant_id="P0001"))
    print(xai.get_shap_summary(X[:50]))
