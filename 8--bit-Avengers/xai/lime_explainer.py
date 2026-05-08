# lime_explainer.py
# TODO: implement
# explainability/lime_explainer.py
# Uses LIME to generate local, human-readable explanations for individual predictions

import logging
import numpy as np
import pandas as pd
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    Wraps LIME (Local Interpretable Model-agnostic Explanations) to produce
    per-prediction explanations for any sklearn-compatible classifier or
    regressor used in the workforce reintegration pipeline.

    Supports tabular data (structured feature vectors).
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        mode: str = "classification",
        training_data: Optional[np.ndarray] = None,
        categorical_features: Optional[List[int]] = None,
        random_state: int = 42,
    ):
        """
        Args:
            model:                sklearn-compatible model with predict_proba (classification)
                                  or predict (regression).
            feature_names:        List of feature names matching the input columns.
            class_names:          Class label strings for classification (e.g. ['No Job', 'Employed']).
            mode:                 'classification' or 'regression'.
            training_data:        Representative background data (ndarray) used to fit
                                  the LIME explainer kernel. If None, must pass at explain() time.
            categorical_features: List of column indices that are categorical.
            random_state:         Seed for reproducibility.
        """
        if mode not in ("classification", "regression"):
            raise ValueError("mode must be 'classification' or 'regression'.")

        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        self.training_data = training_data
        self.categorical_features = categorical_features or []
        self.random_state = random_state
        self.explainer = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def fit(self, training_data: Optional[np.ndarray] = None) -> None:
        """
        Initialise the LIME TabularExplainer.

        Args:
            training_data: Optional override for the background dataset.
        """
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            raise ImportError("lime is not installed. Run: pip install lime")

        data = training_data if training_data is not None else self.training_data
        if data is None:
            raise ValueError(
                "training_data must be provided either at __init__ or fit() time."
            )

        logger.info("Initialising LIME TabularExplainer …")
        self.explainer = LimeTabularExplainer(
            training_data=data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            categorical_features=self.categorical_features,
            mode=self.mode,
            random_state=self.random_state,
        )
        logger.info("LIME explainer ready.")

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
        top_labels: int = 1,
    ):
        """
        Explain a single prediction.

        Args:
            instance:     1-D feature vector (numpy array).
            num_features: Max features to include in the explanation.
            num_samples:  Number of perturbed samples LIME generates internally.
            top_labels:   Number of output labels to explain (classification only).

        Returns:
            lime.explanation.Explanation object.
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialised. Call fit() first.")

        predict_fn = (
            self.model.predict_proba
            if self.mode == "classification"
            else self.model.predict
        )

        logger.debug("Computing LIME explanation for one instance …")
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=top_labels,
        )
        return explanation

    def explain_as_dict(
        self,
        instance: np.ndarray,
        label: int = 1,
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> dict:
        """
        Return the LIME explanation for a single instance as a plain dict.

        Args:
            instance:     1-D feature vector.
            label:        Class index to explain (default 1 = positive class).
            num_features: Max features in the explanation.
            num_samples:  LIME internal sample count.

        Returns:
            dict mapping feature_condition_string → SHAP-like weight.
        """
        exp = self.explain_instance(instance, num_features=num_features, num_samples=num_samples)
        return dict(exp.as_list(label=label))

    def explain_batch(
        self,
        X: np.ndarray,
        label: int = 1,
        num_features: int = 10,
        num_samples: int = 2000,
    ) -> pd.DataFrame:
        """
        Explain every row in X and return a tidy DataFrame.

        Args:
            X:            2-D feature matrix (n_samples × n_features).
            label:        Class index to explain.
            num_features: Max features per explanation.
            num_samples:  LIME internal sample count (lower = faster).

        Returns:
            DataFrame with one row per sample. Columns are feature condition strings,
            values are LIME weights.
        """
        logger.info(f"Computing LIME explanations for {len(X)} samples …")
        records = []
        for i, row in enumerate(X):
            d = self.explain_as_dict(row, label=label, num_features=num_features, num_samples=num_samples)
            d["sample_idx"] = i
            records.append(d)
            if (i + 1) % 50 == 0:
                logger.info(f"  … {i + 1}/{len(X)} done")

        df = pd.DataFrame(records).set_index("sample_idx").fillna(0.0)
        logger.info("Batch LIME explanations complete.")
        return df

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def get_aggregate_importance(
        self,
        X: np.ndarray,
        label: int = 1,
        num_features: int = 10,
        num_samples: int = 2000,
    ) -> pd.DataFrame:
        """
        Compute mean absolute LIME weight per raw feature name across a dataset.

        Returns:
            DataFrame with columns ['feature', 'mean_abs_weight'] sorted descending.
        """
        batch_df = self.explain_batch(X, label=label, num_features=num_features, num_samples=num_samples)

        # LIME conditions look like "feat_3 > 0.50"; collapse to feature name
        feature_totals: dict = {}
        for col in batch_df.columns:
            raw_name = col.split(" ")[0]   # take the feature name part
            abs_vals = batch_df[col].abs()
            feature_totals[raw_name] = feature_totals.get(raw_name, 0) + abs_vals.mean()

        df = pd.DataFrame(
            list(feature_totals.items()), columns=["feature", "mean_abs_weight"]
        ).sort_values("mean_abs_weight", ascending=False).reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def show_explanation(self, instance: np.ndarray, num_features: int = 10) -> None:
        """Print a text-based explanation for a single instance."""
        exp = self.explain_instance(instance, num_features=num_features)
        label = exp.available_labels()[0]
        print(f"\nLIME Explanation (label={label}):")
        for feature, weight in exp.as_list(label=label):
            direction = "▲" if weight > 0 else "▼"
            print(f"  {direction}  {feature:40s}  {weight:+.4f}")

    def plot_explanation(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        title: str = "LIME Explanation",
    ) -> None:
        """Render a bar chart of LIME weights for a single instance."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available. Install it to use plot_explanation.")
            return

        exp = self.explain_instance(instance, num_features=num_features)
        label = exp.available_labels()[0]
        items = sorted(exp.as_list(label=label), key=lambda x: x[1])

        features = [i[0] for i in items]
        weights = [i[1] for i in items]
        colors = ["#d73027" if w < 0 else "#1a9850" for w in weights]

        fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.4)))
        ax.barh(features, weights, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("LIME Weight")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_importance(self, output_path: str, X: np.ndarray, label: int = 1) -> None:
        """Save aggregate feature importance to CSV."""
        import os
        df = self.get_aggregate_importance(X, label=label)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved LIME importance to {output_path}")


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import make_classification

    X_train, y_train = make_classification(n_samples=300, n_features=8, random_state=0)
    feature_names = [f"feat_{i}" for i in range(8)]

    clf = GradientBoostingClassifier(n_estimators=50, random_state=0)
    clf.fit(X_train, y_train)

    lime_exp = LIMEExplainer(
        model=clf,
        feature_names=feature_names,
        class_names=["Not Employed", "Employed"],
        mode="classification",
        training_data=X_train,
    )
    lime_exp.fit()

    lime_exp.show_explanation(X_train[0])
    print(lime_exp.explain_as_dict(X_train[0]))
    print(lime_exp.get_aggregate_importance(X_train[:20]))
