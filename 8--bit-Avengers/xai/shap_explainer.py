# shap_explainer.py
# TODO: implement
# explainability/shap_explainer.py
# Uses SHAP to explain model predictions for employment outcome models

import logging
import numpy as np
import pandas as pd
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Wraps SHAP (SHapley Additive exPlanations) to generate feature-level
    explanations for trained classifiers or regressors used in the
    workforce reintegration pipeline.

    Supports:
    - TreeExplainer  (XGBoost, RandomForest, LightGBM)
    - LinearExplainer (LogisticRegression, Ridge)
    - KernelExplainer (model-agnostic fallback)
    """

    SUPPORTED_EXPLAINER_TYPES = ["tree", "linear", "kernel"]

    def __init__(self, model, explainer_type: str = "tree", background_data=None):
        """
        Args:
            model:            Trained sklearn-compatible model.
            explainer_type:   One of 'tree', 'linear', or 'kernel'.
            background_data:  Required for 'kernel'; optional for others.
                              Should be a representative sample (e.g. training data).
        """
        explainer_type = explainer_type.lower()
        if explainer_type not in self.SUPPORTED_EXPLAINER_TYPES:
            raise ValueError(
                f"explainer_type must be one of {self.SUPPORTED_EXPLAINER_TYPES}, "
                f"got '{explainer_type}'"
            )

        self.model = model
        self.explainer_type = explainer_type
        self.background_data = background_data
        self.explainer = None
        self.shap_values = None
        self.feature_names: Optional[list] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def fit(self, X_background=None) -> None:
        """
        Initialise the SHAP explainer.

        Args:
            X_background: Background dataset for KernelExplainer.
                          Falls back to self.background_data if not passed.
        """
        try:
            import shap
        except ImportError:
            raise ImportError("shap is not installed. Run: pip install shap")

        background = X_background if X_background is not None else self.background_data

        logger.info(f"Initialising {self.explainer_type} SHAP explainer …")

        if self.explainer_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)

        elif self.explainer_type == "linear":
            if background is None:
                raise ValueError("background_data is required for LinearExplainer.")
            self.explainer = shap.LinearExplainer(self.model, background)

        elif self.explainer_type == "kernel":
            if background is None:
                raise ValueError("background_data is required for KernelExplainer.")
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)

        logger.info("SHAP explainer ready.")

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for a dataset.

        Args:
            X: Feature matrix (DataFrame or ndarray).

        Returns:
            np.ndarray of SHAP values with shape (n_samples, n_features)
            or (n_samples, n_features, n_classes) for multi-class.
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialised. Call fit() first.")

        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        logger.info(f"Computing SHAP values for {len(X)} samples …")
        self.shap_values = self.explainer.shap_values(X)
        logger.info("SHAP values computed.")
        return self.shap_values

    def explain_single(self, x: pd.DataFrame) -> dict:
        """
        Explain a single prediction and return a feature→SHAP-value mapping.

        Args:
            x: Single-row DataFrame.

        Returns:
            dict mapping feature name → SHAP value.
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialised. Call fit() first.")

        values = self.explainer.shap_values(x)

        # For binary classifiers shap_values returns a list [neg_class, pos_class]
        if isinstance(values, list):
            values = values[1]  # take positive class

        if isinstance(x, pd.DataFrame):
            names = list(x.columns)
        else:
            names = [f"feature_{i}" for i in range(values.shape[1])]

        return dict(zip(names, values[0]))

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def get_global_importance(self) -> pd.DataFrame:
        """
        Return mean absolute SHAP values per feature (global importance).

        Returns:
            DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending.
        """
        if self.shap_values is None:
            raise RuntimeError("No SHAP values computed. Call explain() first.")

        values = self.shap_values
        if isinstance(values, list):
            values = values[1]  # binary positive class

        mean_abs = np.abs(values).mean(axis=0)

        names = self.feature_names or [f"feature_{i}" for i in range(len(mean_abs))]

        df = pd.DataFrame({
            "feature": names,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        return df

    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Return the top-n most important features by mean |SHAP|."""
        return self.get_global_importance().head(n)

    def get_shap_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a tidy DataFrame of SHAP values aligned with feature names.

        Args:
            X: The same DataFrame passed to explain().

        Returns:
            DataFrame with same columns as X, filled with SHAP values.
        """
        if self.shap_values is None:
            raise RuntimeError("No SHAP values computed. Call explain() first.")

        values = self.shap_values
        if isinstance(values, list):
            values = values[1]

        return pd.DataFrame(values, columns=list(X.columns))

    # ------------------------------------------------------------------
    # Visualisation helpers (optional – requires matplotlib)
    # ------------------------------------------------------------------

    def plot_summary(self, X: pd.DataFrame, plot_type: str = "dot") -> None:
        """
        Render a SHAP summary plot.

        Args:
            X:         Feature matrix used during explain().
            plot_type: 'dot' (beeswarm) or 'bar'.
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib or shap not available for plotting.")
            return

        if self.shap_values is None:
            raise RuntimeError("No SHAP values. Call explain() first.")

        values = self.shap_values
        if isinstance(values, list):
            values = values[1]

        shap.summary_plot(values, X, plot_type=plot_type, show=False)
        plt.tight_layout()
        plt.show()

    def plot_waterfall(self, X: pd.DataFrame, sample_idx: int = 0) -> None:
        """
        Render a waterfall plot for a single sample.

        Args:
            X:          Feature matrix.
            sample_idx: Row index to explain.
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib or shap not available for plotting.")
            return

        if self.explainer is None:
            raise RuntimeError("Explainer not initialised. Call fit() first.")

        explanation = self.explainer(X.iloc[[sample_idx]])
        if hasattr(explanation, "__getitem__") and isinstance(explanation, list):
            explanation = explanation[1]

        shap.plots.waterfall(explanation[0], show=False)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_importance(self, output_path: str) -> None:
        """Save global feature importance to CSV."""
        import os
        df = self.get_global_importance()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved SHAP importance to {output_path}")


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    X_train, y_train = make_classification(n_samples=200, n_features=10, random_state=42)
    X_df = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(10)])

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_df, y_train)

    explainer = SHAPExplainer(model=clf, explainer_type="tree")
    explainer.fit()
    explainer.explain(X_df)

    print(explainer.get_top_features(5))
    print(explainer.explain_single(X_df.iloc[[0]]))
