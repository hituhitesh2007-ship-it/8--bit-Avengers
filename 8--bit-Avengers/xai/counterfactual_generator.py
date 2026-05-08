# counterfactual_generator.py
# TODO: implement
# explainability/counterfactual_generator.py
# Generates counterfactual explanations: "What would need to change to get a different outcome?"

import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CounterfactualGenerator:
    """
    Generates counterfactual explanations for individual predictions.

    A counterfactual answers: "What is the smallest change to this person's
    features that would flip the model's prediction?" — crucial for giving
    actionable feedback in a workforce reintegration context
    (e.g. "If you gained 1 more certification, the model predicts employment").

    Strategy: greedy feature perturbation search guided by the model's
    predicted probability. For production use, consider DiCE or CARLA.
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        feature_ranges: Dict[str, tuple],
        categorical_features: Optional[List[str]] = None,
        target_class: int = 1,
        step_size: float = 0.1,
        max_iter: int = 1000,
        probability_threshold: float = 0.5,
    ):
        """
        Args:
            model:                  Trained sklearn model with predict_proba.
            feature_names:          Ordered list of feature column names.
            feature_ranges:         Dict mapping feature_name → (min, max) for numeric features,
                                    or (list_of_values,) for categoricals.
            categorical_features:   Feature names that are categorical (will be swapped, not nudged).
            target_class:           Class index we want the counterfactual to achieve (default 1).
            step_size:              Fraction of feature range to perturb per iteration.
            max_iter:               Max greedy search iterations.
            probability_threshold:  Minimum predicted probability for the target class
                                    to declare a successful counterfactual.
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges
        self.categorical_features = set(categorical_features or [])
        self.target_class = target_class
        self.step_size = step_size
        self.max_iter = max_iter
        self.probability_threshold = probability_threshold

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        instance: pd.Series,
        immutable_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a counterfactual for a single instance.

        Args:
            instance:           A pd.Series (index = feature names) representing one individual.
            immutable_features: Features that must not be changed (e.g. 'age', 'region').

        Returns:
            dict with keys:
              - 'original':       original feature values
              - 'counterfactual': modified feature values that flip the prediction
              - 'changes':        dict of {feature: (original_val, new_val)}
              - 'original_prob':  predicted probability before change
              - 'cf_prob':        predicted probability after change
              - 'success':        bool — did we reach the threshold?
              - 'iterations':     number of perturbation steps taken
        """
        immutable = set(immutable_features or [])
        mutable = [f for f in self.feature_names if f not in immutable]

        current = instance.copy().astype(float)
        orig_prob = self._predict_prob(current)

        logger.info(
            f"Generating counterfactual | original P(class={self.target_class}) = {orig_prob:.4f}"
        )

        success = False
        for iteration in range(self.max_iter):
            best_feature, best_candidate, best_prob = None, None, self._predict_prob(current)

            for feat in mutable:
                if feat not in self.feature_ranges:
                    continue

                candidates = self._get_candidates(current[feat], feat)

                for val in candidates:
                    candidate = current.copy()
                    candidate[feat] = val
                    prob = self._predict_prob(candidate)
                    if prob > best_prob:
                        best_prob = prob
                        best_feature = feat
                        best_candidate = candidate.copy()

            if best_feature is None:
                logger.info("No improving perturbation found. Stopping early.")
                break

            current = best_candidate
            logger.debug(
                f"  iter {iteration + 1}: changed '{best_feature}' → P = {best_prob:.4f}"
            )

            if best_prob >= self.probability_threshold:
                success = True
                logger.info(
                    f"Counterfactual found at iteration {iteration + 1} | "
                    f"P = {best_prob:.4f}"
                )
                break

        cf_prob = self._predict_prob(current)
        changes = {
            feat: (instance[feat], current[feat])
            for feat in self.feature_names
            if instance[feat] != current[feat]
        }

        return {
            "original": instance.to_dict(),
            "counterfactual": current.to_dict(),
            "changes": changes,
            "original_prob": orig_prob,
            "cf_prob": cf_prob,
            "success": success,
            "iterations": iteration + 1,
        }

    def generate_batch(
        self,
        X: pd.DataFrame,
        immutable_features: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactuals for all rows in X.

        Returns:
            List of result dicts (same structure as generate()).
        """
        results = []
        for i, (_, row) in enumerate(X.iterrows()):
            logger.info(f"Generating counterfactual {i + 1}/{len(X)} …")
            results.append(self.generate(row, immutable_features=immutable_features))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _predict_prob(self, instance: pd.Series) -> float:
        """Return P(target_class) for a single instance."""
        x = instance.values.reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        return float(probs[self.target_class])

    def _get_candidates(self, current_val: float, feature: str) -> List[float]:
        """
        Generate candidate values to try for a single feature.

        For numeric features: nudge up and down by step_size fractions of the range.
        For categorical features: return all possible category values.
        """
        spec = self.feature_ranges[feature]

        if feature in self.categorical_features:
            # spec is expected to be a list/tuple of valid categories
            return [v for v in spec if v != current_val]

        lo, hi = spec
        step = (hi - lo) * self.step_size
        candidates = []

        val_up = min(current_val + step, hi)
        val_down = max(current_val - step, lo)

        if val_up != current_val:
            candidates.append(val_up)
        if val_down != current_val:
            candidates.append(val_down)

        return candidates

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def format_changes(self, result: Dict[str, Any]) -> str:
        """Return a human-readable summary of what changed."""
        if not result["changes"]:
            return "No changes were needed (or possible)."

        lines = [
            f"Counterfactual {'✓ FOUND' if result['success'] else '✗ NOT FOUND'}",
            f"  Original  P(employed) = {result['original_prob']:.4f}",
            f"  CF        P(employed) = {result['cf_prob']:.4f}",
            "",
            "  Changes required:",
        ]
        for feat, (orig, new) in result["changes"].items():
            direction = "▲" if new > orig else "▼"
            lines.append(f"    {direction}  {feat:35s}  {orig:.2f}  →  {new:.2f}")

        return "\n".join(lines)

    def to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Flatten a list of counterfactual results into a summary DataFrame."""
        rows = []
        for r in results:
            row = {
                "success": r["success"],
                "original_prob": r["original_prob"],
                "cf_prob": r["cf_prob"],
                "n_changes": len(r["changes"]),
                "iterations": r["iterations"],
            }
            for feat, (orig, new) in r["changes"].items():
                row[f"change_{feat}"] = new - orig
            rows.append(row)
        return pd.DataFrame(rows)

    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """Save batch counterfactual summary to CSV."""
        import os
        df = self.to_dataframe(results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved counterfactual results to {output_path}")


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    X_raw, y = make_classification(n_samples=300, n_features=5, random_state=42)
    feature_names = ["num_certs", "years_exp", "skills_count", "training_hours", "region_code"]
    X_df = pd.DataFrame(X_raw, columns=feature_names)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_df, y)

    feature_ranges = {
        "num_certs":      (0, 10),
        "years_exp":      (0, 30),
        "skills_count":   (0, 20),
        "training_hours": (0, 500),
        "region_code":    (0, 5),
    }

    cfgen = CounterfactualGenerator(
        model=clf,
        feature_names=feature_names,
        feature_ranges=feature_ranges,
        immutable_features_default=None,
        target_class=1,
    )

    result = cfgen.generate(X_df.iloc[0], immutable_features=["region_code"])
    print(cfgen.format_changes(result))
