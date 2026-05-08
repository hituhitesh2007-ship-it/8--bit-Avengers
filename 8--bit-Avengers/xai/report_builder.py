# report_builder.py
# TODO: implement
# explainability/report_builder.py
# Assembles SHAP, LIME, and counterfactual outputs into structured HTML / PDF / JSON reports

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML template helpers
# ---------------------------------------------------------------------------

_CSS = """
<style>
  body  { font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; color: #222; }
  h1    { color: #1a4d7c; border-bottom: 2px solid #1a4d7c; padding-bottom: 6px; }
  h2    { color: #2c6fad; margin-top: 32px; }
  h3    { color: #444; margin-top: 20px; }
  table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 0.9em; }
  th    { background: #1a4d7c; color: #fff; padding: 8px 12px; text-align: left; }
  td    { padding: 7px 12px; border-bottom: 1px solid #ddd; }
  tr:nth-child(even) td { background: #f4f8fc; }
  .badge-success { background:#1a9850; color:#fff; border-radius:4px; padding:2px 8px; }
  .badge-fail    { background:#d73027; color:#fff; border-radius:4px; padding:2px 8px; }
  .meta          { color:#666; font-size:0.85em; margin-bottom:20px; }
  .section-box   { border:1px solid #d0d8e4; border-radius:6px; padding:16px 20px; margin:16px 0; }
</style>
"""


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Convert a DataFrame to a simple HTML table string."""
    subset = df.head(max_rows)
    rows = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for _, row in subset.iterrows()
    )
    headers = "".join(f"<th>{c}</th>" for c in subset.columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"


# ---------------------------------------------------------------------------
# ReportBuilder
# ---------------------------------------------------------------------------

class ReportBuilder:
    """
    Assembles explainability artefacts (SHAP importance, LIME importance,
    counterfactual results, model metadata) into a unified report.

    Supported output formats:
      - HTML  (rich, browser-viewable)
      - JSON  (machine-readable, for downstream dashboards)
      - CSV   (flat summary for analysts)
    """

    def __init__(self, report_title: str = "Model Explainability Report"):
        """
        Args:
            report_title: Title string shown at the top of the report.
        """
        self.report_title = report_title
        self.generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        # Sections populated by add_* methods
        self._model_meta: Dict[str, Any] = {}
        self._shap_importance: Optional[pd.DataFrame] = None
        self._lime_importance: Optional[pd.DataFrame] = None
        self._counterfactual_summary: Optional[pd.DataFrame] = None
        self._counterfactual_results: Optional[List[Dict]] = None
        self._custom_sections: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Data population
    # ------------------------------------------------------------------

    def add_model_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Record high-level model information.

        Args:
            metadata: e.g. {'model_type': 'RandomForest', 'n_estimators': 100,
                             'accuracy': 0.87, 'roc_auc': 0.92}
        """
        self._model_meta = metadata
        logger.info("Model metadata added.")

    def add_shap_importance(self, df: pd.DataFrame) -> None:
        """
        Add global SHAP feature importance.

        Args:
            df: DataFrame with columns ['feature', 'mean_abs_shap']
                (output of SHAPExplainer.get_global_importance()).
        """
        required = {"feature", "mean_abs_shap"}
        if not required.issubset(df.columns):
            raise ValueError(f"SHAP importance DataFrame must contain columns: {required}")
        self._shap_importance = df.copy()
        logger.info(f"SHAP importance added ({len(df)} features).")

    def add_lime_importance(self, df: pd.DataFrame) -> None:
        """
        Add aggregate LIME feature importance.

        Args:
            df: DataFrame with columns ['feature', 'mean_abs_weight']
                (output of LIMEExplainer.get_aggregate_importance()).
        """
        required = {"feature", "mean_abs_weight"}
        if not required.issubset(df.columns):
            raise ValueError(f"LIME importance DataFrame must contain columns: {required}")
        self._lime_importance = df.copy()
        logger.info(f"LIME importance added ({len(df)} features).")

    def add_counterfactual_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Add batch counterfactual results.

        Args:
            results: List of dicts from CounterfactualGenerator.generate_batch().
        """
        self._counterfactual_results = results

        rows = []
        for i, r in enumerate(results):
            row = {
                "sample_idx": i,
                "success": r.get("success", False),
                "original_prob": round(r.get("original_prob", 0.0), 4),
                "cf_prob": round(r.get("cf_prob", 0.0), 4),
                "n_changes": len(r.get("changes", {})),
                "iterations": r.get("iterations", 0),
            }
            rows.append(row)

        self._counterfactual_summary = pd.DataFrame(rows)
        logger.info(f"Counterfactual results added ({len(results)} instances).")

    def add_custom_section(self, title: str, content_html: str) -> None:
        """
        Add an arbitrary HTML section (e.g. confusion matrix, ROC curve embed).

        Args:
            title:        Section heading.
            content_html: Raw HTML string to embed.
        """
        self._custom_sections.append({"title": title, "html": content_html})
        logger.info(f"Custom section '{title}' added.")

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def build_html(self) -> str:
        """Build and return the full HTML report as a string."""
        sections = [f"<!DOCTYPE html><html><head><meta charset='utf-8'>{_CSS}</head><body>"]
        sections.append(f"<h1>{self.report_title}</h1>")
        sections.append(f"<p class='meta'>Generated: {self.generated_at}</p>")

        # ---- Model Metadata ----
        if self._model_meta:
            sections.append("<h2>Model Metadata</h2><div class='section-box'><ul>")
            for k, v in self._model_meta.items():
                sections.append(f"<li><strong>{k}:</strong> {v}</li>")
            sections.append("</ul></div>")

        # ---- SHAP ----
        if self._shap_importance is not None:
            sections.append("<h2>Global Feature Importance — SHAP</h2>")
            sections.append(
                "<p>Mean absolute SHAP value across all predictions. "
                "Higher = stronger influence on model output.</p>"
            )
            sections.append(_df_to_html_table(self._shap_importance.round(4)))

        # ---- LIME ----
        if self._lime_importance is not None:
            sections.append("<h2>Global Feature Importance — LIME</h2>")
            sections.append(
                "<p>Mean absolute LIME weight aggregated across sampled predictions.</p>"
            )
            sections.append(_df_to_html_table(self._lime_importance.round(4)))

        # ---- Counterfactuals ----
        if self._counterfactual_summary is not None:
            n_success = int(self._counterfactual_summary["success"].sum())
            n_total = len(self._counterfactual_summary)
            rate = n_success / n_total * 100 if n_total else 0

            sections.append("<h2>Counterfactual Analysis</h2>")
            sections.append(
                f"<p>Counterfactuals found for "
                f"<strong>{n_success}/{n_total}</strong> instances "
                f"({rate:.1f}%).</p>"
            )
            sections.append(_df_to_html_table(self._counterfactual_summary))

        # ---- Custom sections ----
        for sec in self._custom_sections:
            sections.append(f"<h2>{sec['title']}</h2>")
            sections.append(f"<div class='section-box'>{sec['html']}</div>")

        sections.append("</body></html>")
        return "\n".join(sections)

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------

    def build_json(self) -> dict:
        """Return the report contents as a Python dict (JSON-serialisable)."""
        payload: Dict[str, Any] = {
            "title": self.report_title,
            "generated_at": self.generated_at,
        }

        if self._model_meta:
            payload["model_metadata"] = self._model_meta

        if self._shap_importance is not None:
            payload["shap_importance"] = self._shap_importance.to_dict(orient="records")

        if self._lime_importance is not None:
            payload["lime_importance"] = self._lime_importance.to_dict(orient="records")

        if self._counterfactual_summary is not None:
            payload["counterfactual_summary"] = self._counterfactual_summary.to_dict(orient="records")

        return payload

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def save_html(self, output_path: str) -> None:
        """Write the HTML report to disk."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        html = self.build_html()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"HTML report saved → {output_path}")

    def save_json(self, output_path: str) -> None:
        """Write the JSON report to disk."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        payload = self.build_json()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(f"JSON report saved → {output_path}")

    def save_csv_summaries(self, output_dir: str) -> None:
        """Write each tabular section to a separate CSV file."""
        os.makedirs(output_dir, exist_ok=True)

        if self._shap_importance is not None:
            path = os.path.join(output_dir, "shap_importance.csv")
            self._shap_importance.to_csv(path, index=False)
            logger.info(f"Saved SHAP importance → {path}")

        if self._lime_importance is not None:
            path = os.path.join(output_dir, "lime_importance.csv")
            self._lime_importance.to_csv(path, index=False)
            logger.info(f"Saved LIME importance → {path}")

        if self._counterfactual_summary is not None:
            path = os.path.join(output_dir, "counterfactual_summary.csv")
            self._counterfactual_summary.to_csv(path, index=False)
            logger.info(f"Saved CF summary → {path}")

    def save_all(self, output_dir: str, base_name: str = "explainability_report") -> None:
        """Convenience method — saves HTML, JSON, and CSV in one call."""
        self.save_html(os.path.join(output_dir, f"{base_name}.html"))
        self.save_json(os.path.join(output_dir, f"{base_name}.json"))
        self.save_csv_summaries(os.path.join(output_dir, "csv"))


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Mock data
    shap_df = pd.DataFrame({
        "feature": ["num_certs", "years_exp", "skills_count", "training_hours", "region_code"],
        "mean_abs_shap": [0.42, 0.31, 0.18, 0.09, 0.04],
    })

    lime_df = pd.DataFrame({
        "feature": ["num_certs", "years_exp", "skills_count", "training_hours", "region_code"],
        "mean_abs_weight": [0.38, 0.29, 0.21, 0.08, 0.04],
    })

    cf_results = [
        {"success": True,  "original_prob": 0.32, "cf_prob": 0.61, "changes": {"num_certs": (1, 3)},  "iterations": 12},
        {"success": False, "original_prob": 0.28, "cf_prob": 0.44, "changes": {"years_exp": (2, 4)},   "iterations": 50},
        {"success": True,  "original_prob": 0.19, "cf_prob": 0.53, "changes": {"skills_count": (3, 7)}, "iterations": 8},
    ]

    builder = ReportBuilder(report_title="Workforce Reintegration — Explainability Report")
    builder.add_model_metadata({
        "model_type": "GradientBoostingClassifier",
        "n_estimators": 200,
        "accuracy": 0.84,
        "roc_auc": 0.91,
        "training_date": "2026-05-08",
    })
    builder.add_shap_importance(shap_df)
    builder.add_lime_importance(lime_df)
    builder.add_counterfactual_results(cf_results)

    builder.save_all(output_dir="reports/explainability")
    print("Report saved to reports/explainability/")
