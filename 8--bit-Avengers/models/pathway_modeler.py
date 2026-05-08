# pathway_modeler.py
# TODO: implement
# models/pathway_modeler.py
# Models and recommends optimal career pathways based on skills, goals, and barriers

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathwayModeler:
    """
    Models career pathway transitions and recommends next steps for participants.

    Predicts:
    - Most likely next job role given current profile
    - Probability of successful transition to a target role
    - Estimated time-to-transition
    - Required upskilling steps

    Uses a multi-class classifier over a predefined job role taxonomy.
    """

    DEFAULT_JOB_ROLES = [
        "data_analyst", "software_developer", "digital_marketer",
        "business_analyst", "project_manager", "ux_designer",
        "accountant", "nurse", "logistics_coordinator",
        "sales_executive", "teacher", "graphic_designer",
        "supply_chain_manager", "customer_support_specialist",
        "hr_specialist", "financial_analyst", "content_writer",
        "network_engineer", "product_manager", "operations_analyst"
    ]

    FEATURE_COLUMNS = [
        "skill_count", "years_experience", "num_certifications",
        "education_level_encoded", "employment_status_encoded",
        "community_engagement_score", "sentiment_score",
        "network_strength", "days_to_employment",
        "region_unemployment_rate", "gap_score",
        "coverage_ratio", "num_barriers"
    ]

    def __init__(
        self,
        job_roles: Optional[List[str]] = None,
        random_state: int = 42
    ):
        self.job_roles = job_roles or self.DEFAULT_JOB_ROLES
        self.random_state = random_state

        self.role_classifier = None
        self.transition_model = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.role_encoder = LabelEncoder()
        self._is_fitted = False

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
                else:
                    if col in self.label_encoders:
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

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "next_job_role"
    ) -> "PathwayModeler":
        """
        Train the pathway classifier.

        Args:
            df:         Training DataFrame with features and target role column
            target_col: Column with the next job role label

        Returns:
            self
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        X = self._prepare(df, fit=True)
        y = self.role_encoder.fit_transform(df[target_col].astype(str))

        self.role_classifier = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            random_state=self.random_state, n_jobs=-1
        )
        self.role_classifier.fit(X, y)
        self._is_fitted = True

        logger.info(
            f"PathwayModeler trained on {len(df)} samples. "
            f"Classes: {len(self.role_encoder.classes_)}"
        )
        return self

    def evaluate(self, df: pd.DataFrame, target_col: str = "next_job_role") -> dict:
        """Return accuracy and classification report."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        X = self._prepare(df, fit=False)
        y_true = self.role_encoder.transform(df[target_col].astype(str))
        y_pred = self.role_classifier.predict(X)

        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "report": classification_report(
                y_true, y_pred,
                target_names=self.role_encoder.classes_,
                output_dict=True, zero_division=0
            )
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_next_role(self, df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
        """
        Predict the top-N most likely next job roles for each participant.

        Returns:
            DataFrame with participant_id and ranked role predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        X = self._prepare(df, fit=False)
        probas = self.role_classifier.predict_proba(X)

        results = []
        for i, row_proba in enumerate(probas):
            top_idx = np.argsort(row_proba)[::-1][:top_n]
            pid = df["participant_id"].iloc[i] if "participant_id" in df.columns else i
            for rank, idx in enumerate(top_idx):
                results.append({
                    "participant_id": pid,
                    "rank": rank + 1,
                    "predicted_role": self.role_encoder.classes_[idx],
                    "probability": round(float(row_proba[idx]), 4)
                })

        return pd.DataFrame(results)

    def predict_transition_success(
        self,
        participant_skills: List[str],
        target_role: str,
        required_skills: List[str]
    ) -> dict:
        """
        Estimate success probability of transitioning to a target role.

        Args:
            participant_skills: Current skills
            target_role:        Desired job role
            required_skills:    Skills required for the target role

        Returns:
            dict with success_probability, missing_skills, readiness_level
        """
        p_set = set(s.lower() for s in participant_skills)
        r_set = set(s.lower() for s in required_skills)
        matched = p_set & r_set
        missing = r_set - p_set

        coverage = len(matched) / len(r_set) if r_set else 0.0

        if coverage >= 0.8:
            readiness = "high"
        elif coverage >= 0.5:
            readiness = "moderate"
        elif coverage >= 0.3:
            readiness = "low"
        else:
            readiness = "insufficient"

        return {
            "target_role": target_role,
            "coverage_ratio": round(coverage, 4),
            "success_probability": round(coverage * 0.85 + 0.1 * (len(participant_skills) / 20), 4),
            "missing_skills": sorted(missing),
            "matched_skills": sorted(matched),
            "readiness_level": readiness,
            "estimated_upskilling_weeks": max(0, int((1 - coverage) * 24))
        }

    # ------------------------------------------------------------------
    # Pathway Map
    # ------------------------------------------------------------------

    def generate_pathway_map(
        self,
        current_role: str,
        target_role: str,
        skill_profile: List[str],
        role_skill_map: Dict[str, List[str]]
    ) -> List[dict]:
        """
        Generate a step-by-step pathway from current to target role.

        Args:
            current_role:   Participant's current or most recent role
            target_role:    Desired end role
            skill_profile:  Skills the participant currently has
            role_skill_map: Dict mapping each role to its required skills

        Returns:
            Ordered list of pathway steps with role, skills to gain, and timeline
        """
        if target_role not in role_skill_map:
            logger.warning(f"Target role '{target_role}' not in role_skill_map.")
            return []

        # Direct single-step path
        transition = self.predict_transition_success(
            skill_profile,
            target_role,
            role_skill_map.get(target_role, [])
        )

        steps = []
        if transition["readiness_level"] in ("high", "moderate"):
            steps.append({
                "step": 1,
                "from_role": current_role,
                "to_role": target_role,
                "skills_to_gain": transition["missing_skills"][:5],
                "estimated_weeks": transition["estimated_upskilling_weeks"],
                "success_probability": transition["success_probability"]
            })
        else:
            # Try to find an intermediate role
            intermediate = self._find_intermediate_role(
                skill_profile, target_role, role_skill_map
            )
            if intermediate:
                step1_transition = self.predict_transition_success(
                    skill_profile, intermediate, role_skill_map.get(intermediate, [])
                )
                combined_skills = skill_profile + role_skill_map.get(intermediate, [])
                step2_transition = self.predict_transition_success(
                    combined_skills, target_role, role_skill_map.get(target_role, [])
                )
                steps = [
                    {
                        "step": 1,
                        "from_role": current_role,
                        "to_role": intermediate,
                        "skills_to_gain": step1_transition["missing_skills"][:5],
                        "estimated_weeks": step1_transition["estimated_upskilling_weeks"],
                        "success_probability": step1_transition["success_probability"]
                    },
                    {
                        "step": 2,
                        "from_role": intermediate,
                        "to_role": target_role,
                        "skills_to_gain": step2_transition["missing_skills"][:5],
                        "estimated_weeks": step2_transition["estimated_upskilling_weeks"],
                        "success_probability": step2_transition["success_probability"]
                    }
                ]
            else:
                steps.append({
                    "step": 1,
                    "from_role": current_role,
                    "to_role": target_role,
                    "skills_to_gain": transition["missing_skills"],
                    "estimated_weeks": transition["estimated_upskilling_weeks"],
                    "success_probability": transition["success_probability"]
                })

        return steps

    def _find_intermediate_role(
        self,
        skill_profile: List[str],
        target_role: str,
        role_skill_map: Dict[str, List[str]]
    ) -> Optional[str]:
        """Find the best intermediate stepping-stone role."""
        target_skills = set(role_skill_map.get(target_role, []))
        best_role = None
        best_overlap = 0.0

        for role, skills in role_skill_map.items():
            if role == target_role:
                continue
            role_skills = set(skills)
            overlap = len(role_skills & target_skills) / len(target_skills) if target_skills else 0
            participant_coverage = len(set(skill_profile) & role_skills) / len(role_skills) if role_skills else 0
            combined = 0.5 * overlap + 0.5 * participant_coverage
            if combined > best_overlap:
                best_overlap = combined
                best_role = role

        return best_role if best_overlap > 0.3 else None

    # ------------------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        available = [c for c in self.FEATURE_COLUMNS]
        importances = self.role_classifier.feature_importances_
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
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.role_classifier, os.path.join(output_dir, "role_classifier.pkl"))
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        joblib.dump(self.role_encoder, os.path.join(output_dir, "role_encoder.pkl"))
        joblib.dump(self.label_encoders, os.path.join(output_dir, "label_encoders.pkl"))
        logger.info(f"PathwayModeler saved to {output_dir}")

    def load(self, output_dir: str) -> "PathwayModeler":
        self.role_classifier = joblib.load(os.path.join(output_dir, "role_classifier.pkl"))
        self.scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))
        self.role_encoder = joblib.load(os.path.join(output_dir, "role_encoder.pkl"))
        self.label_encoders = joblib.load(os.path.join(output_dir, "label_encoders.pkl"))
        self._is_fitted = True
        logger.info(f"PathwayModeler loaded from {output_dir}")
        return self


if __name__ == "__main__":
    np.random.seed(42)
    n = 400
    roles = ["data_analyst", "software_developer", "digital_marketer",
             "business_analyst", "project_manager"]
    dummy = pd.DataFrame({
        "participant_id": [f"P{i:04d}" for i in range(n)],
        "skill_count": np.random.randint(3, 20, n),
        "years_experience": np.random.randint(0, 12, n),
        "num_certifications": np.random.randint(0, 5, n),
        "education_level": np.random.choice(["diploma", "bachelor", "master"], n),
        "employment_status": np.random.choice(["employed", "unemployed"], n),
        "community_engagement_score": np.random.uniform(0, 1, n),
        "sentiment_score": np.random.uniform(-1, 1, n),
        "network_strength": np.random.uniform(0, 1, n),
        "days_to_employment": np.random.randint(0, 365, n),
        "region_unemployment_rate": np.random.uniform(3, 20, n),
        "gap_score": np.random.uniform(0, 1, n),
        "coverage_ratio": np.random.uniform(0, 1, n),
        "num_barriers": np.random.randint(0, 5, n),
        "next_job_role": np.random.choice(roles, n)
    })

    modeler = PathwayModeler()
    modeler.fit(dummy)
    preds = modeler.predict_next_role(dummy.head(5), top_n=3)
    print(preds)
