# skill_gap_nlp.py
# TODO: implement
# nlp/skill_gap_nlp.py
# Computes the skill gap between a participant's current skills and job requirements

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkillGapAnalyzer:
    """
    Computes skill gaps between participants and target job roles.

    Features:
    - Hard gap: exact skills missing from job requirement list
    - Semantic gap: embedding-based similarity for unlisted but related skills
    - Priority-weighted gap score
    - Batch analysis across all participants and roles
    - Actionable gap report generation

    Input:
        - participant_skills: dict or DataFrame of {participant_id: [skills]}
        - job_requirements:   dict or DataFrame of {job_id: [required_skills]}
    """

    def __init__(
        self,
        skill_weights: Optional[Dict[str, float]] = None,
        use_embeddings: bool = False
    ):
        """
        Args:
            skill_weights: Optional priority weights per skill
                           e.g. {"python": 1.5, "communication": 1.0}
            use_embeddings: If True, use transformer embeddings for semantic gap
        """
        self.skill_weights = skill_weights or {}
        self.use_embeddings = use_embeddings
        self._embedding_model = None

        if use_embeddings:
            self._load_embedding_model()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_embedding_model(self):
        """Load sentence-transformers model for semantic similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence transformer model loaded for semantic gap analysis.")
        except ImportError:
            logger.warning("sentence-transformers not installed. Semantic gap disabled.")
            self.use_embeddings = False

    # ------------------------------------------------------------------
    # Core Gap Computation
    # ------------------------------------------------------------------

    def compute_gap(
        self,
        participant_skills: List[str],
        required_skills: List[str]
    ) -> dict:
        """
        Compute the skill gap for a single participant vs a single job role.

        Args:
            participant_skills: Skills the participant currently has
            required_skills:    Skills required by the job

        Returns:
            dict with gap analysis fields
        """
        p_set = set(s.lower() for s in participant_skills)
        r_set = set(s.lower() for s in required_skills)

        missing = r_set - p_set
        matched = r_set & p_set
        extra = p_set - r_set

        coverage_ratio = len(matched) / len(r_set) if r_set else 0.0
        gap_score = self._compute_weighted_gap(missing, r_set)

        result = {
            "matched_skills": sorted(matched),
            "missing_skills": sorted(missing),
            "extra_skills": sorted(extra),
            "num_matched": len(matched),
            "num_missing": len(missing),
            "coverage_ratio": round(coverage_ratio, 4),
            "gap_score": round(gap_score, 4),
            "priority_gaps": self._get_priority_gaps(missing)
        }

        if self.use_embeddings and self._embedding_model:
            result["semantic_gap_score"] = self._semantic_gap(
                list(p_set), list(r_set)
            )

        return result

    def compute_batch(
        self,
        participants_df: pd.DataFrame,
        jobs_df: pd.DataFrame,
        p_id_col: str = "participant_id",
        p_skills_col: str = "skills",
        j_id_col: str = "job_id",
        j_skills_col: str = "required_skills"
    ) -> pd.DataFrame:
        """
        Compute gaps for all participant-job pairs.

        Args:
            participants_df: DataFrame with participant skills
            jobs_df:         DataFrame with job requirements
            ...

        Returns:
            Cross-joined DataFrame with gap analysis per pair
        """
        records = []
        for _, p_row in participants_df.iterrows():
            p_skills = p_row[p_skills_col]
            if isinstance(p_skills, str):
                p_skills = [s.strip() for s in p_skills.split(",")]

            for _, j_row in jobs_df.iterrows():
                j_skills = j_row[j_skills_col]
                if isinstance(j_skills, str):
                    j_skills = [s.strip() for s in j_skills.split(",")]

                gap = self.compute_gap(p_skills, j_skills)
                record = {
                    p_id_col: p_row[p_id_col],
                    j_id_col: j_row[j_id_col],
                    **gap
                }
                records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Computed {len(df)} participant-job gap pairs.")
        return df

    def best_fit_jobs(
        self,
        participant_skills: List[str],
        jobs_df: pd.DataFrame,
        j_id_col: str = "job_id",
        j_skills_col: str = "required_skills",
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Return top-N best-fitting jobs for a participant based on coverage ratio.

        Args:
            participant_skills: List of skills the participant has
            jobs_df:            DataFrame of jobs with required skills
            top_n:              Number of top matches to return

        Returns:
            Ranked DataFrame of job matches
        """
        results = []
        for _, j_row in jobs_df.iterrows():
            j_skills = j_row[j_skills_col]
            if isinstance(j_skills, str):
                j_skills = [s.strip() for s in j_skills.split(",")]

            gap = self.compute_gap(participant_skills, j_skills)
            results.append({
                j_id_col: j_row[j_id_col],
                "coverage_ratio": gap["coverage_ratio"],
                "gap_score": gap["gap_score"],
                "missing_skills": gap["missing_skills"],
                "num_missing": gap["num_missing"]
            })

        return (
            pd.DataFrame(results)
            .sort_values("coverage_ratio", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Weighted & Semantic Gap
    # ------------------------------------------------------------------

    def _compute_weighted_gap(self, missing: set, required: set) -> float:
        """Compute gap score with optional skill weights."""
        if not required:
            return 0.0

        total_weight = sum(self.skill_weights.get(s, 1.0) for s in required)
        missing_weight = sum(self.skill_weights.get(s, 1.0) for s in missing)
        return missing_weight / total_weight if total_weight else 0.0

    def _get_priority_gaps(self, missing: set, top_n: int = 5) -> List[str]:
        """Return top-N highest-priority missing skills by weight."""
        weighted = sorted(
            missing,
            key=lambda s: self.skill_weights.get(s, 1.0),
            reverse=True
        )
        return weighted[:top_n]

    def _semantic_gap(self, participant_skills: List[str], required_skills: List[str]) -> float:
        """
        Compute semantic gap using cosine similarity between skill embeddings.
        Skills present semantically but not lexically reduce the gap.
        """
        if not participant_skills or not required_skills:
            return 1.0

        p_emb = self._embedding_model.encode(participant_skills)
        r_emb = self._embedding_model.encode(required_skills)

        # For each required skill, find max cosine sim with any participant skill
        from numpy.linalg import norm
        covered = 0
        for r in r_emb:
            sims = [
                np.dot(r, p) / (norm(r) * norm(p) + 1e-9)
                for p in p_emb
            ]
            if max(sims) > 0.75:  # threshold for semantic match
                covered += 1

        return round(1.0 - (covered / len(required_skills)), 4)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(
        self,
        participant_id: str,
        participant_skills: List[str],
        job_id: str,
        required_skills: List[str]
    ) -> str:
        """Generate a human-readable skill gap report."""
        gap = self.compute_gap(participant_skills, required_skills)

        lines = [
            f"=== Skill Gap Report ===",
            f"Participant : {participant_id}",
            f"Target Job  : {job_id}",
            f"Coverage    : {gap['coverage_ratio'] * 100:.1f}%",
            f"Gap Score   : {gap['gap_score']:.3f}",
            f"\nMatched Skills ({gap['num_matched']}):",
            "  " + ", ".join(gap["matched_skills"]) if gap["matched_skills"] else "  None",
            f"\nMissing Skills ({gap['num_missing']}):",
            "  " + ", ".join(gap["missing_skills"]) if gap["missing_skills"] else "  None",
            f"\nTop Priority Gaps:",
            "  " + ", ".join(gap["priority_gaps"]) if gap["priority_gaps"] else "  None",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_gap_results(self, df: pd.DataFrame, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved skill gap results to {output_path}")


if __name__ == "__main__":
    analyzer = SkillGapAnalyzer(
        skill_weights={"python": 2.0, "machine learning": 2.0, "sql": 1.5}
    )
    p_skills = ["python", "sql", "communication", "excel"]
    j_skills = ["python", "machine learning", "sql", "deep learning", "tensorflow"]

    print(analyzer.generate_report("P001", p_skills, "JD001", j_skills))
