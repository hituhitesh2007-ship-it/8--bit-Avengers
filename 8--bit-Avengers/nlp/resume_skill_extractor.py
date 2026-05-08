# resume_skill_extractor.py
# TODO: implement
# nlp/resume_skill_extractor.py
# Extracts structured skills from raw resume text using NLP techniques

import re
import logging
import pandas as pd
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResumeSkillExtractor:
    """
    Extracts skills from resume text using:
    - Keyword/phrase matching against a curated skill taxonomy
    - SpaCy NER (if available) for entity recognition
    - Rule-based section parsing (Skills, Certifications, etc.)

    Output: structured list of skills per participant with confidence scores.
    """

    DEFAULT_SKILL_TAXONOMY = [
        # Technical
        "python", "java", "javascript", "sql", "r", "c++", "scala", "go",
        "machine learning", "deep learning", "data analysis", "data science",
        "nlp", "computer vision", "tensorflow", "pytorch", "scikit-learn",
        "tableau", "power bi", "excel", "powerpoint", "hadoop", "spark",
        "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
        # Soft Skills
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "time management", "project management",
        "negotiation", "presentation", "mentoring",
        # Domain Skills
        "accounting", "auditing", "nursing", "welding", "logistics",
        "graphic design", "ux design", "customer service", "sales",
        "marketing", "digital marketing", "seo", "supply chain",
        "financial analysis", "legal research", "teaching", "counseling"
    ]

    SECTION_HEADERS = [
        "skills", "technical skills", "core competencies",
        "key skills", "expertise", "technologies", "tools"
    ]

    def __init__(
        self,
        skill_taxonomy: Optional[List[str]] = None,
        use_spacy: bool = False,
        spacy_model: str = "en_core_web_sm"
    ):
        self.skill_taxonomy = [s.lower() for s in (skill_taxonomy or self.DEFAULT_SKILL_TAXONOMY)]
        self.use_spacy = use_spacy
        self.nlp = None

        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load(spacy_model)
                logger.info(f"SpaCy model '{spacy_model}' loaded.")
            except Exception as e:
                logger.warning(f"SpaCy not available: {e}. Falling back to keyword matching.")
                self.use_spacy = False

    # ------------------------------------------------------------------
    # Core Extraction
    # ------------------------------------------------------------------

    def extract(self, text: str, participant_id: Optional[str] = None) -> dict:
        """
        Extract skills from a single resume text.

        Args:
            text:           Raw resume text
            participant_id: Optional ID for tracking

        Returns:
            dict with participant_id, extracted_skills, skill_count, confidence_scores
        """
        text_lower = text.lower()

        keyword_skills = self._keyword_match(text_lower)
        section_skills = self._section_parse(text)
        combined = list(set(keyword_skills + section_skills))

        confidence_scores = self._compute_confidence(text_lower, combined)

        return {
            "participant_id": participant_id,
            "extracted_skills": combined,
            "skill_count": len(combined),
            "confidence_scores": confidence_scores,
            "source": "keyword+section" if not self.use_spacy else "keyword+section+spacy"
        }

    def extract_batch(self, records: List[dict], text_col: str = "raw_text", id_col: str = "participant_id") -> pd.DataFrame:
        """
        Extract skills from a list of resume records.

        Args:
            records:  List of dicts with text and optional ID
            text_col: Key containing resume text
            id_col:   Key containing participant ID

        Returns:
            DataFrame with extracted skill info
        """
        results = []
        for record in records:
            text = record.get(text_col, "")
            pid = record.get(id_col)
            result = self.extract(text, participant_id=pid)
            results.append(result)

        logger.info(f"Extracted skills from {len(results)} resumes.")
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _keyword_match(self, text_lower: str) -> List[str]:
        """Match skills from taxonomy using substring search."""
        return [skill for skill in self.skill_taxonomy if skill in text_lower]

    def _section_parse(self, text: str) -> List[str]:
        """
        Parse skills from dedicated 'Skills' sections using line-by-line heuristics.
        Captures comma/bullet separated skill lists under known section headers.
        """
        lines = text.split("\n")
        in_section = False
        section_skills = []

        for line in lines:
            line_stripped = line.strip().lower()

            # Detect section header
            if any(header in line_stripped for header in self.SECTION_HEADERS):
                in_section = True
                continue

            # Exit section on blank line after content or new all-caps header
            if in_section:
                if not line.strip():
                    in_section = False
                    continue
                # Try to parse comma/bullet separated items
                items = re.split(r"[,|•\-\n]", line)
                for item in items:
                    cleaned = item.strip().lower()
                    if cleaned and len(cleaned) < 50:
                        section_skills.append(cleaned)

        # Filter to known taxonomy
        return [s for s in section_skills if s in self.skill_taxonomy]

    def _compute_confidence(self, text_lower: str, skills: List[str]) -> dict:
        """
        Assign confidence scores based on:
        - Frequency of mention
        - Presence in a dedicated section (heuristic)
        """
        scores = {}
        for skill in skills:
            count = text_lower.count(skill)
            scores[skill] = min(round(0.5 + 0.1 * count, 2), 1.0)
        return scores

    # ------------------------------------------------------------------
    # Taxonomy Management
    # ------------------------------------------------------------------

    def add_skills(self, new_skills: List[str]) -> None:
        """Add new skills to the taxonomy."""
        additions = [s.lower() for s in new_skills if s.lower() not in self.skill_taxonomy]
        self.skill_taxonomy.extend(additions)
        logger.info(f"Added {len(additions)} new skills to taxonomy.")

    def get_taxonomy(self) -> List[str]:
        """Return the current skill taxonomy."""
        return sorted(self.skill_taxonomy)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_results(self, df: pd.DataFrame, output_path: str) -> None:
        """Save extracted skill results to CSV."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved skill extraction results to {output_path}")


if __name__ == "__main__":
    extractor = ResumeSkillExtractor()
    sample = {
        "participant_id": "P001",
        "raw_text": """
        John Doe | Python Developer
        Skills: Python, SQL, Machine Learning, Communication, Teamwork
        Experience: 3 years in data analysis and deep learning projects.
        Certifications: AWS Certified Solutions Architect
        """
    }
    result = extractor.extract(sample["raw_text"], participant_id=sample["participant_id"])
    print(result)
