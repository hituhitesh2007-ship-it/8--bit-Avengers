# jd_parser.py
# TODO: implement
# nlp/jd_parser.py
# Parses Job Descriptions (JDs) to extract required skills, qualifications, and metadata

import re
import os
import json
import logging
import pandas as pd
from typing import List, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JDParser:
    """
    Parses job descriptions to extract:
    - Required and preferred skills
    - Minimum qualifications (education, years of experience)
    - Job title, industry, seniority level
    - Location / remote flag
    - Salary range (if mentioned)
    - Key responsibilities

    Input: raw JD text or a CSV/JSON file of JD records
    Output: structured DataFrame
    """

    SENIORITY_KEYWORDS = {
        "junior": ["junior", "entry level", "fresher", "graduate"],
        "mid": ["mid", "associate", "experienced", "3+ years", "2+ years"],
        "senior": ["senior", "lead", "principal", "5+ years", "7+ years"],
        "manager": ["manager", "head of", "director", "vp", "chief"]
    }

    SECTION_PATTERNS = {
        "requirements": r"(requirements?|qualifications?|what we(\'re| are) looking for|must have)",
        "responsibilities": r"(responsibilities|what you(\'ll| will) do|duties|your role)",
        "preferred": r"(preferred|nice to have|bonus|good to have|plus)"
    }

    EDUCATION_LEVELS = ["phd", "master", "mba", "bachelor", "b.sc", "b.tech", "diploma", "degree"]

    def __init__(self, skill_taxonomy: Optional[List[str]] = None):
        """
        Args:
            skill_taxonomy: Optional list of known skills for matching
        """
        self.skill_taxonomy = [s.lower() for s in (skill_taxonomy or [])]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_file(self, path: str, text_col: str = "description", id_col: str = "job_id") -> pd.DataFrame:
        """
        Load and parse all JDs from a CSV or JSON file.

        Args:
            path:     File path
            text_col: Column containing raw JD text
            id_col:   Column containing job ID

        Returns:
            DataFrame with parsed fields
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        ext = os.path.splitext(path)[-1].lower()
        if ext == ".csv":
            df_raw = pd.read_csv(path)
        elif ext == ".json":
            with open(path) as f:
                raw = json.load(f)
            df_raw = pd.DataFrame(raw if isinstance(raw, list) else raw.get("data", []))
        else:
            raise ValueError(f"Unsupported format: {ext}")

        records = []
        for _, row in df_raw.iterrows():
            text = str(row.get(text_col, ""))
            job_id = row.get(id_col, None)
            parsed = self.parse(text, job_id=job_id)
            # Carry through any additional columns
            for col in df_raw.columns:
                if col not in (text_col, id_col):
                    parsed[col] = row[col]
            records.append(parsed)

        logger.info(f"Parsed {len(records)} job descriptions from {path}.")
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Core Parsing
    # ------------------------------------------------------------------

    def parse(self, text: str, job_id: Optional[str] = None) -> dict:
        """
        Parse a single job description text.

        Returns:
            dict with structured fields
        """
        text_lower = text.lower()

        return {
            "job_id": job_id,
            "title": self._extract_title(text),
            "seniority_level": self._extract_seniority(text_lower),
            "required_skills": self._extract_skills(text_lower, section="requirements"),
            "preferred_skills": self._extract_skills(text_lower, section="preferred"),
            "all_skills_mentioned": self._extract_skills(text_lower),
            "min_years_experience": self._extract_years_experience(text_lower),
            "education_level": self._extract_education(text_lower),
            "is_remote": self._detect_remote(text_lower),
            "salary_range": self._extract_salary(text),
            "responsibilities": self._extract_section(text, "responsibilities"),
            "raw_length": len(text)
        }

    # ------------------------------------------------------------------
    # Field Extractors
    # ------------------------------------------------------------------

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract job title from first line of JD."""
        first_line = text.strip().split("\n")[0]
        return first_line.strip() if len(first_line) < 100 else None

    def _extract_seniority(self, text_lower: str) -> str:
        for level, keywords in self.SENIORITY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return level
        return "unknown"

    def _extract_skills(self, text_lower: str, section: Optional[str] = None) -> List[str]:
        """Extract skills from full text or a specific section."""
        if not self.skill_taxonomy:
            return []

        target = text_lower
        if section:
            target = self._extract_section_text(text_lower, section)

        return [skill for skill in self.skill_taxonomy if skill in target]

    def _extract_years_experience(self, text_lower: str) -> Optional[int]:
        """Extract minimum years of experience mentioned."""
        matches = re.findall(r"(\d+)\+?\s*years?\s*(of\s+)?(experience|exp)", text_lower)
        if matches:
            return min(int(m[0]) for m in matches)
        return None

    def _extract_education(self, text_lower: str) -> Optional[str]:
        """Return the highest detected education level required."""
        order = ["phd", "master", "mba", "bachelor", "b.tech", "b.sc", "diploma", "degree"]
        for level in order:
            if level in text_lower:
                return level
        return None

    def _detect_remote(self, text_lower: str) -> bool:
        return any(kw in text_lower for kw in ["remote", "work from home", "wfh", "fully remote"])

    def _extract_salary(self, text: str) -> Optional[str]:
        """Extract raw salary mention as string."""
        match = re.search(
            r"(\$|₹|€|£|USD|INR)?\s?[\d,]+\s?[-–to]+\s?[\d,]+\s?(per\s(year|month|annum|hr|hour))?",
            text, re.IGNORECASE
        )
        return match.group(0).strip() if match else None

    def _extract_section(self, text: str, section: str) -> List[str]:
        """Extract bullet points / sentences from a named section."""
        pattern = self.SECTION_PATTERNS.get(section, "")
        if not pattern:
            return []

        lines = text.split("\n")
        in_section = False
        items = []

        for line in lines:
            if re.search(pattern, line.lower()):
                in_section = True
                continue
            if in_section:
                if not line.strip():
                    continue
                # Stop at next section header
                if re.search(r"^[A-Z][A-Z\s]{3,}:", line):
                    break
                cleaned = re.sub(r"^[\-•*\d.]\s*", "", line).strip()
                if cleaned:
                    items.append(cleaned)

        return items[:15]  # cap at 15 bullets

    def _extract_section_text(self, text_lower: str, section: str) -> str:
        """Return raw text of a named section (lowercase)."""
        pattern = self.SECTION_PATTERNS.get(section, "")
        if not pattern:
            return text_lower
        match = re.search(pattern, text_lower)
        if not match:
            return ""
        return text_lower[match.start():]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_parsed(self, df: pd.DataFrame, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved parsed JD data to {output_path}")


if __name__ == "__main__":
    sample_jd = """
    Senior Data Scientist
    We are looking for a Senior Data Scientist with 5+ years of experience.
    Requirements:
    - Python, SQL, Machine Learning, Deep Learning
    - Bachelor's degree in Computer Science or related field
    Preferred:
    - Experience with PyTorch or TensorFlow
    - AWS certification
    Salary: $90,000 - $130,000 per year
    Remote: Yes
    """
    taxonomy = ["python", "sql", "machine learning", "deep learning", "pytorch", "tensorflow", "aws"]
    parser = JDParser(skill_taxonomy=taxonomy)
    result = parser.parse(sample_jd, job_id="JD001")
    import pprint; pprint.pprint(result)
