# ingestion/resume_parser.py
# Parses resumes (PDF/DOCX/text) and extracts structured skill and experience data

import os
import re
import json
import logging
import pandas as pd
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResumeParser:
    """
    Parses resumes from PDF, DOCX, or plain text formats.
    Extracts:
    - Participant ID (from filename or metadata)
    - Skills mentioned
    - Education history
    - Work experience
    - Certifications listed
    - Contact region (if present)
    """

    SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

    SKILL_KEYWORDS = [
        "python", "sql", "machine learning", "data analysis", "excel",
        "communication", "project management", "java", "javascript",
        "accounting", "nursing", "welding", "logistics", "graphic design",
        "customer service", "leadership", "research", "teaching"
    ]

    def __init__(self, source_dir: str):
        """
        Args:
            source_dir: Folder containing resume files
        """
        self.source_dir = source_dir
        self.records = []

    def parse_all(self) -> pd.DataFrame:
        """Parse all resumes in the source directory."""
        if not os.path.exists(self.source_dir):
            raise FileNotFoundError(f"Directory not found: {self.source_dir}")

        files = [
            f for f in os.listdir(self.source_dir)
            if os.path.splitext(f)[-1].lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not files:
            logger.warning(f"No supported resume files found in {self.source_dir}")
            return pd.DataFrame()

        for filename in files:
            path = os.path.join(self.source_dir, filename)
            record = self._parse_file(path, filename)
            if record:
                self.records.append(record)

        logger.info(f"Parsed {len(self.records)} resumes.")
        return pd.DataFrame(self.records)

    def _parse_file(self, path: str, filename: str) -> Optional[dict]:
        """Dispatch to the correct parser based on file extension."""
        ext = os.path.splitext(filename)[-1].lower()
        participant_id = os.path.splitext(filename)[0]

        try:
            if ext == ".txt":
                text = self._read_txt(path)
            elif ext == ".pdf":
                text = self._read_pdf(path)
            elif ext == ".docx":
                text = self._read_docx(path)
            else:
                return None

            return self._extract_fields(text, participant_id)

        except Exception as e:
            logger.error(f"Failed to parse {filename}: {e}")
            return None

    def _read_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _read_pdf(self, path: str) -> str:
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except ImportError:
            logger.warning("pdfplumber not installed. Run: pip install pdfplumber")
            return ""

    def _read_docx(self, path: str) -> str:
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            logger.warning("python-docx not installed. Run: pip install python-docx")
            return ""

    def _extract_fields(self, text: str, participant_id: str) -> dict:
        """Extract structured fields from raw resume text."""
        text_lower = text.lower()

        # Skills extraction via keyword match
        found_skills = [
            skill for skill in self.SKILL_KEYWORDS
            if skill in text_lower
        ]

        # Email extraction
        email_match = re.search(r"[\w.-]+@[\w.-]+\.\w+", text)
        email = email_match.group(0) if email_match else None

        # Phone extraction
        phone_match = re.search(r"\+?[\d\s\-().]{7,15}", text)
        phone = phone_match.group(0).strip() if phone_match else None

        # Years of experience (heuristic)
        exp_match = re.search(r"(\d+)\s*\+?\s*years?\s*(of\s+)?(experience|exp)", text_lower)
        years_experience = int(exp_match.group(1)) if exp_match else None

        # Certifications mentioned
        cert_keywords = ["certified", "certificate", "certification", "diploma", "license"]
        cert_lines = [
            line.strip() for line in text.split("\n")
            if any(k in line.lower() for k in cert_keywords)
        ]

        # Education detection
        edu_keywords = ["bachelor", "master", "phd", "b.sc", "m.sc", "b.tech", "mba", "diploma"]
        education_lines = [
            line.strip() for line in text.split("\n")
            if any(k in line.lower() for k in edu_keywords)
        ]

        return {
            "participant_id": participant_id,
            "email": email,
            "phone": phone,
            "years_experience": years_experience,
            "skills_mentioned": found_skills,
            "num_skills": len(found_skills),
            "certifications_on_resume": cert_lines,
            "education_mentions": education_lines,
            "raw_text_length": len(text)
        }

    def get_skill_frequency(self) -> pd.DataFrame:
        """Return a frequency count of all skills across all parsed resumes."""
        if not self.records:
            raise RuntimeError("No records parsed yet. Call parse_all() first.")

        from collections import Counter
        all_skills = []
        for record in self.records:
            all_skills.extend(record.get("skills_mentioned", []))

        freq = Counter(all_skills)
        return pd.DataFrame(freq.items(), columns=["skill", "count"]).sort_values(
            "count", ascending=False
        ).reset_index(drop=True)

    def save_processed(self, output_path: str) -> None:
        """Save parsed resume data to CSV."""
        df = pd.DataFrame(self.records)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved parsed resume data to {output_path}")


if __name__ == "__main__":
    parser = ResumeParser(source_dir="data/raw/resumes")
    df = parser.parse_all()
    print(df.head())
    print(parser.get_skill_frequency())
    parser.save_processed("data/processed/resumes_parsed.csv")