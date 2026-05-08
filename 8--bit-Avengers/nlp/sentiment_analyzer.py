# sentiment_analyzer.py
# TODO: implement
# nlp/sentiment_analyzer.py
# Analyzes sentiment in participant feedback, community posts, and counselor notes

import os
import re
import logging
import pandas as pd
from typing import List, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyzes sentiment in text data from:
    - Participant feedback surveys
    - Community forum posts
    - Counselor/mentor session notes
    - Employer feedback on participants

    Modes:
    1. Rule-based VADER (fast, no GPU needed)
    2. Transformer-based (higher accuracy, requires transformers library)

    Output:
    - Sentiment label: positive / neutral / negative
    - Compound score: -1.0 to +1.0
    - Confidence score
    - Detected emotion keywords
    """

    EMOTION_KEYWORDS = {
        "anxiety": ["nervous", "anxious", "worried", "scared", "fear", "dread"],
        "confidence": ["confident", "capable", "ready", "prepared", "motivated"],
        "frustration": ["frustrated", "stuck", "difficult", "hard", "failing", "lost"],
        "hope": ["hopeful", "excited", "looking forward", "optimistic", "positive"],
        "disengagement": ["bored", "uninterested", "disconnected", "dropout", "quit"]
    }

    def __init__(self, mode: str = "vader"):
        """
        Args:
            mode: 'vader' for rule-based (default) or 'transformer' for HuggingFace
        """
        self.mode = mode
        self._vader = None
        self._transformer = None
        self._init_model()

    def _init_model(self):
        if self.mode == "vader":
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
                logger.info("VADER sentiment analyzer initialized.")
            except ImportError:
                logger.warning(
                    "vaderSentiment not installed. Run: pip install vaderSentiment\n"
                    "Falling back to lexicon-based fallback."
                )
                self.mode = "lexicon"

        elif self.mode == "transformer":
            try:
                from transformers import pipeline
                self._transformer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                logger.info("Transformer sentiment pipeline initialized.")
            except ImportError:
                logger.warning(
                    "transformers not installed. Falling back to VADER."
                )
                self.mode = "vader"
                self._init_model()

    # ------------------------------------------------------------------
    # Core Analysis
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of a single text string.

        Returns:
            dict with label, score, confidence, emotions
        """
        if self.mode == "vader" and self._vader:
            return self._analyze_vader(text)
        elif self.mode == "transformer" and self._transformer:
            return self._analyze_transformer(text)
        else:
            return self._analyze_lexicon(text)

    def analyze_batch(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        id_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze sentiment for all rows in a DataFrame.

        Args:
            df:       DataFrame containing text
            text_col: Column with text content
            id_col:   Optional ID column to preserve

        Returns:
            DataFrame with sentiment fields appended
        """
        results = []
        for _, row in df.iterrows():
            text = str(row.get(text_col, ""))
            result = self.analyze(text)
            record = {}
            if id_col and id_col in row:
                record[id_col] = row[id_col]
            record.update(result)
            results.append(record)

        logger.info(f"Analyzed sentiment for {len(results)} records.")
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Mode-specific Analyzers
    # ------------------------------------------------------------------

    def _analyze_vader(self, text: str) -> dict:
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]
        label = (
            "positive" if compound >= 0.05
            else "negative" if compound <= -0.05
            else "neutral"
        )
        return {
            "sentiment_label": label,
            "compound_score": round(compound, 4),
            "positive_score": round(scores["pos"], 4),
            "negative_score": round(scores["neg"], 4),
            "neutral_score": round(scores["neu"], 4),
            "confidence": round(abs(compound), 4),
            "detected_emotions": self._detect_emotions(text)
        }

    def _analyze_transformer(self, text: str) -> dict:
        # Truncate to 512 tokens for model limit
        truncated = text[:1024]
        result = self._transformer(truncated)[0]
        label = result["label"].lower()
        score = result["score"]
        compound = score if label == "positive" else -score

        return {
            "sentiment_label": label,
            "compound_score": round(compound, 4),
            "confidence": round(score, 4),
            "detected_emotions": self._detect_emotions(text)
        }

    def _analyze_lexicon(self, text: str) -> dict:
        """Simple positive/negative word count fallback."""
        positive_words = {"good", "great", "excellent", "happy", "success", "motivated",
                          "confident", "proud", "improve", "better", "hopeful"}
        negative_words = {"bad", "poor", "fail", "sad", "frustrated", "difficult",
                          "struggle", "problem", "issue", "worried", "lost"}

        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        total = pos_count + neg_count

        if total == 0:
            compound = 0.0
        else:
            compound = (pos_count - neg_count) / total

        label = "positive" if compound > 0.1 else "negative" if compound < -0.1 else "neutral"

        return {
            "sentiment_label": label,
            "compound_score": round(compound, 4),
            "confidence": round(abs(compound), 4),
            "detected_emotions": self._detect_emotions(text)
        }

    # ------------------------------------------------------------------
    # Emotion Detection
    # ------------------------------------------------------------------

    def _detect_emotions(self, text: str) -> List[str]:
        """Detect emotion categories present in text via keyword matching."""
        text_lower = text.lower()
        detected = [
            emotion for emotion, keywords in self.EMOTION_KEYWORDS.items()
            if any(kw in text_lower for kw in keywords)
        ]
        return detected

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def get_sentiment_summary(self, df: pd.DataFrame, label_col: str = "sentiment_label") -> dict:
        """Return distribution of sentiment labels."""
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found.")
        counts = df[label_col].value_counts().to_dict()
        total = len(df)
        return {
            "total": total,
            "distribution": counts,
            "pct_positive": round(counts.get("positive", 0) / total * 100, 1),
            "pct_negative": round(counts.get("negative", 0) / total * 100, 1),
            "pct_neutral": round(counts.get("neutral", 0) / total * 100, 1),
        }

    def get_at_risk_participants(
        self,
        df: pd.DataFrame,
        id_col: str = "participant_id",
        score_col: str = "compound_score",
        threshold: float = -0.3
    ) -> pd.DataFrame:
        """
        Return participants with consistently negative sentiment — potential dropout risk.

        Args:
            df:         Results DataFrame from analyze_batch
            id_col:     Participant ID column
            score_col:  Compound score column
            threshold:  Score below which participant is considered at-risk

        Returns:
            DataFrame of at-risk participants with avg score
        """
        if score_col not in df.columns:
            raise ValueError(f"'{score_col}' not found in DataFrame.")

        at_risk = (
            df.groupby(id_col)[score_col]
            .mean()
            .reset_index()
            .rename(columns={score_col: "avg_sentiment_score"})
        )
        return at_risk[at_risk["avg_sentiment_score"] < threshold].sort_values("avg_sentiment_score")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_results(self, df: pd.DataFrame, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved sentiment results to {output_path}")


if __name__ == "__main__":
    analyzer = SentimentAnalyzer(mode="vader")
    samples = [
        "I am really excited and confident about the new job opportunity!",
        "I feel stuck and frustrated. The training was not helpful at all.",
        "The session was okay. Nothing special."
    ]
    for s in samples:
        print(analyzer.analyze(s))
