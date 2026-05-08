# topic_modeler.py
# TODO: implement
# nlp/topic_modeler.py
# Performs topic modeling on resumes, JDs, community posts, and feedback text

import os
import re
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicModeler:
    """
    Discovers latent topics in a text corpus using:
    - LDA (Latent Dirichlet Allocation) via scikit-learn
    - Optional NMF (Non-Negative Matrix Factorization) for cleaner topics
    - BERTopic (if installed) for transformer-based topic modeling

    Use cases:
    - Discover recurring themes in participant feedback
    - Cluster job descriptions by domain
    - Identify common skill clusters across resumes
    - Map community forum discussions to career topics
    """

    STOPWORDS = {
        "i", "me", "my", "we", "our", "you", "he", "she", "they", "it",
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "to", "of", "and", "in", "that", "for", "on", "with", "as",
        "this", "at", "by", "from", "or", "but", "not", "have", "has",
        "do", "did", "will", "would", "can", "could", "should", "may",
        "also", "more", "very", "so", "if", "about", "up", "out", "their"
    }

    def __init__(
        self,
        n_topics: int = 10,
        method: str = "lda",
        max_features: int = 5000,
        n_top_words: int = 10,
        random_state: int = 42
    ):
        """
        Args:
            n_topics:    Number of topics to extract
            method:      'lda', 'nmf', or 'bertopic'
            max_features: Max vocabulary size for TF-IDF
            n_top_words: Number of top words to show per topic
            random_state: For reproducibility
        """
        self.n_topics = n_topics
        self.method = method
        self.max_features = max_features
        self.n_top_words = n_top_words
        self.random_state = random_state

        self._vectorizer = None
        self._model = None
        self._feature_names = None
        self._doc_topic_matrix = None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, texts: List[str]) -> List[str]:
        """
        Basic text cleaning:
        - Lowercase
        - Remove punctuation and digits
        - Remove stopwords
        - Remove very short tokens
        """
        cleaned = []
        for text in texts:
            text = text.lower()
            text = re.sub(r"[^a-z\s]", " ", text)
            tokens = [
                t for t in text.split()
                if t not in self.STOPWORDS and len(t) > 2
            ]
            cleaned.append(" ".join(tokens))
        return cleaned

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]) -> "TopicModeler":
        """
        Fit the topic model on a corpus.

        Args:
            texts: List of raw document strings

        Returns:
            self (for chaining)
        """
        cleaned = self.preprocess(texts)

        if self.method in ("lda", "nmf"):
            self._fit_sklearn(cleaned)
        elif self.method == "bertopic":
            self._fit_bertopic(texts)  # BERTopic works better on raw text
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'lda', 'nmf', or 'bertopic'.")

        logger.info(f"Topic model fitted with {self.n_topics} topics using {self.method.upper()}.")
        return self

    def _fit_sklearn(self, cleaned_texts: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation, NMF

        if self.method == "lda":
            self._vectorizer = CountVectorizer(max_features=self.max_features, min_df=2)
            dtm = self._vectorizer.fit_transform(cleaned_texts)
            self._model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=20
            )
        else:  # nmf
            self._vectorizer = TfidfVectorizer(max_features=self.max_features, min_df=2)
            dtm = self._vectorizer.fit_transform(cleaned_texts)
            self._model = NMF(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=300
            )

        self._doc_topic_matrix = self._model.fit_transform(dtm)
        self._feature_names = self._vectorizer.get_feature_names_out()

    def _fit_bertopic(self, texts: List[str]):
        try:
            from bertopic import BERTopic
            self._model = BERTopic(nr_topics=self.n_topics, verbose=False)
            topics, _ = self._model.fit_transform(texts)
            self._doc_topic_matrix = np.array(topics).reshape(-1, 1)
            logger.info("BERTopic model fitted.")
        except ImportError:
            logger.error("BERTopic not installed. Run: pip install bertopic")
            raise

    # ------------------------------------------------------------------
    # Topic Inspection
    # ------------------------------------------------------------------

    def get_topics(self) -> List[dict]:
        """
        Return top words for each topic.

        Returns:
            List of dicts: {topic_id, top_words}
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.method == "bertopic":
            topic_info = self._model.get_topic_info()
            return [
                {
                    "topic_id": row["Topic"],
                    "top_words": self._model.get_topic(row["Topic"])[:self.n_top_words]
                }
                for _, row in topic_info.iterrows()
            ]

        topics = []
        for topic_idx, topic in enumerate(self._model.components_):
            top_word_ids = topic.argsort()[: -self.n_top_words - 1: -1]
            top_words = [self._feature_names[i] for i in top_word_ids]
            topics.append({
                "topic_id": topic_idx,
                "top_words": top_words,
                "top_words_str": ", ".join(top_words)
            })
        return topics

    def get_document_topics(self, texts: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Return dominant topic assignment per document.

        Args:
            texts: Optional list of document strings (uses training docs if None)

        Returns:
            DataFrame with dominant_topic and topic_score per document
        """
        if self._doc_topic_matrix is None:
            raise RuntimeError("Model not fitted.")

        matrix = self._doc_topic_matrix
        if texts is not None and self.method != "bertopic":
            cleaned = self.preprocess(texts)
            dtm = self._vectorizer.transform(cleaned)
            matrix = self._model.transform(dtm)

        dominant_topic = np.argmax(matrix, axis=1)
        topic_score = np.max(matrix, axis=1)

        return pd.DataFrame({
            "dominant_topic": dominant_topic,
            "topic_score": np.round(topic_score, 4)
        })

    def label_topics(self, custom_labels: dict) -> None:
        """
        Assign human-readable labels to topic IDs.

        Args:
            custom_labels: dict of {topic_id: label_string}
                           e.g. {0: "Data Science", 1: "Healthcare"}
        """
        self._topic_labels = custom_labels
        logger.info(f"Applied {len(custom_labels)} custom topic labels.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Infer topic distributions for new texts.

        Args:
            texts: List of new document strings

        Returns:
            Document-topic matrix of shape (N, n_topics)
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        cleaned = self.preprocess(texts)
        dtm = self._vectorizer.transform(cleaned)
        return self._model.transform(dtm)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_topics(self) -> None:
        """Pretty-print all topics."""
        for t in self.get_topics():
            label = getattr(self, "_topic_labels", {}).get(t["topic_id"], f"Topic {t['topic_id']}")
            words = t.get("top_words_str", str(t.get("top_words", "")))
            print(f"[{label}]: {words}")

    def get_topic_summary_df(self) -> pd.DataFrame:
        """Return topics as a DataFrame."""
        topics = self.get_topics()
        return pd.DataFrame([
            {
                "topic_id": t["topic_id"],
                "label": getattr(self, "_topic_labels", {}).get(t["topic_id"], f"Topic {t['topic_id']}"),
                "top_words": t.get("top_words_str", ", ".join(str(w) for w in t.get("top_words", [])))
            }
            for t in topics
        ])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, output_dir: str) -> None:
        """Save the fitted model and vectorizer to disk."""
        import joblib
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self._model, os.path.join(output_dir, "topic_model.pkl"))
        if self._vectorizer:
            joblib.dump(self._vectorizer, os.path.join(output_dir, "vectorizer.pkl"))
        logger.info(f"Topic model saved to {output_dir}")

    def load_model(self, output_dir: str) -> None:
        """Load a previously saved model."""
        import joblib
        self._model = joblib.load(os.path.join(output_dir, "topic_model.pkl"))
        vec_path = os.path.join(output_dir, "vectorizer.pkl")
        if os.path.exists(vec_path):
            self._vectorizer = joblib.load(vec_path)
            self._feature_names = self._vectorizer.get_feature_names_out()
        logger.info(f"Topic model loaded from {output_dir}")


if __name__ == "__main__":
    corpus = [
        "Python machine learning data science deep learning neural networks",
        "Nursing healthcare patient care clinical training hospital",
        "Welding fabrication metal construction safety equipment",
        "Sales marketing customer engagement digital media SEO content",
        "Logistics supply chain procurement inventory warehouse operations",
        "Accounting finance budgeting auditing tax compliance",
        "Teaching curriculum development classroom management education",
        "Java software engineering backend API REST microservices",
        "Graphic design UI UX Adobe Photoshop Illustrator Figma",
        "Customer service communication problem solving CRM support"
    ]

    modeler = TopicModeler(n_topics=5, method="lda")
    modeler.fit(corpus * 5)  # replicate for meaningful training
    modeler.print_topics()

    doc_topics = modeler.get_document_topics()
    print(doc_topics.head())
