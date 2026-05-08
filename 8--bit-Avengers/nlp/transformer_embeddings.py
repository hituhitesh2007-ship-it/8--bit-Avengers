# transformer_embeddings.py
# TODO: implement
# nlp/transformer_embeddings.py
# Generates dense vector embeddings for skills, JDs, and resumes using transformers

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerEmbeddings:
    """
    Generates sentence/document-level embeddings using pre-trained transformer models.

    Use cases:
    - Embed skill phrases for semantic similarity matching
    - Embed resume text and JD text for ranking
    - Embed participant profiles for clustering
    - Compute semantic skill gap (embedding distance)

    Default model: sentence-transformers/all-MiniLM-L6-v2
    (fast, lightweight, 384-dim embeddings)
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        """
        Args:
            model_name: HuggingFace model name or path
            device:     'cpu' or 'cuda'
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded embedding model: {self.model_name} on {self.device}")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            raise

    # ------------------------------------------------------------------
    # Embedding Generation
    # ------------------------------------------------------------------

    def embed(self, texts: Union[str, List[str]], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for one or more texts.

        Args:
            texts:      Single string or list of strings
            batch_size: Batch size for encoding
            normalize:  L2-normalize embeddings (recommended for cosine similarity)

        Returns:
            numpy array of shape (N, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100
        )

        logger.info(f"Generated embeddings for {len(texts)} texts. Shape: {embeddings.shape}")
        return embeddings

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str,
        id_col: Optional[str] = None,
        prefix: str = "emb"
    ) -> pd.DataFrame:
        """
        Embed a text column in a DataFrame and return expanded embedding columns.

        Args:
            df:       Source DataFrame
            text_col: Column containing text to embed
            id_col:   Optional ID column to preserve
            prefix:   Prefix for embedding column names

        Returns:
            DataFrame with ID column + embedding dimensions as columns
        """
        texts = df[text_col].fillna("").tolist()
        embeddings = self.embed(texts)

        emb_cols = [f"{prefix}_{i}" for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols)

        if id_col and id_col in df.columns:
            emb_df.insert(0, id_col, df[id_col].values)

        return emb_df

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)

        norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)
        return float(np.dot(emb1 / (norm1 + 1e-9), (emb2 / (norm2 + 1e-9)).T)[0][0])

    def pairwise_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Compute a pairwise cosine similarity matrix for a list of texts.

        Returns:
            NxN numpy array
        """
        embeddings = self.embed(texts)
        sim_matrix = np.dot(embeddings, embeddings.T)
        logger.info(f"Pairwise similarity matrix shape: {sim_matrix.shape}")
        return sim_matrix

    def top_k_similar(
        self,
        query: str,
        candidates: List[str],
        k: int = 5
    ) -> List[dict]:
        """
        Find top-K most semantically similar candidates to a query.

        Args:
            query:      Query string (e.g. a skill or JD snippet)
            candidates: List of candidate strings to rank
            k:          Number of top results to return

        Returns:
            List of dicts with text and similarity score
        """
        query_emb = self.embed(query)[0]
        candidate_embs = self.embed(candidates)

        scores = np.dot(candidate_embs, query_emb)
        top_idx = np.argsort(scores)[::-1][:k]

        return [
            {"text": candidates[i], "similarity": round(float(scores[i]), 4)}
            for i in top_idx
        ]

    # ------------------------------------------------------------------
    # Skill Matching
    # ------------------------------------------------------------------

    def match_skills_semantically(
        self,
        participant_skills: List[str],
        required_skills: List[str],
        threshold: float = 0.75
    ) -> dict:
        """
        Match participant skills to required skills using semantic similarity.

        Returns:
            dict with matched pairs and unmatched required skills
        """
        matched = []
        unmatched = []

        p_embs = self.embed(participant_skills)
        r_embs = self.embed(required_skills)

        for i, req_skill in enumerate(required_skills):
            sims = np.dot(p_embs, r_embs[i])
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])

            if best_score >= threshold:
                matched.append({
                    "required_skill": req_skill,
                    "matched_participant_skill": participant_skills[best_idx],
                    "similarity": round(best_score, 4)
                })
            else:
                unmatched.append(req_skill)

        return {
            "semantically_matched": matched,
            "unmatched_required": unmatched,
            "semantic_coverage": round(len(matched) / len(required_skills), 4) if required_skills else 0.0
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_embeddings(self, embeddings: np.ndarray, path: str) -> None:
        """Save embeddings as a .npy file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, embeddings)
        logger.info(f"Saved embeddings to {path}. Shape: {embeddings.shape}")

    def load_embeddings(self, path: str) -> np.ndarray:
        """Load embeddings from a .npy file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found: {path}")
        embeddings = np.load(path)
        logger.info(f"Loaded embeddings from {path}. Shape: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    embedder = TransformerEmbeddings()

    skills = ["python programming", "data analysis", "machine learning", "communication"]
    required = ["python", "deep learning", "teamwork", "sql"]

    result = embedder.match_skills_semantically(skills, required, threshold=0.7)
    import pprint; pprint.pprint(result)

    top = embedder.top_k_similar("machine learning engineer", skills, k=3)
    print("\nTop-K Similar to 'machine learning engineer':")
    pprint.pprint(top)
