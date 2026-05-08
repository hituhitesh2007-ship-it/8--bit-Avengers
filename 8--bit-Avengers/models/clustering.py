# clustering.py
# TODO: implement
# models/clustering.py
# Segments participants into meaningful clusters for targeted interventions

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParticipantClusterer:
    """
    Segments participants into clusters for targeted interventions and analysis.

    Clustering Algorithms:
    - KMeans: Fast, interpretable, good for balanced clusters
    - DBSCAN: Density-based, handles outliers
    - Agglomerative: Hierarchical, good for nested structures

    Cluster profiles can be used to:
    - Design group-specific intervention programs
    - Identify underserved segments
    - Route participants to appropriate resources
    - Benchmark cohort outcomes against similar peers
    """

    FEATURE_COLUMNS = [
        "skill_count", "years_experience", "num_certifications",
        "education_level_encoded", "employment_status_encoded",
        "community_engagement_score", "sentiment_score",
        "network_strength", "gap_score", "coverage_ratio",
        "num_barriers", "utilization_score", "days_to_employment",
        "region_unemployment_rate"
    ]

    CLUSTER_PROFILE_LABELS = {
        "high_skill_employed":        "High skill, employed, low barriers",
        "high_skill_unemployed":      "High skill, unemployed, systemic barriers",
        "low_skill_underemployed":    "Low skill, underemployed, needs upskilling",
        "mid_skill_transitioning":    "Mid-skill, in career transition",
        "disengaged_at_risk":         "Low engagement, high dropout risk",
        "early_career_motivated":     "Early career, high potential, low experience"
    }

    def __init__(
        self,
        method: str = "kmeans",
        n_clusters: int = 6,
        random_state: int = 42,
        pca_components: Optional[int] = None
    ):
        """
        Args:
            method:         'kmeans', 'dbscan', or 'agglomerative'
            n_clusters:     Number of clusters (not used by DBSCAN)
            random_state:   Seed
            pca_components: If set, apply PCA before clustering
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.pca_components = pca_components

        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components) if pca_components else None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.cluster_profiles: Optional[pd.DataFrame] = None
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
                elif col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[enc_col] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        return df

    def _prepare(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        df = self._encode(df, fit=fit)
        available = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        X = df[available].fillna(0).values

        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        if self.pca:
            if fit:
                X = self.pca.fit_transform(X)
            else:
                X = self.pca.transform(X)

        return X

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ParticipantClusterer":
        """
        Fit the clustering model on participant data.

        Args:
            df: Input DataFrame with participant features

        Returns:
            self
        """
        X = self._prepare(df, fit=True)

        if self.method == "kmeans":
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
        elif self.method == "dbscan":
            self.model = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
        elif self.method == "agglomerative":
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        else:
            raise ValueError(f"Unknown method '{self.method}'.")

        labels = self.model.fit_predict(X)

        # Attach labels back
        df = df.copy()
        df["cluster"] = labels

        # Compute cluster profiles
        self.cluster_profiles = self._compute_profiles(df)
        self._is_fitted = True

        logger.info(
            f"Clustering complete. Method: {self.method} | "
            f"Clusters: {len(np.unique(labels))} | "
            f"Samples: {len(df)}"
        )
        return self

    # ------------------------------------------------------------------
    # Cluster Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Compute clustering quality metrics.

        Returns:
            dict with silhouette score and Davies-Bouldin index
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        X = self._prepare(df, fit=False)
        labels = self.predict(df)["cluster"].values

        n_unique = len(np.unique(labels[labels >= 0]))
        if n_unique < 2:
            return {"silhouette_score": None, "davies_bouldin": None, "n_clusters": n_unique}

        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)

        return {
            "silhouette_score": round(sil, 4),
            "davies_bouldin_index": round(db, 4),
            "n_clusters": n_unique,
            "cluster_sizes": pd.Series(labels).value_counts().to_dict()
        }

    def find_optimal_k(
        self,
        df: pd.DataFrame,
        k_range: Tuple[int, int] = (2, 12)
    ) -> pd.DataFrame:
        """
        Run KMeans for a range of k values and return elbow/silhouette metrics.

        Args:
            df:      Training data
            k_range: (min_k, max_k) range to evaluate

        Returns:
            DataFrame with k, inertia, silhouette_score
        """
        X = self._prepare(df, fit=True)
        results = []

        for k in range(k_range[0], k_range[1] + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels) if k > 1 else 0.0
            results.append({
                "k": k,
                "inertia": round(km.inertia_, 2),
                "silhouette_score": round(sil, 4)
            })

        logger.info(f"Optimal K search complete for k={k_range[0]} to {k_range[1]}.")
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign cluster labels to new participants.

        Returns:
            DataFrame with participant_id and cluster columns
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        X = self._prepare(df, fit=False)
        labels = self.model.predict(X) if hasattr(self.model, "predict") else self.model.fit_predict(X)

        result = pd.DataFrame()
        if "participant_id" in df.columns:
            result["participant_id"] = df["participant_id"].values
        result["cluster"] = labels
        return result

    def predict_with_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict clusters and join cluster profile descriptions."""
        pred = self.predict(df)
        if self.cluster_profiles is not None:
            pred = pred.merge(
                self.cluster_profiles[["cluster", "profile_label", "description"]],
                on="cluster", how="left"
            )
        return pred

    # ------------------------------------------------------------------
    # Cluster Profiles
    # ------------------------------------------------------------------

    def _compute_profiles(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean feature values per cluster for profiling."""
        available = [c for c in self.FEATURE_COLUMNS if c in labeled_df.columns]
        profiles = (
            labeled_df.groupby("cluster")[available]
            .mean()
            .round(3)
            .reset_index()
        )
        profiles["size"] = labeled_df.groupby("cluster").size().values
        profiles["profile_label"] = profiles["cluster"].apply(
            lambda c: f"Cluster_{c}"
        )
        profiles["description"] = "Auto-generated cluster"
        return profiles

    def label_clusters(self, cluster_labels: Dict[int, str]) -> None:
        """
        Apply human-readable labels to cluster IDs.

        Args:
            cluster_labels: {cluster_id: "label string"}
        """
        if self.cluster_profiles is None:
            raise RuntimeError("Fit the model first.")
        self.cluster_profiles["profile_label"] = self.cluster_profiles["cluster"].map(
            lambda c: cluster_labels.get(c, f"Cluster_{c}")
        )
        logger.info(f"Applied labels to {len(cluster_labels)} clusters.")

    def get_cluster_profile(self, cluster_id: int) -> Optional[dict]:
        """Return the profile of a specific cluster."""
        if self.cluster_profiles is None:
            return None
        row = self.cluster_profiles[self.cluster_profiles["cluster"] == cluster_id]
        return row.iloc[0].to_dict() if not row.empty else None

    def get_cluster_summary(self) -> pd.DataFrame:
        """Return full cluster profile table."""
        if self.cluster_profiles is None:
            raise RuntimeError("Fit the model first.")
        return self.cluster_profiles

    # ------------------------------------------------------------------
    # PCA Visualization Data
    # ------------------------------------------------------------------

    def get_pca_plot_data(self, df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
        """
        Return 2D PCA projection of feature space with cluster labels for visualization.

        Returns:
            DataFrame with pca_1, pca_2, cluster columns
        """
        X = self._prepare(df, fit=False)
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_2d = pca.fit_transform(X)

        labels = self.predict(df)["cluster"].values

        result = pd.DataFrame({
            "pca_1": X_2d[:, 0].round(4),
            "pca_2": X_2d[:, 1].round(4),
            "cluster": labels
        })

        if "participant_id" in df.columns:
            result.insert(0, "participant_id", df["participant_id"].values)

        logger.info(
            f"PCA explained variance: "
            f"{pca.explained_variance_ratio_.cumsum()[-1] * 100:.1f}%"
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(output_dir, "cluster_model.pkl"))
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(output_dir, "label_encoders.pkl"))
        if self.pca:
            joblib.dump(self.pca, os.path.join(output_dir, "pca.pkl"))
        if self.cluster_profiles is not None:
            self.cluster_profiles.to_csv(
                os.path.join(output_dir, "cluster_profiles.csv"), index=False
            )
        logger.info(f"ParticipantClusterer saved to {output_dir}")

    def load(self, output_dir: str) -> "ParticipantClusterer":
        self.model = joblib.load(os.path.join(output_dir, "cluster_model.pkl"))
        self.scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))
        self.label_encoders = joblib.load(os.path.join(output_dir, "label_encoders.pkl"))
        pca_path = os.path.join(output_dir, "pca.pkl")
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)
        profiles_path = os.path.join(output_dir, "cluster_profiles.csv")
        if os.path.exists(profiles_path):
            self.cluster_profiles = pd.read_csv(profiles_path)
        self._is_fitted = True
        logger.info(f"ParticipantClusterer loaded from {output_dir}")
        return self


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    dummy = pd.DataFrame({
        "participant_id": [f"P{i:04d}" for i in range(n)],
        "skill_count": np.random.randint(2, 20, n),
        "years_experience": np.random.randint(0, 15, n),
        "num_certifications": np.random.randint(0, 6, n),
        "education_level": np.random.choice(["diploma", "bachelor", "master"], n),
        "employment_status": np.random.choice(["employed", "unemployed", "underemployed"], n),
        "community_engagement_score": np.random.uniform(0, 1, n),
        "sentiment_score": np.random.uniform(-1, 1, n),
        "network_strength": np.random.uniform(0, 1, n),
        "gap_score": np.random.uniform(0, 1, n),
        "coverage_ratio": np.random.uniform(0, 1, n),
        "num_barriers": np.random.randint(0, 5, n),
        "utilization_score": np.random.uniform(0, 1, n),
        "days_to_employment": np.random.randint(0, 365, n),
        "region_unemployment_rate": np.random.uniform(3, 20, n)
    })

    clusterer = ParticipantClusterer(method="kmeans", n_clusters=6)
    clusterer.fit(dummy)

    preds = clusterer.predict_with_profile(dummy.head(10))
    print(preds[["participant_id", "cluster", "profile_label"]])

    print("\nCluster Summary:")
    print(clusterer.get_cluster_summary()[["cluster", "size", "skill_count", "utilization_score"]])

    print("\nEvaluation:")
    print(clusterer.evaluate(dummy))
