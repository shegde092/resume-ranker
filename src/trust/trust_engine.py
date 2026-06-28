import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class TrustEngine:
    def __init__(self):
        self.template_centroids = None
        self.anomaly_threshold = 3.0
        self.optimal_k = 44  # Fixed architecture constraint

    def evaluate(
        self,
        parsed_resume: Dict[str, Any],
        vector_features: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Evaluate trust score using soft penalties.
        Returns:
        {
            "trust_score": float,
            "penalties": list
        }
        """
        penalties = []
        base_trust = 1.0

        career_history = parsed_resume.get("career_history", [])
        education = parsed_resume.get("education", [])
        redrob_signals = parsed_resume.get("redrob_signals", {})

        # 1. Overlapping jobs
        overlap_penalty = self._check_overlaps(career_history)
        if overlap_penalty > 0:
            penalties.append("overlapping_fulltime_roles")
            base_trust -= overlap_penalty

        # 2. Suspicious education duration
        education_penalty = self._check_education_duration(education)
        if education_penalty > 0:
            penalties.append("suspicious_degree_duration")
            base_trust -= education_penalty

        # 3. Career template anomaly
        anomaly_penalty = self._detect_template_anomaly(vector_features)
        if anomaly_penalty > 0:
            penalties.append("abnormal_career_progression")
            base_trust -= anomaly_penalty

        # 4. Recruiter behavior signals
        response_rate = redrob_signals.get("recruiter_response_rate", 1.0)
        if response_rate < 0.1:
            penalties.append("low_response_rate")
            base_trust -= 0.1

        trust_score = float(max(0.0, min(1.0, base_trust)))

        return {
            "trust_score": trust_score,
            "penalties": penalties
        }

    def _check_overlaps(self, career_history: List[Dict[str, Any]]) -> float:
        """
        Soft penalty for multiple active roles.
        """
        active_roles = [
            role for role in career_history
            if role.get("end_date") is None
        ]

        if len(active_roles) >= 2:
            return 0.2

        return 0.0

    def _check_education_duration(
        self,
        education: List[Dict[str, Any]]
    ) -> float:
        """
        Soft penalty for suspicious degree duration.
        """
        penalty = 0.0

        for edu in education:
            start = edu.get("start_year")
            end = edu.get("end_year")
            degree = str(edu.get("degree", "")).lower()

            if start is None or end is None:
                continue

            try:
                duration = int(end) - int(start)

                if "phd" in degree and duration < 2:
                    penalty += 0.15

                elif ("b.e" in degree or "btech" in degree or "bachelor" in degree) and duration < 3:
                    penalty += 0.1

            except (TypeError, ValueError):
                continue

        return penalty

    def _detect_template_anomaly(
        self,
        vector_features: np.ndarray
    ) -> float:
        """
        Detect anomalous career trajectory using centroid distance.
        """
        if vector_features is None:
            return 0.0

        if self.template_centroids is None:
            logger.warning("Template centroids not initialized.")
            return 0.0

        distances = cdist(
            vector_features.reshape(1, -1),
            self.template_centroids,
            metric="euclidean"
        )

        min_dist = float(np.min(distances))

        if min_dist > self.anomaly_threshold:
            return 0.2

        return 0.0

    def cluster_historical_trajectories(
        self,
        historical_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Build exactly 44 career templates using KMeans.
        """
        if historical_df.empty:
            raise ValueError("Historical trajectory dataframe is empty.")

        features = historical_df.values

        clusterer = KMeans(
            n_clusters=self.optimal_k,
            random_state=42,
            n_init=10
        )

        labels = clusterer.fit_predict(features)
        self.template_centroids = clusterer.cluster_centers_

        # Metrics
        if len(set(labels)) > 1:
            sil_score = float(silhouette_score(features, labels))
            db_score = float(davies_bouldin_score(features, labels))
        else:
            sil_score = -1.0
            db_score = -1.0

        # Dynamic anomaly threshold (95 percentile)
        assigned_centroids = self.template_centroids[labels]
        distances = np.linalg.norm(features - assigned_centroids, axis=1)
        self.anomaly_threshold = float(np.percentile(distances, 95))

        return {
            "optimal_clusters": self.optimal_k,
            "silhouette_score": sil_score,
            "davies_bouldin_index": db_score
        }
