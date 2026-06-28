import pandas as pd
import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class FeaturePersistence:
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_candidate_features(self, features_list: List[Dict[str, Any]]):
        if not features_list:
            return
        df = pd.DataFrame(features_list)
        path = os.path.join(self.output_dir, "candidate_features.parquet")
        df.to_parquet(path, index=False)
        logger.info(f"Saved candidate features to {path}")
        
    def save_trust_scores(self, scores_list: List[Dict[str, Any]]):
        if not scores_list:
            return
        df = pd.DataFrame(scores_list)
        path = os.path.join(self.output_dir, "trust_scores.parquet")
        df.to_parquet(path, index=False)
        logger.info(f"Saved trust scores to {path}")
        
    def save_template_stats(self, stats: Dict[str, Any]):
        if not stats:
            return
        # Saving clustering metadata and centroids
        df = pd.DataFrame([stats])
        path = os.path.join(self.output_dir, "template_stats.parquet")
        df.to_parquet(path, index=False)
        logger.info(f"Saved template stats to {path}")
