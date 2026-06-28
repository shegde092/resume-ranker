import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        # k is a smoothing constant, 60 is standard for RRF
        self.k = k
        
    def fuse(self, dense_results: Dict[str, float], bm25_results: Dict[str, float], top_k: int = 2000) -> List[Dict[str, Any]]:
        """
        Applies Reciprocal Rank Fusion to dense and BM25 results.
        Returns a sorted list of dicts: [{'candidate_id': ..., 'score': ...}, ...]
        """
        rrf_scores = {}
        
        # Rank dense results (sort by score descending)
        dense_ranked = sorted(dense_results.items(), key=lambda item: item[1], reverse=True)
        for rank, (cand_id, _) in enumerate(dense_ranked, start=1):
            if cand_id not in rrf_scores:
                rrf_scores[cand_id] = 0.0
            rrf_scores[cand_id] += 1.0 / (self.k + rank)
            
        # Rank BM25 results
        bm25_ranked = sorted(bm25_results.items(), key=lambda item: item[1], reverse=True)
        for rank, (cand_id, _) in enumerate(bm25_ranked, start=1):
            if cand_id not in rrf_scores:
                rrf_scores[cand_id] = 0.0
            rrf_scores[cand_id] += 1.0 / (self.k + rank)
            
        # Sort combined results by RRF score
        fused_ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Take Top K
        final_list = []
        for cand_id, score in fused_ranked[:top_k]:
            final_list.append({
                "candidate_id": cand_id,
                "score": score
            })
            
        return final_list
