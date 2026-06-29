import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        # k is a smoothing constant, 60 is standard for RRF
        self.k = k
        
    def fuse(self, *results_dicts: Dict[str, float], top_k: int = 2000) -> List[Dict[str, Any]]:
        """
        Applies Reciprocal Rank Fusion to any number of result dictionaries.
        Returns a sorted list of dicts: [{'candidate_id': ..., 'score': ...}, ...]
        """
        rrf_scores = {}
        
        for res_dict in results_dicts:
            if not res_dict:
                continue
                
            # Sort by score descending
            ranked = sorted(res_dict.items(), key=lambda item: item[1], reverse=True)
            for rank, (cand_id, _) in enumerate(ranked, start=1):
                if cand_id not in rrf_scores:
                    rrf_scores[cand_id] = 0.0
                rrf_scores[cand_id] += 1.0 / (self.k + rank)
                
        # Sort combined results by RRF score descending
        fused_ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Take Top K
        final_list = []
        for cand_id, score in fused_ranked[:top_k]:
            final_list.append({
                "candidate_id": cand_id,
                "score": score
            })
            
        return final_list
