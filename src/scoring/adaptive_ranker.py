import logging
import time
from typing import List, Dict, Any
import numpy as np

from src.config import settings
from src.scoring.feature_assembler import FeatureAssembler

logger = logging.getLogger(__name__)

class AdaptiveFusionRanker:
    """
    Computes final candidate score using weighted fusion of:
    - Cross Encoder average
    - Retrieval RRF
    - Trust score
    - Logistics score
    
    Applies a smooth trust penalty to final scores for low trust.
    """
    def __init__(self):
        self.feature_assembler = FeatureAssembler()
        
        # Load weights from config
        w_ce = getattr(settings, 'WEIGHT_CE', 0.40)
        w_rrf = getattr(settings, 'WEIGHT_RRF', 0.30)
        w_trust = getattr(settings, 'WEIGHT_TRUST', 0.20)
        w_logistics = getattr(settings, 'WEIGHT_LOGISTICS', 0.10)
        
        # 1. Weight Normalization
        total_weight = w_ce + w_rrf + w_trust + w_logistics
        if total_weight <= 0:
            raise ValueError("Sum of fusion weights must be strictly greater than 0.")
            
        self.w_ce = w_ce / total_weight
        self.w_rrf = w_rrf / total_weight
        self.w_trust = w_trust / total_weight
        self.w_logistics = w_logistics / total_weight

    def rank(self, candidates: List[Dict[str, Any]], feature_matrix: np.ndarray = None) -> List[Dict[str, Any]]:
        """
        Executes the adaptive fusion ranking strategy.
        """
        if not candidates:
            return []
            
        start_time = time.time()
        
        # Extract features dynamically or use provided matrix
        if feature_matrix is not None:
            X = feature_matrix
        else:
            X = self.feature_assembler.process_batch(candidates)
        
        # 2. Empty Matrix Guard
        if X.size == 0:
            logger.warning("Feature matrix is empty. Returning empty candidate list.")
            return []
        
        # Use dynamic index lookups instead of hardcoded X[:, N]
        idx_rrf = self.feature_assembler.feature_index.get("retrieval_rrf", 0)
        idx_ce = self.feature_assembler.feature_index.get("ce_score_avg", 5)
        idx_trust = self.feature_assembler.feature_index.get("trust_score", 6)
        idx_logistics = self.feature_assembler.feature_index.get("logistics_score", 7)
        
        rrf_scores = X[:, idx_rrf]
        ce_scores = X[:, idx_ce]
        trust_scores = X[:, idx_trust]
        logistics_scores = X[:, idx_logistics]
        
        # RRF is now securely normalized in FeatureAssembler prior to X matrix construction.
        final_scores = (
            (self.w_ce * ce_scores) +
            (self.w_rrf * rrf_scores) +
            (self.w_trust * trust_scores) +
            (self.w_logistics * logistics_scores)
        )
        
        scored_candidates = []
        for i, cand in enumerate(candidates):
            cand_out = cand.copy()
            f_score = float(final_scores[i])
            t_score = float(trust_scores[i])
            
            if i < 3:
                print(f"=== AdaptiveRanker Debug: Candidate {cand.get('candidate_id', 'unknown')} ===")
                print(f"CE: {ce_scores[i]:.4f} * {self.w_ce:.2f} = {ce_scores[i]*self.w_ce:.4f}")
                print(f"RRF: {rrf_scores[i]:.4f} * {self.w_rrf:.2f} = {rrf_scores[i]*self.w_rrf:.4f}")
                print(f"Trust: {t_score:.4f} * {self.w_trust:.2f} = {t_score*self.w_trust:.4f}")
                print(f"Logistics: {logistics_scores[i]:.4f} * {self.w_logistics:.2f} = {logistics_scores[i]*self.w_logistics:.4f}")
                print(f"Raw Final: {f_score:.4f}")
            
            # 3. Smooth Trust Penalty
            if t_score < 0.4:
                penalty_multiplier = max(0.7, 1.0 - ((0.4 - t_score) * 0.75))
                f_score *= penalty_multiplier
                
            # 4. Clamp final score
            f_score = max(0.0, min(1.0, f_score))
                
            cand_out["final_score"] = f_score
            scored_candidates.append(cand_out)
            
        # 5. Deterministic Tie-Breaking
        # Sort descending by final score, then ascending by candidate_id
        scored_candidates.sort(key=lambda x: (-float(x.get("final_score", 0.0)), str(x.get("candidate_id", ""))))
        
        total_duration = time.time() - start_time
        logger.info(f"Adaptive fusion took {total_duration:.3f}s for {len(candidates)} candidates")
        
        return scored_candidates
