import logging
import math
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

class FeatureAssembler:
    """
    Constructs the feature vector for each candidate prior to Adaptive Fusion ranking.
    Incorporates signals from:
    1. Retrieval (RRF score)
    2. Cross Encoder (Semantic section scores)
    3. Trust Engine (Plausibility, Template anomaly)
    4. Logistics (Location, Visa, Salary expectations)
    
    Expected Feature Ranges (for normalization/model):
    - retrieval_rrf: [0.0, 1.0] (normalized via min-max)
    - ce_* scores: [0.0, 1.0] (sigmoid)
    - trust_score: [0.0, 1.0]
    - logistics_score: [0.0, 1.0]
    - availability_score: [0.0, 1.0]
    - location_friction: [0.0, 1.0] (penalty)
    - salary_penalty: [0.0, 1.0]
    - visa_penalty: [0.0, 1.0]
    - years_of_experience_match: [0.0, 1.0]
    - skill_overlap_ratio: [0.0, 1.0]
    - certification_score: [0.0, 1.0]
    """
    
    def __init__(self):
        self.feature_names = [
            "retrieval_rrf",         
            "ce_career_fit",         
            "ce_skill_fit",          
            "ce_profile_fit",        
            "ce_education_fit",      
            "ce_score_avg",          
            "trust_score",           
            "logistics_score",       
            "availability_score",    
            "location_friction",     
            "salary_penalty",        
            "visa_penalty",          
            "years_of_experience_match", 
            "skill_overlap_ratio",       
            "certification_score",
            "profile_completeness_score",
            "recruiter_response_rate",
            "open_to_work_flag",
            "notice_period_days",
            "interview_completion_rate"
        ]
        
        self.feature_index = {
            name: idx for idx, name in enumerate(self.feature_names)
        }
        
        self.num_features = len(self.feature_names)

    def _safe_float(self, val: Any, default: float = 0.0) -> float:
        """Helper to safely parse floats and explicitly reject NaN and inf."""
        if val is None:
            return default
        try:
            parsed = float(val)
            if math.isnan(parsed) or math.isinf(parsed):
                return default
            return parsed
        except (ValueError, TypeError):
            return default
            
    def _clamp(self, val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamps bounded features strictly to [min_val, max_val] range."""
        return max(min_val, min(max_val, val))
            
    def _normalize_feature(self, value: float, method: str = "min-max", min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Optional feature normalization utility.
        """
        if method == "log1p":
            return math.log1p(max(0.0, value))
        elif method == "min-max":
            if max_val > min_val:
                normalized = (value - min_val) / (max_val - min_val)
                return self._clamp(normalized)
            return 0.0
        return value
        
    def extract_features(self, candidate: Dict[str, Any]) -> List[float]:
        """
        Extracts a deterministic float feature array from a candidate dictionary.
        """
        # 1. Retrieval Score (Raw RRF to be normalized across the batch)
        retrieval_rrf = self._safe_float(candidate.get("retrieval_rrf_score"), 0.0)
        
        # 2. Cross Encoder Scores (Clamped to 0-1)
        ce_career = self._clamp(self._safe_float(candidate.get("ce_career_score"), 0.0))
        ce_skill = self._clamp(self._safe_float(candidate.get("ce_skills_score"), 0.0))
        ce_profile = self._clamp(self._safe_float(candidate.get("ce_profile_score"), 0.0))
        ce_edu = self._clamp(self._safe_float(candidate.get("ce_education_score"), 0.0))
        
        ce_avg = self._clamp(self._safe_float(candidate.get("ce_score"), 0.0))
        
        # 3. Trust Score (Clamped)
        trust_score = self._clamp(self._safe_float(candidate.get("trust_score"), 0.5))
        
        # 4. Logistics Score & Sub-features (Clamped)
        logistics_score = self._clamp(self._safe_float(candidate.get("logistics_score"), 0.5))
        availability_score = self._clamp(self._safe_float(candidate.get("availability_score"), 0.5))
        location_friction = self._clamp(self._safe_float(candidate.get("location_friction"), 0.0))
        salary_penalty = self._clamp(self._safe_float(candidate.get("salary_penalty"), 0.0))
        visa_penalty = self._clamp(self._safe_float(candidate.get("visa_penalty"), 0.0))
        
        # 5. Domain Features (Clamped)
        years_of_experience_match = self._clamp(self._safe_float(candidate.get("years_of_experience_match"), 0.0))
        skill_overlap_ratio = self._clamp(self._safe_float(candidate.get("skill_overlap_ratio"), 0.0))
        certification_score = self._clamp(self._safe_float(candidate.get("certification_score"), 0.0))
        
        # 6. Recruiter & Profile Quality Signals (P1)
        redrob_signals = candidate.get("redrob_signals", {})
        profile_completeness_score = self._clamp(self._safe_float(candidate.get("profile_completeness_score", redrob_signals.get("profile_completeness_score")), 0.5))
        recruiter_response_rate = self._clamp(self._safe_float(redrob_signals.get("recruiter_response_rate"), 0.5))
        open_to_work_flag = 1.0 if candidate.get("open_to_work") or redrob_signals.get("open_to_work") else 0.0
        raw_notice = self._safe_float(candidate.get("notice_period_days", redrob_signals.get("notice_period_days")), 30.0)
        notice_period_days = self._clamp(raw_notice / 90.0)
        interview_completion_rate = self._clamp(self._safe_float(redrob_signals.get("interview_completion_rate"), 0.5))
        
        features = [
            retrieval_rrf,
            ce_career,
            ce_skill,
            ce_profile,
            ce_edu,
            ce_avg,
            trust_score,
            logistics_score,
            availability_score,
            location_friction,
            salary_penalty,
            visa_penalty,
            years_of_experience_match,
            skill_overlap_ratio,
            certification_score,
            profile_completeness_score,
            recruiter_response_rate,
            open_to_work_flag,
            notice_period_days,
            interview_completion_rate
        ]
        
        # 6. Feature Shape Validation
        if len(features) != self.num_features:
            raise ValueError(
                f"Feature vector shape mismatch! Expected {self.num_features} features, "
                f"but extracted {len(features)}."
            )
            
        return features

    def process_batch(self, candidates: List[Dict[str, Any]]) -> np.ndarray:
        """
        Processes a list of candidates and returns a numpy array of feature vectors.
        Shape: (num_candidates, num_features)
        """
        if not candidates:
            return np.empty((0, self.num_features), dtype=np.float32)
            
        feature_matrix = []
        for cand in candidates:
            feat_vec = self.extract_features(cand)
            feature_matrix.append(feat_vec)
            
        matrix = np.array(feature_matrix, dtype=np.float32)
        
        # Dynamically normalize retrieval_rrf across the batch
        rrf_idx = self.feature_index.get("retrieval_rrf", 0)
        rrf_scores = matrix[:, rrf_idx]
        
        rrf_min = np.min(rrf_scores)
        rrf_max = np.max(rrf_scores)
        
        if rrf_max > rrf_min:
            matrix[:, rrf_idx] = (rrf_scores - rrf_min) / (rrf_max - rrf_min)
        else:
            matrix[:, rrf_idx] = np.full_like(rrf_scores, 0.5)
            
        return matrix
