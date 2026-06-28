import logging
import re
from typing import Dict, Any

from src.config import settings

logger = logging.getLogger(__name__)

class LogisticsEngine:
    """
    Computes a logistics score based on notice period, location match,
    and salary expectations with safe parsing and configurable penalties.
    """
    def __init__(self):
        # Configurable constants
        self.PENALTY_NOTICE_60 = getattr(settings, 'PENALTY_NOTICE_60', 0.2)
        self.PENALTY_NOTICE_30 = getattr(settings, 'PENALTY_NOTICE_30', 0.1)
        self.PENALTY_SALARY = getattr(settings, 'PENALTY_SALARY', 0.3)
        self.PENALTY_RELOCATION_YES = getattr(settings, 'PENALTY_RELOCATION_YES', 0.1)
        self.PENALTY_RELOCATION_NO = getattr(settings, 'PENALTY_RELOCATION_NO', 0.4)
        self.PENALTY_VISA = getattr(settings, 'PENALTY_VISA', 0.3)
        
    def _safe_parse_numeric(self, val: Any) -> float:
        """Safely extract the first numeric value from strings or numbers."""
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            # Remove commas and extract first numeric sequence
            matches = re.findall(r'\d+\.?\d*', val.replace(',', ''))
            if matches:
                return float(matches[0])
        return 0.0

    def score(self, candidate: Dict[str, Any], jd: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Returns structured logistics features.
        """
        jd = jd or {}
        
        availability_score = 1.0
        location_friction = 0.0
        salary_penalty = 0.0
        visa_penalty = 0.0
        
        # 1. Notice Period parsing
        notice_val = candidate.get("notice_period") or candidate.get("notice_period_days")
        if notice_val:
            if isinstance(notice_val, str) and notice_val.strip().lower() in ["immediate", "0", "na", "n/a"]:
                notice_days = 0.0
            else:
                notice_days = self._safe_parse_numeric(notice_val)
                
            if notice_days > 60:
                availability_score -= self.PENALTY_NOTICE_60
            elif notice_days > 30:
                availability_score -= self.PENALTY_NOTICE_30
                
        # 2. Salary Match
        cand_salary = self._safe_parse_numeric(candidate.get("salary_expectation"))
        jd_budget = self._safe_parse_numeric(jd.get("max_salary"))
        
        if cand_salary > 0 and jd_budget > 0:
            if cand_salary > jd_budget:
                salary_penalty = self.PENALTY_SALARY
                
        # 3. Location & Relocation Match
        cand_loc = str(candidate.get("location", "")).lower().strip()
        jd_loc = str(jd.get("location", "")).lower().strip()
        
        is_remote_jd = "remote" in jd_loc
        is_hybrid_jd = "hybrid" in jd_loc
        
        open_to_relocate = candidate.get("open_to_relocate", False)
        
        if not is_remote_jd and jd_loc and cand_loc:
            # Check for non-match
            if jd_loc not in cand_loc and cand_loc not in jd_loc:
                if open_to_relocate:
                    location_friction = self.PENALTY_RELOCATION_YES
                else:
                    location_friction = self.PENALTY_RELOCATION_NO
                    
        # 4. Visa Logic
        requires_visa = candidate.get("requires_visa", False)
        visa_sponsorship = jd.get("visa_sponsorship", False)
        
        if requires_visa and not visa_sponsorship:
            visa_penalty = self.PENALTY_VISA

        # 5. Final computation
        availability_score = max(0.0, availability_score)
        base_logistics = availability_score - location_friction - salary_penalty - visa_penalty
        logistics_score = max(0.0, min(1.0, base_logistics))
        
        return {
            "logistics_score": float(logistics_score),
            "availability_score": float(availability_score),
            "location_friction": float(location_friction),
            "salary_penalty": float(salary_penalty),
            "visa_penalty": float(visa_penalty)
        }
