import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ReasonGenerator:
    """
    Generates human-readable ranking explanations for each candidate
    based on CE score, trust score, years of experience, and logistics score.
    """
    
    def generate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes a scored candidate dictionary and appends 'confidence' and 'reason',
        returning the exact required output schema.
        """
        final_score = candidate.get("final_score", 0.0)
        trust_score = candidate.get("trust_score", 1.0)
        ce_score = candidate.get("ce_score_avg", 0.0)
        
        # Determine confidence based on final fusion score thresholds
        if final_score >= 0.75:
            confidence = "high"
        elif final_score >= 0.50:
            confidence = "medium"
        else:
            confidence = "low"
            
        # Build explanation reason
        reasons = []
        
        # CE / Semantic match
        if ce_score >= 0.75:
            reasons.append("Strong semantic skill match")
        elif ce_score >= 0.5:
            reasons.append("Moderate semantic skill match")
        else:
            reasons.append("Weak semantic match")
            
        # Trust score
        if trust_score >= 0.8:
            reasons.append("high trust score")
        elif trust_score < 0.4:
            reasons.append("low trust score penalty applied")
            
        # Experience
        yoe = candidate.get("years_of_experience")
        if yoe is not None:
            reasons.append(f"{yoe} years of relevant experience")
        else:
            reasons.append("relevant experience")
            
        reason_str = ", ".join(reasons)
        
        # Format the required output schema
        return {
            "candidate_id": candidate.get("candidate_id", "UNKNOWN"),
            "final_score": float(final_score),
            "confidence": confidence,
            "reason": reason_str
        }
