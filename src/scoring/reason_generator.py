import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ReasonGenerator:
    """
    Generates human-readable ranking explanations for each candidate
    based on JD requirements, trust score, years of experience, and anomalies.
    """
    
    def generate(self, candidate: Dict[str, Any], parsed_jd: Dict[str, Any] = None, rank: int = None) -> Dict[str, Any]:
        """
        Takes a scored candidate dictionary, parsed JD, and rank, appending 'confidence' and 'reason'.
        """
        if parsed_jd is None:
            parsed_jd = {}
            
        final_score = candidate.get("final_score", 0.0)
        trust_score = candidate.get("trust_score", 1.0)
        
        # Determine confidence based on rank and score thresholds
        if rank is not None:
            if rank <= 20 and final_score >= 0.50:
                confidence = "high"
            elif rank <= 100 and final_score >= 0.30:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            if final_score >= 0.75:
                confidence = "high"
            elif final_score >= 0.50:
                confidence = "medium"
            else:
                confidence = "low"
            
        reasons = []
        
        jd_skills = set(str(s).lower() for s in parsed_jd.get("required_skills", []))
        cand_skills = set(str(s).lower() for s in candidate.get("skills", []))
        
        matched_skills = list(jd_skills.intersection(cand_skills))[:3]
        
        yoe = candidate.get("years_of_experience")
        jd_yoe = parsed_jd.get("min_years_experience")
        
        if final_score >= 0.75:
            base_reason = "Strong match due to"
            if matched_skills:
                base_reason += f" {', '.join(matched_skills).title()} overlap"
            else:
                base_reason += " semantic skill overlap"
                
            if yoe is not None:
                base_reason += f" and {yoe} years relevant experience"
                
            reasons.append(base_reason)
            
        elif final_score >= 0.50:
            base_reason = "Good semantic fit"
            if matched_skills:
                base_reason += f" with some {', '.join(matched_skills).title()} overlap"
            
            missing_skills = list(jd_skills - cand_skills)[:2]
            if missing_skills:
                base_reason += f" but lacks required {', '.join(missing_skills).title()} experience"
                
            reasons.append(base_reason)
            
        else:
            base_reason = "Partial skill overlap"
            if yoe is not None and jd_yoe is not None and yoe < jd_yoe:
                base_reason += f" but weaker experience ({yoe} vs {jd_yoe} required)"
            elif trust_score < 0.7:
                base_reason += " but lower trust confidence"
            reasons.append(base_reason)
            
        if candidate.get("is_suspicious"):
            honeypot_reasons = candidate.get("honeypot_reasons", [])
            reason_str = ", ".join(honeypot_reasons) if honeypot_reasons else "suspicious profile anomalies"
            reasons.append(f"Major concern: {reason_str}")
            
        if not reasons:
            reasons.append("Profile matches baseline requirements")
            
        reason_str = "; ".join(reasons) + "."
        # Capitalize first letter
        if reason_str:
            reason_str = reason_str[0].upper() + reason_str[1:]
        
        # Format the required output schema
        return {
            "candidate_id": candidate.get("candidate_id", "UNKNOWN"),
            "final_score": float(final_score),
            "confidence": confidence,
            "reason": reason_str
        }
