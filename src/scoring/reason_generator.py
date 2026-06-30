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
        Generates explanation mapping candidate facts to JD requirements.
        """
        if parsed_jd is None:
            parsed_jd = {}
            
        final_score = candidate.get("final_score", 0.0)
        trust_score = candidate.get("trust_score", 1.0)
        
        if final_score >= 0.65:
            confidence = "high"
        elif final_score >= 0.35:
            confidence = "medium"
        else:
            confidence = "low"
            
        reasons = []
        
        # Skill overlap
        jd_skills = parsed_jd.get("required_skills", [])
        cand_skills_str = str(candidate.get("skills", "")).lower()
        matched_skills = []
        for skill in jd_skills:
            if skill.lower().replace('_', ' ') in cand_skills_str:
                matched_skills.append(skill)
                
        # YoE check
        profile = candidate.get("profile", {})
        yoe = candidate.get("years_of_experience") or profile.get("years_of_experience")
        jd_yoe_min = parsed_jd.get("yoe_min")
        
        # Match Base
        if final_score >= 0.65:
            base = "Excellent profile match"
            if matched_skills:
                base += f" with key skills in {', '.join(matched_skills[:3]).replace('_', ' ').title()}"
            if yoe is not None:
                base += f" and {yoe} years of relevant experience"
            reasons.append(base)
        elif final_score >= 0.35:
            base = "Good match"
            if matched_skills:
                base += f" showing capability in {', '.join(matched_skills[:2]).replace('_', ' ').title()}"
            reasons.append(base)
        else:
            base = "Partial match"
            if yoe is not None and jd_yoe_min is not None and float(yoe) < float(jd_yoe_min):
                base += f" with lower experience ({yoe} years vs required {jd_yoe_min} years)"
            reasons.append(base)
            
        # Logistics & Availability
        redrob_signals = candidate.get("redrob_signals", {})
        notice = candidate.get("notice_period_days") or candidate.get("notice_period") or redrob_signals.get("notice_period_days")
        if notice is not None:
            try:
                n_days = int(notice)
                if n_days <= 15:
                    reasons.append("Highly available (notice <= 15 days)")
            except:
                pass
                
        location_mult = candidate.get("location_multiplier", 1.0)
        if location_mult >= 0.9:
            reasons.append("Strong location alignment")
            
        # Trust Warning
        if trust_score < 0.7:
            reasons.append("Tenure hopping or chronology overlaps detected")
            
        # Anti-Persona Warning
        anti_mult = candidate.get("anti_persona_penalty", 1.0)
        if anti_mult < 0.9:
            reasons.append("Indicators of consulting background or research focus")
            
        reason_str = "; ".join(reasons) + "."
        if reason_str:
            reason_str = reason_str[0].upper() + reason_str[1:]
            
        return {
            "candidate_id": candidate.get("candidate_id", "UNKNOWN"),
            "final_score": float(final_score),
            "confidence": confidence,
            "reason": reason_str
        }
