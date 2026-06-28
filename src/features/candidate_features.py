from typing import Dict, Any

class CandidateFeatureExtractor:
    def __init__(self):
        pass
        
    def extract(self, parsed_resume: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts deterministic features for the fusion model.
        """
        profile = parsed_resume.get("profile", {})
        redrob = parsed_resume.get("redrob_signals", {})
        
        # Calculate skill counts
        skills = parsed_resume.get("skills", [])
        skill_count = len(skills)
        advanced_skill_count = sum(1 for s in skills if str(s.get("proficiency", "")).lower() in ["advanced", "expert"])
        
        # Certifications & Projects
        cert_count = len(parsed_resume.get("certifications", []))
        project_count = len(parsed_resume.get("projects", []))
        
        # YOE
        yoe = profile.get("years_of_experience", 0.0)
        
        # Redrob Signals
        notice_period_days = redrob.get("notice_period_days", 90)
        open_to_work = int(redrob.get("open_to_work_flag", False))
        response_rate = redrob.get("recruiter_response_rate", 0.0)
        
        # Career progression rate (heuristic)
        career_history = parsed_resume.get("career_history", [])
        num_roles = len(career_history)
        progression_rate = num_roles / yoe if yoe > 0 else 0.0
        
        return {
            "total_years_experience": yoe,
            "number_of_projects": project_count,
            "certification_count": cert_count,
            "skill_count": skill_count,
            "advanced_skill_count": advanced_skill_count,
            "notice_period_days": notice_period_days,
            "open_to_work_flag": open_to_work,
            "recruiter_response_rate": response_rate,
            "career_progression_rate": progression_rate,
            "num_roles": num_roles
        }

    def hard_business_rule_validator(self, parsed_resume: Dict[str, Any], jd_rules: Dict[str, Any]) -> bool:
        """
        Evaluates strict constraints. Returns True if candidate passes all rules, False otherwise.
        """
        profile = parsed_resume.get("profile", {})
        yoe = profile.get("years_of_experience", 0.0)
        min_yoe = jd_rules.get("min_yoe", 0)
        
        # 1. Minimum YOE check
        if yoe < min_yoe:
            return False
            
        # 2. Mandatory Certifications check
        required_certs = jd_rules.get("mandatory_certifications", [])
        if required_certs:
            candidate_certs = [str(c.get("name", "")).lower() for c in parsed_resume.get("certifications", [])]
            for req_cert in required_certs:
                if not any(req_cert.lower() in c for c in candidate_certs):
                    return False
                    
        # 3. Mandatory Visa check
        req_visa = jd_rules.get("mandatory_visa", False)
        if req_visa:
            # Assumes visa availability is a redrob signal if tracked
            candidate_visa = parsed_resume.get("redrob_signals", {}).get("has_visa", False)
            if not candidate_visa:
                return False
                
        return True
