import logging
import re
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class TrustEngine:
    def __init__(self):
        pass

    def evaluate(self, parsed_resume: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate trust score using chronology, tenure stability, and inflation metrics.
        """
        career_history = parsed_resume.get("career_history", [])
        skills = parsed_resume.get("skills", [])
        profile = parsed_resume.get("profile", {})
        stated_yoe = parsed_resume.get("years_of_experience") or profile.get("years_of_experience")
        
        try:
            stated_yoe = float(stated_yoe) if stated_yoe is not None else 0.0
        except (ValueError, TypeError):
            stated_yoe = 0.0

        # 1. Chronology Anomalies (overlapping timelines)
        chronology_anomaly = self._check_chronology_anomaly(career_history)
        
        # 2. Title Chaser Score (Job Hopping Recency Weighted)
        title_chaser_score = self._check_title_chaser_score(career_history)
        
        # 3. Consulting Ratio
        consulting_ratio = self._check_consulting_ratio(career_history)
        
        # 4. Skill Inflation
        skill_inflation = self._check_skill_inflation(skills, stated_yoe)
        
        # 5. Experience Inflation
        experience_inflation = self._check_experience_inflation(stated_yoe, career_history)
        
        # Compute base trust score
        penalties = []
        base_trust = 1.0
        
        if chronology_anomaly > 0:
            penalties.append("chronology_anomaly")
            base_trust -= 0.3
            
        if title_chaser_score > 0.3:
            penalties.append("title_chaser")
            base_trust -= (title_chaser_score * 0.25)
            
        if consulting_ratio > 0.5:
            penalties.append("consulting_heavy")
            base_trust -= (consulting_ratio * 0.15)
            
        if skill_inflation > 2.0:
            penalties.append("skill_inflation")
            base_trust -= 0.2
            
        if experience_inflation > 0:
            penalties.append("experience_inflation")
            base_trust -= experience_inflation
            
        trust_score = float(max(0.0, min(1.0, base_trust)))
        
        return {
            "trust_score": trust_score,
            "chronology_anomaly": float(chronology_anomaly),
            "title_chaser_score": float(title_chaser_score),
            "consulting_ratio": float(consulting_ratio),
            "skill_inflation": float(skill_inflation),
            "experience_inflation": float(experience_inflation),
            "penalties": penalties
        }

    def _parse_date(self, date_str):
        if not date_str:
            return None
        if isinstance(date_str, (int, float)):
            return datetime(int(date_str), 1, 1)
        date_str = str(date_str).strip().lower()
        if date_str in ["present", "current", "now", "ongoing"]:
            return datetime.now()
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if match:
            return datetime(int(match.group(0)), 1, 1)
        return None

    def _check_chronology_anomaly(self, career_history: List[Dict[str, Any]]) -> float:
        intervals = []
        for exp in career_history:
            start_date = self._parse_date(exp.get("start_date") or exp.get("start_year"))
            end_date = self._parse_date(exp.get("end_date") or exp.get("end_year"))
            title = str(exp.get("title", "")).lower()
            if "intern" in title:
                continue
            if start_date and end_date and start_date < end_date:
                intervals.append((start_date, end_date))
        
        intervals.sort(key=lambda x: x[0])
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i+1][0]:
                overlap = (intervals[i][1] - intervals[i+1][0]).days / 30.0
                if overlap > 2.0:
                    return 1.0
        return 0.0

    def _check_title_chaser_score(self, career_history: List[Dict[str, Any]]) -> float:
        valid_roles = []
        for exp in career_history:
            title = str(exp.get("title", "")).lower()
            if "intern" in title:
                continue
            dur = exp.get("duration_months")
            if dur is None:
                start_date = self._parse_date(exp.get("start_date") or exp.get("start_year"))
                end_date = self._parse_date(exp.get("end_date") or exp.get("end_year"))
                if start_date and end_date:
                    dur = (end_date - start_date).days / 30.0
            if dur is not None and dur >= 3.0:
                valid_roles.append((self._parse_date(exp.get("start_date") or exp.get("start_year")) or datetime.min, dur))
                
        valid_roles.sort(key=lambda x: x[0], reverse=True)
        recent_roles = valid_roles[:4]
        
        if not recent_roles:
            return 0.0
            
        weights = [0.4, 0.3, 0.2, 0.1][:len(recent_roles)]
        weighted_sum = sum(role[1] * w for role, w in zip(recent_roles, weights))
        w_total = sum(weights)
        weighted_avg = weighted_sum / w_total if w_total > 0 else 0.0
        
        normalized = max(0.0, min(1.0, (36.0 - weighted_avg) / 24.0))
        penalty = normalized ** 1.5
        return penalty

    def _check_consulting_ratio(self, career_history: List[Dict[str, Any]]) -> float:
        consulting_keywords = {"consulting", "services", "outsourcing", "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tech mahindra", "hcl", "lti"}
        total_months = 0.0
        consulting_months = 0.0
        for exp in career_history:
            dur = exp.get("duration_months") or 0.0
            company = str(exp.get("company", "")).lower()
            desc = str(exp.get("description", "")).lower()
            
            is_consulting = any(k in company or k in desc for k in consulting_keywords)
            total_months += dur
            if is_consulting:
                consulting_months += dur
                
        return consulting_months / total_months if total_months > 0 else 0.0

    def _check_skill_inflation(self, skills: List[Dict[str, Any]], stated_yoe: float) -> float:
        expert_count = 0
        if not isinstance(skills, list):
            return 0.0
        for s in skills:
            if isinstance(s, dict):
                prof = str(s.get("proficiency", "")).lower()
                if prof in ["expert", "advanced", "lead", "senior"]:
                    expert_count += 1
        denom = max(1.0, stated_yoe)
        return expert_count / denom

    def _check_experience_inflation(self, stated_yoe: float, career_history: List[Dict[str, Any]]) -> float:
        if stated_yoe <= 0.0:
            return 0.0
        total_years = 0.0
        for exp in career_history:
            dur = exp.get("duration_months")
            if dur is not None:
                total_years += dur / 12.0
            else:
                start = self._parse_date(exp.get("start_date") or exp.get("start_year"))
                end = self._parse_date(exp.get("end_date") or exp.get("end_year"))
                if start and end:
                    total_years += (end - start).days / 365.25
        
        diff = stated_yoe - total_years
        if diff > 2.0:
            return min(0.4, (diff - 2.0) * 0.1)
        return 0.0
