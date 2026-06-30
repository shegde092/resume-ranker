import logging
import hashlib
import re
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class HoneypotDetector:
    """
    Detects anomalous or suspicious profiles using structural heuristics.
    """
    def __init__(self):
        self.current_year = datetime.now().year
        self.seen_profiles = set()

    def _get_level(self, title: str) -> int:
        title = str(title).lower()
        if "intern" in title or "trainee" in title: return 1
        if "junior" in title or "jr" in title: return 2
        if "senior" in title or "sr" in title or "lead" in title: return 4
        if "manager" in title or "director" in title or "head" in title: return 5
        if "vp" in title or "vice president" in title or "chief" in title or "cto" in title or "ceo" in title: return 6
        return 3 # Mid level

    def _parse_year(self, val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)
        val_str = str(val).strip()
        if val_str.lower() in ["present", "current", "now", "ongoing"]:
            return self.current_year
        match = re.search(r'\b(19|20)\d{2}\b', val_str)
        if match:
            return int(match.group(0))
        return None

    def evaluate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        reasons = []
        is_suspicious = False
        honeypot_score = 1.0

        # 1. Duplicate Profile Check
        stable_id = candidate.get("email") or candidate.get("phone") or candidate.get("linkedin_url") or candidate.get("github_url")
        if stable_id:
            profile_fingerprint = str(stable_id).strip().lower()
        else:
            skills_list = []
            for s in candidate.get("skills", []):
                if isinstance(s, dict):
                    skills_list.append(s.get("name") or s.get("skill_name") or "")
                else:
                    skills_list.append(str(s))
            raw_text = str(candidate.get("raw_text", "")) + "".join(skills_list)
            raw_text = raw_text.strip().lower()
            profile_fingerprint = hashlib.md5(raw_text.encode("utf-8")).hexdigest()
            
        if profile_fingerprint in self.seen_profiles and len(profile_fingerprint) > 5:
            reasons.append("Duplicate candidate profile detected")
            is_suspicious = True
        self.seen_profiles.add(profile_fingerprint)

        profile = candidate.get("profile", {})
        yoe = candidate.get("years_of_experience") or profile.get("years_of_experience", 0)
        if yoe is None:
            yoe = 0
            
        # 2. Skill Stuffing Ratio
        skills = candidate.get("skills", [])
        if isinstance(skills, list):
            ratio = len(skills) / max(float(yoe), 1.0)
            if ratio > 15.0 and len(skills) > 20:
                reasons.append(f"Suspicious skill/YOE ratio ({len(skills)} skills for {yoe} YOE)")
                is_suspicious = True

        career_history = candidate.get("career_history", [])
        
        # 3. Timeline structural checks
        if isinstance(career_history, list) and len(career_history) > 0:
            active_roles = 0
            prev_level = None
            prev_start = None
            
            # Map start/end years to integers
            mapped_history = []
            for exp in career_history:
                start_yr = self._parse_year(exp.get("start_year") or exp.get("start_date"))
                end_yr = self._parse_year(exp.get("end_year") or exp.get("end_date"))
                mapped_history.append({
                    "start_year": start_yr,
                    "end_year": end_yr,
                    "title": exp.get("title", "")
                })
            
            # Sort by start year ascending to check progression
            sorted_history = sorted([h for h in mapped_history if h.get("start_year") is not None], key=lambda x: x["start_year"])
                
            for exp in sorted_history:
                start_year = exp.get("start_year")
                end_year = exp.get("end_year")
                title = exp.get("title", "")
                
                # Check overlapping active roles
                if not end_year or end_year >= self.current_year:
                    active_roles += 1
                    
                if start_year is not None:
                    if end_year is not None:
                        if start_year > end_year:
                            reasons.append("Inconsistent dates: start > end")
                            is_suspicious = True
                        if (end_year - start_year) > 40:
                            reasons.append("Impossible single-role duration")
                            is_suspicious = True
                    
                    # 4. Impossible title progression (e.g., Intern to VP in < 2 years)
                    current_level = self._get_level(title)
                    if prev_level is not None and prev_start is not None:
                        if current_level >= 6 and prev_level <= 1:
                            if (start_year - prev_start) <= 2:
                                reasons.append(f"Suspicious promotion speed: Intern to Exec in <= 2 years")
                                is_suspicious = True
                                
                    prev_level = current_level
                    prev_start = start_year
                    
            if active_roles >= 3:
                reasons.append(f"Too many overlapping full-time roles ({active_roles})")
                is_suspicious = True
        
        if is_suspicious:
            honeypot_score = 0.1
            
        return {
            "honeypot_score": float(honeypot_score),
            "is_suspicious": bool(is_suspicious),
            "honeypot_reasons": reasons
        }
