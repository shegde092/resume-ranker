import json
import re
from typing import Dict, Any, List, Tuple
from src.parsing.anti_persona_parser import AntiPersonaParser

class JDParser:
    def __init__(self, skills_master_path="configs/skills_master.json"):
        with open(skills_master_path, 'r', encoding='utf-8') as f:
            self.skills_master = json.load(f)
        self.anti_persona_parser = AntiPersonaParser()
        
        # Precompile regex patterns for each individual skill and its aliases
        self.skill_patterns = {}
        for canonical, aliases in self.skills_master.items():
            all_forms = [canonical.replace('_', ' ')] + aliases
            combined_pattern = "|".join(re.escape(form) for form in all_forms)
            self.skill_patterns[canonical] = re.compile(rf'\b({combined_pattern})\b', re.IGNORECASE)
            
        # Precompile one mega regex matching all skills/aliases to optimize chunk building
        all_forms = []
        for canonical, aliases in self.skills_master.items():
            all_forms.append(canonical.replace('_', ' '))
            for alias in aliases:
                all_forms.append(alias)
        all_forms = sorted(list(set(all_forms)), key=len, reverse=True)
        combined_all_pattern = "|".join(re.escape(form) for form in all_forms)
        self.all_skills_pattern = re.compile(rf'\b({combined_all_pattern})\b', re.IGNORECASE)
            
        # Expanded city hub list (Indian and major international hubs)
        self.cities = [
            "pune", "noida", "hyderabad", "mumbai", "delhi", "bengaluru", "bangalore", 
            "chennai", "gurugram", "gurgaon", "kolkata", "ahmedabad",
            "san francisco", "london", "berlin", "toronto", "new york", "singapore"
        ]
        
        # Expanded countries dictionary
        self.countries = {
            "india": "India",
            "united states": "United States",
            "us": "United States",
            "usa": "United States",
            "united kingdom": "United Kingdom",
            "uk": "United Kingdom",
            "germany": "Germany",
            "canada": "Canada",
            "singapore": "Singapore",
            "united arab emirates": "United Arab Emirates",
            "uae": "United Arab Emirates"
        }

    def parse(self, jd_text: str) -> Dict[str, Any]:
        # Extract metadata
        yoe_min, yoe_max = self._extract_yoe(jd_text)
        locations = self._extract_locations(jd_text)
        work_mode = self._extract_work_mode(jd_text)
        country = self._extract_country(jd_text)
        
        # Skill extraction (required vs preferred)
        required_skills, preferred_skills = self._extract_skills(jd_text)
        
        # Parse anti-personas
        anti_personas = self.anti_persona_parser.parse(jd_text)
        
        # Split into chunks
        chunks = self._build_chunks(jd_text)
        
        return {
            "query": jd_text,
            "yoe_min": yoe_min,
            "yoe_max": yoe_max,
            "locations": locations,
            "work_mode": work_mode,
            "country": country,
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "anti_personas": anti_personas,
            "chunks": chunks
        }

    def _extract_skills(self, text: str) -> Tuple[List[str], List[str]]:
        required_triggers = ["need", "must", "required", "essential", "absolute", "criteria", "expected", "necessary", "minimum", "should have"]
        preferred_triggers = ["like", "preferred", "nice to have", "plus", "desirable", "optional", "bonus", "helpful", "good to have", "advantage"]
        boundary_triggers = ["responsibilities", "about company", "culture", "benefits", "what we offer", "role overview"]
        
        sentences = re.split(r'[.\n•\-]+', text)
        
        required_skills = set()
        preferred_skills = set()
        
        current_section_required = True
        
        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                # Reset state on empty section lines
                current_section_required = True
                continue
            
            sent_lower = sent_clean.lower()
            
            # Reset section state on explicit boundary triggers
            if any(b in sent_lower for b in boundary_triggers):
                current_section_required = True
            
            has_req = any(trig in sent_lower for trig in required_triggers)
            has_pref = any(trig in sent_lower for trig in preferred_triggers)
            
            # Heading change detection (explicit triggers shift section state)
            if has_pref and not has_req:
                current_section_required = False
            elif has_req and not has_pref:
                current_section_required = True
                
            for canonical, pattern in self.skill_patterns.items():
                if pattern.search(sent_lower):
                    if any(trig in sent_lower for trig in preferred_triggers) and not any(trig in sent_lower for trig in required_triggers):
                        preferred_skills.add(canonical)
                    elif any(trig in sent_lower for trig in required_triggers):
                        required_skills.add(canonical)
                    else:
                        if current_section_required:
                            required_skills.add(canonical)
                        else:
                            preferred_skills.add(canonical)
                            
        preferred_skills = preferred_skills - required_skills
        return list(required_skills), list(preferred_skills)

    def _extract_country(self, text: str) -> str:
        text_lower = text.lower()
        for k, v in self.countries.items():
            pattern = rf'\b{re.escape(k)}\b'
            if re.search(pattern, text_lower):
                return v
        return None

    def _extract_yoe(self, text: str) -> Tuple[int, int]:
        text_lower = text.lower()
        
        # 1. Range match: e.g. "3-5 years", "3 to 5 years", "3 to 5 yrs exp"
        range_match = re.search(r'(?:min(?:imum)?|at least)?\s*(\d+)\s*(?:to|-)\s*(\d+)\s*(?:years?|yrs?)(?:\s*exp(?:erience)?)?', text_lower)
        if range_match:
            return int(range_match.group(1)), int(range_match.group(2))
            
        # 2. Plus/Min match: e.g. "5+ years", "5+ yrs", "5+ years exp"
        plus_match = re.search(r'(?:min(?:imum)?|at least|experience:\s*)?\s*(\d+)\+\s*(?:years?|yrs?)(?:\s*exp(?:erience)?)?', text_lower)
        if plus_match:
            return int(plus_match.group(1)), None
            
        # 3. Explicit prefix match: e.g. "minimum 5 years", "min 5 years", "at least 5 years", "experience: 5 years"
        prefixes = r'(?:min(?:imum)?|at least|experience:\s*)\s*'
        prefix_match = re.search(rf'{prefixes}(\d+)\s*(?:years?|yrs?)(?:\s*exp(?:erience)?)?', text_lower)
        if prefix_match:
            return int(prefix_match.group(1)), None
            
        # 4. Standard fallback match: e.g. "5 years", "5 yrs exp"
        fallback_match = re.search(r'\b(\d+)\s*(?:years?|yrs?)(?:\s*exp(?:erience)?)?', text_lower)
        if fallback_match:
            return int(fallback_match.group(1)), None
            
        return None, None

    def _extract_locations(self, text: str) -> List[str]:
        locations = []
        text_lower = text.lower()
        for city in self.cities:
            pattern = rf'\b{re.escape(city)}\b'
            if re.search(pattern, text_lower):
                locations.append(city.title())
        if "remote" in text_lower:
            locations.append("Remote")
        return locations

    def _extract_work_mode(self, text: str) -> str:
        text_lower = text.lower()
        if "remote" in text_lower:
            return "remote"
        if "hybrid" in text_lower:
            return "hybrid"
        if "onsite" in text_lower or "on-site" in text_lower:
            return "onsite"
        return None

    def _build_chunks(self, text: str) -> Dict[str, str]:
        sentences = re.split(r'[.\n•\-]+', text)
        
        chunks = {
            "career": [],
            "skills": [],
            "profile": [],
            "education": []
        }
        
        career_keywords = {'experience', 'worked', 'built', 'developed', 'led', 'managed', 'responsible', 'years', 'production', 'system', 'architect', 'engineering', 'roles', 'history'}
        profile_keywords = {'looking for', 'seeking', 'candidate', 'team', 'culture', 'passion', 'driven', 'role', 'impact', 'notice period', 'location', 'salary', 'remote', 'hybrid', 'visa', 'relocation', 'founding'}
        education_keywords = {'degree', 'bachelor', 'master', 'phd', 'university', 'graduated', 'academic', 'school', 'education', 'computer science'}
        
        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue
                
            sent_lower = sent_clean.lower()
            matched_categories = []
            
            if any(k in sent_lower for k in career_keywords):
                matched_categories.append("career")
            if any(k in sent_lower for k in ['skills', 'proficient', 'knowledge', 'technologies', 'stack', 'using', 'tools', 'languages', 'frameworks']) or self.all_skills_pattern.search(sent_lower):
                matched_categories.append("skills")
            if any(k in sent_lower for k in profile_keywords):
                matched_categories.append("profile")
            if any(k in sent_lower for k in education_keywords):
                matched_categories.append("education")
                
            if matched_categories:
                for cat in matched_categories:
                    chunks[cat].append(sent_clean)
            else:
                is_technical = bool(self.all_skills_pattern.search(sent_lower))
                if is_technical:
                    chunks["skills"].append(sent_clean)
                else:
                    chunks["career"].append(sent_clean)
                    
        return {
            "career": " ".join(chunks["career"]),
            "skills": " ".join(chunks["skills"]),
            "profile": " ".join(chunks["profile"]),
            "education": " ".join(chunks["education"])
        }
