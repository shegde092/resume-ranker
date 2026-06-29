import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BlockSelector:
    """
    Generates rich natural-language semantic chunks for Candidates.
    (Kept named BlockSelector for compatibility with RankingPipeline).
    """
    def __init__(self):
        self._block_cache = {}

    def clear_cache(self):
        """Clear internal chunk cache to prevent memory growth."""
        self._block_cache.clear()

    def get_query_terms(self, query: str) -> set:
        # Kept for compatibility if still called, but we don't strictly need lexical overlap filtering anymore
        return set()

    def select_blocks(self, parsed_resume: Dict[str, Any], query_terms: set = None) -> Dict[str, str]:
        """
        Generates 4 distinct semantic chunks from the candidate schema.
        """
        cand_id = parsed_resume.get("candidate_id")
        if cand_id and cand_id in self._block_cache:
            return self._block_cache[cand_id]
            
        redrob_signals = parsed_resume.get("redrob_signals", {})
        profile = parsed_resume.get("profile", {})
        
        # 1. Career Chunk
        career_sentences = []
        yoe = profile.get("years_of_experience", parsed_resume.get("years_of_experience"))
        if yoe is not None:
            career_sentences.append(f"Candidate has {yoe} years experience.")
            
        current_title = profile.get("current_title", parsed_resume.get("current_title"))
        current_company = profile.get("current_company", parsed_resume.get("current_company"))
        if current_title and current_company:
            career_sentences.append(f"Currently working as {current_title} at {current_company}.")
            
        career_history = parsed_resume.get("career_history", [])
        def _sort_key(exp):
            ed = str(exp.get("end_date", "")).strip().lower()
            if not ed or ed in ("present", "current"):
                return "9999-99-99"
            return ed
        career_history = sorted(career_history, key=_sort_key, reverse=True)
        for exp in career_history[:3]: # top 3 most recent
            title = exp.get("title", "professional")
            company = exp.get("company", "a company")
            months = exp.get("duration_months", exp.get("duration", 0))
            desc = exp.get("description", "")
            career_sentences.append(f"Worked as {title} at {company} for {months} months. {desc}")
            
        career_text = " ".join(career_sentences).strip()
        
        # 2. Skills Chunk
        skills_sentences = []
        skills = parsed_resume.get("skills", profile.get("skills", []))
        assessment_scores = redrob_signals.get("skill_assessment_scores", {})
        
        if isinstance(skills, list):
            for skill in skills[:15]:
                if isinstance(skill, dict):
                    name = skill.get("name", "")
                    prof = skill.get("proficiency", "experienced")
                    months = skill.get("duration_months", "")
                    score = assessment_scores.get(name)
                    
                    s = f"Candidate is {prof} in {name}"
                    if months:
                        s += f" with {months} months experience"
                    if score:
                        s += f" and assessment score {score}"
                    skills_sentences.append(s + ".")
                else:
                    skills_sentences.append(f"Skilled in {skill}.")
                    
        skills_text = " ".join(skills_sentences).strip()
        
        # 3. Profile Chunk
        profile_sentences = []
        loc = profile.get("location", parsed_resume.get("location"))
        if loc:
            profile_sentences.append(f"Candidate located in {loc}.")
            
        work_mode = redrob_signals.get("preferred_work_mode")
        if work_mode:
            profile_sentences.append(f"Prefers {work_mode} work.")
            
        if redrob_signals.get("willing_to_relocate"):
            profile_sentences.append("Willing to relocate.")
            
        summary = profile.get("summary", parsed_resume.get("summary", ""))
        if summary:
            profile_sentences.append(f"Summary: {summary}")
            
        profile_text = " ".join(profile_sentences).strip()
        
        # 4. Education Chunk
        edu_sentences = []
        education = parsed_resume.get("education", [])
        for edu in education[:3]:
            degree = edu.get("degree", "Degree")
            field = edu.get("field_of_study", "a field")
            inst = edu.get("institution", "institution")
            tier = edu.get("tier", "")
            grade = edu.get("grade", "")
            
            s = f"Candidate holds {degree} in {field} from {inst}"
            if tier:
                s += f" (tier {tier})"
            if grade:
                s += f" with grade {grade}"
            edu_sentences.append(s + ".")
            
        edu_text = " ".join(edu_sentences).strip()
        
        blocks = {
            "career_text": career_text,
            "skills_text": skills_text,
            "profile_text": profile_text,
            "education_text": edu_text
        }
        
        if cand_id:
            self._block_cache[cand_id] = blocks
            
        return blocks
