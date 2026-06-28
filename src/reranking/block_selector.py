import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BlockSelector:
    def __init__(self, block_size: int = 200, overlap: int = 20, max_blocks: int = 3):
        self.block_size = block_size
        self.overlap = overlap
        self.max_blocks = max_blocks
        
    def _split_into_blocks(self, text: str) -> List[str]:
        words = text.split()
        blocks = []
        for i in range(0, len(words), self.block_size - self.overlap):
            block_words = words[i:i + self.block_size]
            if block_words:
                blocks.append(" ".join(block_words))
        return blocks
        
    def _score_block_lexical(self, block: str, query_terms: set) -> float:
        block_lower = block.lower()
        block_terms = set(re.sub(r'[^a-z0-9\s]', '', block_lower).split())
        
        score = 0.0
        for term in query_terms.intersection(block_terms):
            if not term:
                continue
            if len(term) > 3:
                score += 2.0
            else:
                score += 1.0
        return float(score)

    def select_blocks(self, parsed_resume: Dict[str, Any], query: str) -> Dict[str, str]:
        """
        Splits sections into blocks, selects top_k blocks using exact token overlap,
        and restores global context (headline, YOE, skills, certs) per architecture.
        """
        profile = parsed_resume.get("profile", {})
        headline = profile.get("headline", "")
        yoe = profile.get("years_of_experience", "")
        
        skills = parsed_resume.get("skills", [])
        if not skills:
            skills = profile.get("skills", [])
            
        certs = parsed_resume.get("certifications", [])
        if not certs:
            certs = profile.get("certifications", [])
            
        skill_strings = [str(s) for s in skills[:15]]
        cert_strings = [str(c) for c in certs[:5]]
            
        # Global context string precisely as required
        global_context = (
            f"Headline: {headline} | "
            f"YOE: {yoe} | "
            f"Skills: {', '.join(skill_strings)} | "
            f"Certifications: {', '.join(cert_strings)}"
        )
        
        # Tokenize query for overlap using regex normalization
        query_terms = set(re.sub(r'[^a-z0-9\s]', '', query.lower()).split())
        # Remove empty tokens
        query_terms = {t for t in query_terms if t}
        
        def _get_top_text(section_text: str) -> str:
            if not section_text:
                return global_context
                
            if not query_terms:
                return global_context + "\n\n" + section_text[:1000]
                
            blocks = self._split_into_blocks(section_text)
            scored_blocks = []
            for b in blocks:
                score = self._score_block_lexical(b, query_terms)
                scored_blocks.append((score, b))
            scored_blocks.sort(key=lambda x: x[0], reverse=True)
            top_blocks = [b for score, b in scored_blocks[:self.max_blocks]]
            return global_context + "\n\n" + "\n...\n".join(top_blocks)

        # Process each section independently for section-wise CE scoring
        career_history = parsed_resume.get("career_history", [])
        career_text = " ".join([str(c.get("description", "")) + " " + str(c.get("title", "")) for c in career_history])
        
        skills_text = ", ".join([str(s) for s in skills])
        profile_text = str(profile.get("summary", "")) + " " + str(headline)
        
        education = parsed_resume.get("education", [])
        edu_text = " ".join([str(e.get("degree", "")) + " " + str(e.get("field_of_study", "")) for e in education])
        
        return {
            "career_text": _get_top_text(career_text),
            "skills_text": _get_top_text(skills_text),
            "profile_text": _get_top_text(profile_text),
            "education_text": _get_top_text(edu_text)
        }
