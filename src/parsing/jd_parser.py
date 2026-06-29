import re
import spacy
from spacy.matcher import PhraseMatcher
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class JDParser:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            
            # Expanded skills including multi-word phrases
            skill_phrases = [
                "machine learning", "vector database", "deep learning", "python",
                "java", "c++", "go", "sql", "spark", "kafka", "aws", "gcp", "azure",
                "docker", "kubernetes", "llm", "llms", "embeddings", "faiss",
                "pinecone", "pytorch", "tensorflow", "nlp", "rag", "qdrant", "milvus", "weaviate"
            ]
            patterns = [self.nlp.make_doc(text) for text in skill_phrases]
            self.matcher.add("TECH_SKILLS", patterns)
            
        except OSError:
            logger.warning("Spacy model not found. Run 'python -m spacy download en_core_web_sm' to enable NER.")
            self.nlp = None

    def parse(self, jd_text: str) -> Dict[str, Any]:
        return self.parse_text(jd_text)

    def parse_text(self, jd_text: str) -> Dict[str, Any]:
        """
        Parse raw JD text into structured requirements.
        Uses PhraseMatcher to extract complex multi-word skills.
        """
        text_lower = jd_text.lower()
        
        # 1. Extract YOE (Years of Experience)
        yoe_pattern = r'(\d+)\s*(?:-|to|–)\s*(\d+)\s*years?|\b(\d+)\+?\s*years?'
        yoe_matches = re.findall(yoe_pattern, text_lower)
        min_yoe, max_yoe = self._resolve_yoe(yoe_matches)
        
        # 2. Extract Skills (Now using PhraseMatcher for multi-word capability)
        found_skills = self._extract_skills(text_lower)
        
        # 3. Extract Locations
        locations = self._extract_locations(text_lower)
        
        # 4. Sectional Heuristic Routing
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +|\n+', jd_text) if s.strip()]
        
        career_sentences = []
        skills_sentences = []
        profile_sentences = []
        edu_sentences = []
        
        # Keyword markers
        edu_markers = {"degree", "bachelor", "master", "phd", "university", "college", "graduat"}
        profile_markers = {"location", "remote", "hybrid", "onsite", "relocat", "culture", "visa", "timezone"}
        skill_markers = {"experience with", "proficient in", "knowledge of", "stack", "tools", "technologies", "familiarity"}
        career_markers = {"years", "experience", "role", "responsibilities", "manage", "lead", "develop", "build"}
        
        for sentence in sentences:
            s_lower = sentence.lower()
            matched = False
            
            if any(m in s_lower for m in edu_markers):
                edu_sentences.append(sentence)
                matched = True
                
            if any(m in s_lower for m in profile_markers) or any(l in s_lower for l in locations):
                profile_sentences.append(sentence)
                matched = True
                
            # If the sentence explicitly contains known extracted skills, it definitely belongs in skills
            has_explicit_skill = any(sk in s_lower for sk in found_skills)
            
            if has_explicit_skill or any(m in s_lower for m in skill_markers):
                skills_sentences.append(sentence)
                matched = True
                
            if any(m in s_lower for m in career_markers):
                career_sentences.append(sentence)
                matched = True
                
            # Fallback: If a sentence didn't match anything specific, duplicate into career and skills to avoid information loss
            if not matched:
                career_sentences.append(sentence)
                skills_sentences.append(sentence)
                
        # Extract Work Mode
        required_work_modes = []
        if "remote" in text_lower:
            required_work_modes.append("remote")
        if "hybrid" in text_lower:
            required_work_modes.append("hybrid")
        if "onsite" in text_lower or "on-site" in text_lower:
            required_work_modes.append("onsite")
            
        return {
            "raw_text": jd_text,
            "query": jd_text,
            "min_yoe": min_yoe,
            "max_yoe": max_yoe,
            "required_skills": found_skills,
            "preferred_skills": [],
            "locations": locations,
            "required_work_mode": required_work_modes,
            "career_query": " ".join(career_sentences),
            "skills_query": " ".join(skills_sentences),
            "profile_query": " ".join(profile_sentences),
            "education_query": " ".join(edu_sentences)
        }

    def _resolve_yoe(self, matches: List[Tuple[str, str, str]]) -> Tuple[int, int]:
        min_yoe, max_yoe = 0, 0
        for match in matches:
            if match[0] and match[1]: 
                min_yoe = max(min_yoe, int(match[0]))
                max_yoe = max(max_yoe, int(match[1]))
            elif match[2]: 
                min_yoe = max(min_yoe, int(match[2]))
                max_yoe = max(max_yoe, int(match[2]) + 5) 
        return min_yoe, max_yoe

    def _extract_skills(self, text: str) -> List[str]:
        if not self.nlp:
            return []
            
        doc = self.nlp(text)
        matches = self.matcher(doc)
        found = set()
        for match_id, start, end in matches:
            span = doc[start:end]
            found.add(span.text.lower())
        return list(found)
        
    def _extract_locations(self, text: str) -> List[str]:
        if not self.nlp:
            return []
            
        doc = self.nlp(text)
        locations = set()
        for ent in doc.ents:
            if ent.label_ == "GPE":
                locations.add(ent.text.lower())
        return list(locations)
