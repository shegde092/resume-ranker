import json
import logging
import pdfplumber
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

class ResumeParser:
    def __init__(self):
        pass

    def parse_json(self, raw_json: Union[str, dict]) -> Dict[str, Any]:
        """Parses the structured JSON resume format."""
        try:
            data = raw_json if isinstance(raw_json, dict) else json.loads(raw_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON resume: {e}")
            return {}

        return {
            "candidate_id": data.get("candidate_id", ""),
            "profile": data.get("profile", {}),
            "career_history": data.get("career_history", []),
            "education": data.get("education", []),
            "skills": data.get("skills", []),
            "certifications": data.get("certifications", []),
            "languages": data.get("languages", []),
            "redrob_signals": data.get("redrob_signals", {})
        }
        
    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF using pdfplumber.
        """
        raw_text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text + "\n"
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}")
            return {}
            
        # Structure it mapping extracted text back to the expected schema
        # In a true deployment, the raw text goes through an NER / LLM extraction layer
        return {
            "candidate_id": f"extracted_from_{file_path}",
            "raw_extracted_text": raw_text,
            "profile": {"summary": raw_text[:300]}, 
            "career_history": [],
            "education": [],
            "skills": [],
            "certifications": [],
            "languages": [],
            "redrob_signals": {}
        }
        
    def parse(self, source: Union[str, dict], format_type: str = "json") -> Dict[str, Any]:
        if format_type == "json":
            return self.parse_json(source)
        elif format_type in ["pdf", "doc"]:
            return self.parse_pdf(source)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
