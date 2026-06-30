import re
from typing import Dict, List

class AntiPersonaParser:
    def __init__(self):
        self.keywords = {
            "consulting": ["consulting", "services", "outsourcing", "tcs", "infosys", "wipro", "accenture"],
            "research_only": ["academic", "lab", "postdoc", "publications", "research"],
            "framework_wrapper": ["langchain", "wrapper", "tutorial", "wrappers"],
            "title_chaser": ["job hopper", "hopper", "hopping", "stable tenure", "tenure"]
        }
        self.explicit_modifiers = ["no", "exclude", "not", "never", "avoid", "don't", "dont", "without"]
        self.negative_modifiers = ["prefer no", "minus", "less", "rather not", "discourage"]

    def _get_tokens(self, text: str) -> List[str]:
        clean_text = re.sub(r'[^a-z0-9\s-]', ' ', text.lower())
        return [t.strip() for t in clean_text.split() if t.strip()]

    def parse(self, jd_text: str) -> Dict[str, float]:
        results = {
            "consulting": 0.0,
            "research_only": 0.0,
            "framework_wrapper": 0.0,
            "title_chaser": 0.0
        }
        
        tokens = self._get_tokens(jd_text)
        num_tokens = len(tokens)
        window_size = 5
        
        for persona, keywords in self.keywords.items():
            max_severity = 0.0
            for keyword in keywords:
                kw_tokens = keyword.split()
                kw_len = len(kw_tokens)
                
                for idx in range(num_tokens - kw_len + 1):
                    if tokens[idx:idx+kw_len] == kw_tokens:
                        start_idx = max(0, idx - window_size)
                        end_idx = min(num_tokens, idx + kw_len + window_size)
                        
                        window_tokens = tokens[start_idx:end_idx]
                        window_text = " ".join(window_tokens)
                        
                        severity = 0.25
                        if any(mod in window_text for mod in self.explicit_modifiers):
                            severity = 1.0
                        elif any(mod in window_text for mod in self.negative_modifiers):
                            severity = 0.5
                            
                        max_severity = max(max_severity, severity)
            results[persona] = max_severity
        return results
