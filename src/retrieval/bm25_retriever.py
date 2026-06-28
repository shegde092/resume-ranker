import logging
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self):
        self.bm25_model = None
        self.candidate_ids = []
        
    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace tokenizer for BM25 lexical keyword retrieval
        return text.lower().split()
        
    def add_candidates(self, documents: List[str], candidate_ids: List[str]):
        """
        Build the BM25 sparse index.
        """
        if not documents:
            return
            
        tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.bm25_model = BM25Okapi(tokenized_corpus)
        self.candidate_ids = list(candidate_ids)
        
    def search(self, query: str, top_k: int = 2000) -> Dict[str, float]:
        """
        Search for top_k candidates given a query (JD).
        Returns a dict mapping candidate_id -> bm25_score
        """
        if self.bm25_model is None:
            logger.warning("BM25 index is empty.")
            return {}
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25_model.get_scores(tokenized_query)
        
        # Get top k indices
        k = min(top_k, len(self.candidate_ids))
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = {}
        for idx in top_indices:
            results[self.candidate_ids[idx]] = float(scores[idx])
            
        return results
