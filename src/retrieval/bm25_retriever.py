import logging
import numpy as np
from rank_bm25 import BM25L
from typing import List, Dict

logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self):
        self.bm25_model = None
        self.candidate_ids = []
        self.corpus = []
        
    def _tokenize(self, text: str) -> List[str]:
        # lowercase text, whitespace split, remove empty tokens
        return [t for t in text.lower().split() if t]
        
    def add_candidates(self, documents: List[str], candidate_ids: List[str]):
        """
        Build the BM25 index cumulatively.
        """
        if not documents:
            return
            
        tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.corpus.extend(tokenized_corpus)
        self.candidate_ids.extend(candidate_ids)
        
        self.bm25_model = BM25L(self.corpus)
        
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
