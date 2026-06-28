import logging
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SparseRetriever:
    def __init__(self, use_splade: bool = False):
        self.use_splade = use_splade
        if self.use_splade:
            # SPLADE model logic would go here.
            # Given CPU limitations and simplicity for the BM25 fallback path,
            # we default to the architecture's BM25 fallback logic when SPLADE isn't configured
            logger.info("SPLADE enabled, but falling back to BM25 in this implementation block.")
            self.use_splade = False
            
        self.bm25_model = None
        self.candidate_ids = []
        
    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace tokenizer for BM25 fallback
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
        Returns a dict mapping candidate_id -> score
        """
        if self.bm25_model is None:
            logger.warning("Sparse index is empty.")
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
