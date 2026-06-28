import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        # BGE Small is excellent for general CPU retrieval
        self.model = SentenceTransformer(model_name)
        # BGE Small output dimension is 384
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Using IndexFlatIP for Cosine Similarity (requires normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        self.candidate_ids = []
        
    def add_candidates(self, documents: List[str], candidate_ids: List[str]):
        """
        Embed and add candidates to the FAISS index.
        """
        if not documents:
            return
            
        logger.info(f"Encoding {len(documents)} candidates for dense retrieval...")
        embeddings = self.model.encode(documents, normalize_embeddings=True)
        
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.candidate_ids.extend(candidate_ids)
        
    def search(self, query: str, top_k: int = 2000) -> Dict[str, float]:
        """
        Search for top_k candidates given a query (JD).
        Returns a dict mapping candidate_id -> dense_retrieval_score
        """
        if self.index.ntotal == 0:
            logger.warning("Dense index is empty.")
            return {}
            
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Ensure we don't request more than what we have
        k = min(top_k, self.index.ntotal)
        
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        
        results = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                cand_id = self.candidate_ids[idx]
                results[cand_id] = float(dist)
                
        return results
