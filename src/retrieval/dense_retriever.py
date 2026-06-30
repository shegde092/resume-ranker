import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Union

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(self, model_or_name: Union[str, SentenceTransformer] = "BAAI/bge-small-en-v1.5"):
        # CPU-optimized BGE Small model loader (allowing shared instance)
        if isinstance(model_or_name, SentenceTransformer):
            self.model = model_or_name
        else:
            self.model = SentenceTransformer(model_or_name)
            
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.candidate_ids = []
        
    def add_candidates(self, documents: List[str], candidate_ids: List[str], precomputed_embeddings: np.ndarray = None):
        """
        Embed and add candidates to the FAISS index.
        """
        if not documents:
            return
            
        if precomputed_embeddings is not None:
            embeddings = precomputed_embeddings
        else:
            logger.info(f"Encoding {len(documents)} candidates for dense retrieval...")
            embeddings = self.model.encode(documents, normalize_embeddings=True)
        
        self.candidate_ids.extend(candidate_ids)
        self.index.add(np.array(embeddings, dtype=np.float32))
        
    def search(self, query: str, top_k: int = 2000, precomputed_query_embedding: np.ndarray = None) -> Dict[str, float]:
        """
        Search for top_k candidates given a query (JD).
        Returns a dict mapping candidate_id -> dense_retrieval_score
        """
        if self.index.ntotal == 0:
            logger.warning("Dense index is empty.")
            return {}
            
        if precomputed_query_embedding is not None:
            query_embedding = precomputed_query_embedding
        else:
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        
        results = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                cand_id = self.candidate_ids[idx]
                results[cand_id] = float(dist)
                
        return results

    def save(self, path_prefix: str):
        """Save FAISS index and candidate mapping to disk."""
        faiss.write_index(self.index, f"{path_prefix}.faiss")
        import json
        with open(f"{path_prefix}_candidates.json", "w", encoding="utf-8") as f:
            json.dump(self.candidate_ids, f)

    def load(self, path_prefix: str):
        """Load FAISS index and candidate mapping from disk."""
        import json
        self.index = faiss.read_index(f"{path_prefix}.faiss")
        with open(f"{path_prefix}_candidates.json", "r", encoding="utf-8") as f:
            self.candidate_ids = json.load(f)
