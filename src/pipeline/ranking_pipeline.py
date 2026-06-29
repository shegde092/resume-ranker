import logging
import uuid
import copy
import threading
from typing import List, Dict, Any

from src.parsing.jd_parser import JDParser
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.reranking.block_selector import BlockSelector
from src.reranking.cross_encoder import ONNXCrossEncoder
from src.scoring.trust_engine import TrustEngine
from src.scoring.logistics_engine import LogisticsEngine
from src.scoring.adaptive_ranker import AdaptiveFusionRanker
from src.scoring.reason_generator import ReasonGenerator
from src.scoring.feature_assembler import FeatureAssembler
from src.scoring.honeypot_detector import HoneypotDetector

logger = logging.getLogger(__name__)

from src.retrieval.fusion import ReciprocalRankFusion
class RankingPipeline:
    """
    Master orchestrator for the candidate ranking flow.
    """
    def __init__(self):
        self.jd_parser = JDParser()
        self.dense_retriever = DenseRetriever()
        self.bm25_retriever = BM25Retriever()
        self.fusion = ReciprocalRankFusion()
        self.block_selector = BlockSelector()
        self.cross_encoder = ONNXCrossEncoder()
        
        if hasattr(self.cross_encoder, "warmup"):
            self.cross_encoder.warmup()
            
        self.trust_engine = TrustEngine()
        self.logistics_engine = LogisticsEngine()
        self.feature_assembler = FeatureAssembler()
        self.adaptive_ranker = AdaptiveFusionRanker()
        self.reason_generator = ReasonGenerator()
        self.honeypot_detector = HoneypotDetector()
        
        self._index_lock = threading.Lock()
        self._indexed_candidate_ids = set()

    def _build_indices(self, candidates: List[Dict[str, Any]]):
        """
        Incrementally builds or updates global FAISS and BM25 indices with new candidates.
        """
        new_documents = []
        new_candidate_ids = []
        
        with self._index_lock:
            for cand in candidates:
                cand_id = cand.get("candidate_id")
                if not cand_id:
                    continue  # Should be assigned prior
                    
                if cand_id in self._indexed_candidate_ids:
                    continue
                    
                text_parts = []
                if "profile" in cand:
                    text_parts.append(str(cand["profile"].get("headline", "")))
                if "skills" in cand:
                    text_parts.append(" ".join(cand["skills"]))
                for exp in cand.get("career_history", []):
                    text_parts.append(str(exp.get("title", "")))
                    text_parts.append(str(exp.get("description", "")))
                    
                if not text_parts:
                    # Fallback for flat schemas or missing keys
                    if "raw_text" in cand:
                        text_parts.append(str(cand["raw_text"]))
                    elif "resume_text" in cand:
                        text_parts.append(str(cand["resume_text"]))
                    else:
                        # Extract all possible string content as a final safety net
                        for v in cand.values():
                            if isinstance(v, str):
                                text_parts.append(v)
                            elif isinstance(v, list):
                                text_parts.append(" ".join(str(item) for item in v))
                
                new_documents.append(" ".join(text_parts))
                new_candidate_ids.append(cand_id)
                self._indexed_candidate_ids.add(cand_id)
                
            if new_documents:
                logger.info(f"Adding {len(new_documents)} new candidates to indices...")
                self.dense_retriever.add_candidates(new_documents, new_candidate_ids)
                self.bm25_retriever.add_candidates(new_documents, new_candidate_ids)

    def load_artifacts(self, artifacts_dir: str):
        """Loads precomputed FAISS, BM25 indices, and candidate cache."""
        logger.info("Loading offline artifacts...")
        import os, pickle
        self.dense_retriever.load(os.path.join(artifacts_dir, "dense"))
        self.bm25_retriever.load(os.path.join(artifacts_dir, "bm25.pkl"))
        with open(os.path.join(artifacts_dir, "candidate_cache.pkl"), "rb") as f:
            self.candidate_cache = pickle.load(f)
        logger.info(f"Loaded {len(self.candidate_cache)} candidates into cache.")

    def run(self, raw_jd: str, all_resumes: List[Dict[str, Any]] = None, top_k: int = 2000) -> List[Dict[str, Any]]:
        logger.info("Starting candidate ranking pipeline...")
        
        parsed_jd = self.jd_parser.parse(raw_jd)
        query = parsed_jd.get("query", raw_jd)
        
        resume_dict = getattr(self, "candidate_cache", {})
        
        if all_resumes is not None:
            for cand in all_resumes:
                cand_id = cand.get("candidate_id")
                if not cand_id:
                    cand_id = uuid.uuid4().hex
                
                local_cand = copy.deepcopy(cand)
                local_cand["candidate_id"] = cand_id
                resume_dict[cand_id] = local_cand
                
            self._build_indices(list(resume_dict.values()))
        
        dense_results = self.dense_retriever.search(query, top_k=top_k)
        bm25_results = self.bm25_retriever.search(query, top_k=top_k)
        
        allowed_ids = set(resume_dict.keys())
        
        dense_results = {
            cid: score for cid, score in dense_results.items()
            if cid in allowed_ids
        }
        
        bm25_results = {
            cid: score for cid, score in bm25_results.items()
            if cid in allowed_ids
        }
        
        retrieved_candidates = self.fusion.fuse(dense_results, bm25_results, top_k=top_k)
        
        if not retrieved_candidates:
            logger.warning("Retrieval returned empty candidates.")
            return []
            
        enriched_retrieved = []
        for cand in retrieved_candidates:
            cand_id = cand["candidate_id"]
            if cand_id not in resume_dict:
                continue
                
            full_cand = resume_dict[cand_id].copy()
            full_cand["score"] = cand["score"]
            enriched_retrieved.append(full_cand)
            
        # Task 3: Reduce CPU bottleneck by limiting cross-encoder pool to 100
        candidates_to_rerank = enriched_retrieved[:100]
            
        query_terms = self.block_selector.get_query_terms(query)
        for cand in candidates_to_rerank:
            selected_blocks = self.block_selector.select_blocks(cand, query_terms)
            cand.update(selected_blocks)
            
        ce_results = self.cross_encoder.rerank(query, candidates_to_rerank)
        
        if not ce_results:
            logger.warning("Reranking returned empty candidates.")
            return []
            
        ce_dict = {res.get("candidate_id"): res for res in ce_results}
        reranked_candidates = []
        for cand in candidates_to_rerank:
            cand_id = cand.get("candidate_id")
            if cand_id in ce_dict:
                cand.update(ce_dict[cand_id])
                reranked_candidates.append(cand)
        
        for cand in reranked_candidates:
            trust_result = self.trust_engine.evaluate(cand)
            cand["trust_score"] = trust_result.get("trust_score", 1.0)
            
            logistics_result = self.logistics_engine.score(cand, parsed_jd)
            cand.update(logistics_result)
            
            honeypot_result = self.honeypot_detector.evaluate(cand)
            cand.update(honeypot_result)
            
        feature_matrix = self.feature_assembler.process_batch(reranked_candidates)
        
        final_ranked = self.adaptive_ranker.rank(reranked_candidates, feature_matrix)
        
        # Apply honeypot penalty and re-sort
        for cand in final_ranked:
            honeypot_score = cand.get("honeypot_score", 1.0)
            cand["final_score"] = cand.get("final_score", 0.0) * honeypot_score
            
        final_ranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        
        explained_results = []
        for rank_idx, cand in enumerate(final_ranked, start=1):
            explanation = self.reason_generator.generate(cand, parsed_jd, rank=rank_idx)
            cand.update(explanation)
            explained_results.append(cand)
            
        return explained_results
