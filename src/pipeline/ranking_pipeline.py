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

logger = logging.getLogger(__name__)

class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k
        
    def fuse(self, dense_results: Dict[str, float], bm25_results: Dict[str, float], top_k: int = 2000) -> List[Dict[str, Any]]:
        rrf_scores = {}
        for results in [dense_results, bm25_results]:
            ranked = sorted(results.items(), key=lambda item: item[1], reverse=True)
            for rank, (cand_id, _) in enumerate(ranked, start=1):
                rrf_scores[cand_id] = rrf_scores.get(cand_id, 0.0) + 1.0 / (self.k + rank)
                
        fused_ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return [{"candidate_id": cand_id, "score": score} for cand_id, score in fused_ranked[:top_k]]


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
                
                new_documents.append(" ".join(text_parts))
                new_candidate_ids.append(cand_id)
                self._indexed_candidate_ids.add(cand_id)
                
            if new_documents:
                logger.info(f"Adding {len(new_documents)} new candidates to indices...")
                self.dense_retriever.add_candidates(new_documents, new_candidate_ids)
                self.bm25_retriever.add_candidates(new_documents, new_candidate_ids)

    def run(self, raw_jd: str, all_resumes: List[Dict[str, Any]], top_k: int = 2000) -> List[Dict[str, Any]]:
        logger.info("Starting candidate ranking pipeline...")
        
        parsed_jd = self.jd_parser.parse(raw_jd)
        query = parsed_jd.get("query", raw_jd)
        
        resume_dict = {}
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
            
        candidates_to_rerank = enriched_retrieved[:800]
            
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
            
        feature_matrix = self.feature_assembler.process_batch(reranked_candidates)
        
        final_ranked = self.adaptive_ranker.rank(reranked_candidates, feature_matrix)
        
        explained_results = []
        for cand in final_ranked:
            explanation = self.reason_generator.generate(cand)
            cand.update(explanation)
            explained_results.append(cand)
            
        return explained_results
