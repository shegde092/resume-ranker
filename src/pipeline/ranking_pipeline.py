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
        
        # 4 distinct retrieval index partitions
        self.career_dense = DenseRetriever()
        self.career_bm25 = BM25Retriever()
        
        self.skills_dense = DenseRetriever()
        self.skills_bm25 = BM25Retriever()
        
        self.profile_dense = DenseRetriever()
        self.profile_bm25 = BM25Retriever()
        
        self.education_dense = DenseRetriever()
        self.education_bm25 = BM25Retriever()
        
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
        
        self.candidate_cache = {}
        self._index_lock = threading.Lock()
        self._indexed_candidate_ids = set()

    def reset_indices(self):
        """Clears all indices and candidate cache for a new dataset."""
        with self._index_lock:
            self.career_dense = DenseRetriever()
            self.career_bm25 = BM25Retriever()
            self.skills_dense = DenseRetriever()
            self.skills_bm25 = BM25Retriever()
            self.profile_dense = DenseRetriever()
            self.profile_bm25 = BM25Retriever()
            self.education_dense = DenseRetriever()
            self.education_bm25 = BM25Retriever()
            self._indexed_candidate_ids.clear()
            self.candidate_cache.clear()
            self.block_selector.clear_cache()

    def _build_indices(self, candidates: List[Dict[str, Any]]):
        """
        Incrementally builds 4 distinct candidate sections for FAISS and BM25 indices.
        """
        c_texts, s_texts, p_texts, e_texts = [], [], [], []
        new_candidate_ids = []
        
        with self._index_lock:
            for cand in candidates:
                cand_id = cand.get("candidate_id")
                if not cand_id:
                    continue
                    
                if cand_id in self._indexed_candidate_ids:
                    continue
                    
                blocks = self.block_selector.select_blocks(cand)
                
                c_texts.append(blocks["career_text"])
                s_texts.append(blocks["skills_text"])
                p_texts.append(blocks["profile_text"])
                e_texts.append(blocks["education_text"])
                
                new_candidate_ids.append(cand_id)
                self._indexed_candidate_ids.add(cand_id)
                
            if new_candidate_ids:
                logger.info(f"Adding {len(new_candidate_ids)} new candidates to 8 sectional indices...")
                self.career_dense.add_candidates(c_texts, new_candidate_ids)
                self.career_bm25.add_candidates(c_texts, new_candidate_ids)
                
                self.skills_dense.add_candidates(s_texts, new_candidate_ids)
                self.skills_bm25.add_candidates(s_texts, new_candidate_ids)
                
                self.profile_dense.add_candidates(p_texts, new_candidate_ids)
                self.profile_bm25.add_candidates(p_texts, new_candidate_ids)
                
                self.education_dense.add_candidates(e_texts, new_candidate_ids)
                self.education_bm25.add_candidates(e_texts, new_candidate_ids)

    def load_artifacts(self, artifacts_dir: str):
        """Deprecated in Sectional Multi-Index Architecture."""
        raise NotImplementedError("Offline artifacts are no longer supported. Indices are built dynamically.")

    def run(self, raw_jd: str, all_resumes: List[Dict[str, Any]] = None, top_k: int = 2000) -> List[Dict[str, Any]]:
        import time
        logger.info("Starting candidate ranking pipeline...")
        pipeline_start = time.time()
        
        # 1. Parsing
        t0 = time.time()
        parsed_jd = self.jd_parser.parse(raw_jd)
        parse_t = time.time() - t0
        
        # 2. Index Building
        t0 = time.time()
        
        if all_resumes is not None:
            # Clear all indices and cache on new dataset load
            self.reset_indices()
            for cand in all_resumes:
                cand_id = cand.get("candidate_id")
                if not cand_id:
                    cand_id = uuid.uuid4().hex
                
                local_cand = cand.copy()
                local_cand["candidate_id"] = cand_id
                self.candidate_cache[cand_id] = local_cand
                
            self._build_indices(list(self.candidate_cache.values()))
            
        allowed_ids = set(self.candidate_cache.keys())
        index_build_t = time.time() - t0
        
        # 3. Retrieval
        def _safe_search(dense_idx, bm25_idx, query_str):
            t0 = time.time()
            d_res = {k: v for k, v in dense_idx.search(query_str, top_k=top_k).items() if k in allowed_ids}
            b_res = {k: v for k, v in bm25_idx.search(query_str, top_k=top_k).items() if k in allowed_ids}
            return d_res, b_res, time.time() - t0
            
        t0 = time.time()
        retrieval_results = []
        
        career_q = parsed_jd.get("career_query", "")
        if career_q.strip():
            c_dense, c_bm25, c_t = _safe_search(self.career_dense, self.career_bm25, career_q)
            retrieval_results.extend([c_dense, c_bm25])
        else:
            c_t = 0.0
            
        skills_q = parsed_jd.get("skills_query", "")
        if skills_q.strip():
            s_dense, s_bm25, s_t = _safe_search(self.skills_dense, self.skills_bm25, skills_q)
            retrieval_results.extend([s_dense, s_bm25])
        else:
            s_t = 0.0
            
        profile_q = parsed_jd.get("profile_query", "")
        if profile_q.strip():
            p_dense, p_bm25, p_t = _safe_search(self.profile_dense, self.profile_bm25, profile_q)
            retrieval_results.extend([p_dense, p_bm25])
        else:
            p_t = 0.0
            
        education_q = parsed_jd.get("education_query", "")
        if education_q.strip():
            e_dense, e_bm25, e_t = _safe_search(self.education_dense, self.education_bm25, education_q)
            retrieval_results.extend([e_dense, e_bm25])
        else:
            e_t = 0.0
            
        retrieval_t = time.time() - t0
        logger.info(f"Retrieval Latencies -> Career: {c_t:.3f}s, Skills: {s_t:.3f}s, Profile: {p_t:.3f}s, Education: {e_t:.3f}s")
        
        # 4. Fusion
        t0 = time.time()
        if retrieval_results:
            retrieved_candidates = self.fusion.fuse(*retrieval_results, top_k=top_k)
        else:
            retrieved_candidates = []
        rrf_t = time.time() - t0
        
        if not retrieved_candidates:
            logger.warning("Retrieval returned empty candidates.")
            return []
            
        enriched_retrieved = []
        for cand in retrieved_candidates:
            cand_id = cand["candidate_id"]
            if cand_id not in self.candidate_cache:
                continue
                
            full_cand = self.candidate_cache[cand_id].copy()
            full_cand["score"] = cand["score"]
            enriched_retrieved.append(full_cand)
            
        # 5. Cross Encoder
        t0 = time.time()
        from src.config import settings
        max_ce_candidates = min(getattr(settings, 'MAX_CE_CANDIDATES', 300), 500)
        candidates_to_rerank = enriched_retrieved[:max_ce_candidates]
            
        t_chunk = time.time()
        for cand in candidates_to_rerank:
            selected_blocks = self.block_selector.select_blocks(cand)
            cand.update(selected_blocks)
        chunk_t = time.time() - t_chunk
            
        ce_results = self.cross_encoder.rerank(parsed_jd, candidates_to_rerank)
        ce_t = time.time() - t0
        
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
        
        # 6. Feature Assembly & Trust/Logistics
        t0 = time.time()
        for cand in reranked_candidates:
            trust_result = self.trust_engine.evaluate(cand)
            cand["trust_score"] = trust_result.get("trust_score", 1.0)
            
            logistics_result = self.logistics_engine.score(cand, parsed_jd)
            cand.update(logistics_result)
            
            honeypot_result = self.honeypot_detector.evaluate(cand)
            cand.update(honeypot_result)
            
        feature_matrix = self.feature_assembler.process_batch(reranked_candidates, parsed_jd)
        feature_t = time.time() - t0
        
        # 7. Final Ranking
        t0 = time.time()
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
        rank_t = time.time() - t0
        
        total_latency = time.time() - pipeline_start
        
        # Pipeline Telemetry
        logger.info(f"Pipeline Telemetry:")
        logger.info(f" - Parse time:      {parse_t:.3f}s")
        logger.info(f" - Index Build:     {index_build_t:.3f}s")
        logger.info(f" - Retrieval:       {retrieval_t:.3f}s")
        logger.info(f" - RRF Fusion:      {rrf_t:.3f}s")
        logger.info(f" - Chunk Gen:       {chunk_t:.3f}s")
        logger.info(f" - Cross Encoder:   {ce_t:.3f}s")
        logger.info(f" - Feat Assembly:   {feature_t:.3f}s")
        logger.info(f" - Final Rank:      {rank_t:.3f}s")
        logger.info(f" => Total Latency:  {total_latency:.3f}s")
            
        return explained_results
