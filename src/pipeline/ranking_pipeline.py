import logging
from typing import List, Dict, Any

from src.parsing.jd_parser import JDParser
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import ReciprocalRankFusion
from src.reranking.block_selector import BlockSelector
from src.reranking.cross_encoder import ONNXCrossEncoder
from src.trust.trust_engine import TrustEngine
from src.features.logistics_engine import LogisticsEngine
from src.fusion.adaptive_fusion_ranker import AdaptiveFusionRanker
from src.explanation.reason_generator import ReasonGenerator

logger = logging.getLogger(__name__)

class RankingPipeline:
    """
    Master orchestrator for the candidate ranking flow.
    Executes all stages in strict architectural sequence.
    """
    def __init__(self):
        self.jd_parser = JDParser()
        self.dense_retriever = DenseRetriever()
        self.bm25_retriever = BM25Retriever()
        self.fusion = ReciprocalRankFusion()
        self.block_selector = BlockSelector()
        self.cross_encoder = ONNXCrossEncoder()
        
        # Warmup ONNX models to prevent first-request latency spikes
        if hasattr(self.cross_encoder, "warmup"):
            self.cross_encoder.warmup()
            
        self.trust_engine = TrustEngine()
        self.logistics_engine = LogisticsEngine()
        self.adaptive_ranker = AdaptiveFusionRanker()
        self.reason_generator = ReasonGenerator()

    def _build_indices(self, candidates: List[Dict[str, Any]]):
        """
        Converts resumes into text documents and initializes FAISS and BM25 indices.
        """
        documents = []
        candidate_ids = []
        for cand in candidates:
            cand_id = cand.get("candidate_id", "UNKNOWN")
            
            text_parts = []
            if "profile" in cand:
                text_parts.append(str(cand["profile"].get("headline", "")))
            if "skills" in cand:
                text_parts.append(" ".join(cand["skills"]))
            for exp in cand.get("career_history", []):
                text_parts.append(str(exp.get("title", "")))
                text_parts.append(str(exp.get("description", "")))
            
            documents.append(" ".join(text_parts))
            candidate_ids.append(cand_id)
            
        self.dense_retriever.add_candidates(documents, candidate_ids)
        self.bm25_retriever.add_candidates(documents, candidate_ids)

    def run(self, raw_jd: str, all_resumes: List[Dict[str, Any]], top_k: int = 2000) -> List[Dict[str, Any]]:
        """
        Executes the complete ranking pipeline.
        Returns a sorted list of candidates with explanations.
        """
        logger.info("Starting candidate ranking pipeline...")
        
        # 1. Parse JD
        parsed_jd = self.jd_parser.parse(raw_jd)
        query = parsed_jd.get("query", raw_jd)
        
        # 2. Build Indices for Retrieval (Offline step simulated at runtime for orchestration)
        self._build_indices(all_resumes)
        
        # 3. Hybrid Retrieval (Using correct search signature)
        dense_results = self.dense_retriever.search(query, top_k=top_k)
        bm25_results = self.bm25_retriever.search(query, top_k=top_k)
        
        # 4. Reciprocal Rank Fusion
        retrieved_candidates = self.fusion.fuse(dense_results, bm25_results)
        
        if not retrieved_candidates:
            logger.warning("Retrieval returned empty candidates.")
            return []
            
        # Re-attach full candidate data to retrieved IDs
        resume_dict = {cand.get("candidate_id"): cand for cand in all_resumes}
        enriched_retrieved = []
        for cand in retrieved_candidates:
            full_cand = resume_dict.get(cand["candidate_id"], {}).copy()
            # Preserve RRF score
            full_cand["score"] = cand["score"]
            enriched_retrieved.append(full_cand)
            
        # 5. Block Selection
        for cand in enriched_retrieved:
            selected_blocks = self.block_selector.select_blocks(cand, query)
            cand.update(selected_blocks)
            
        # 6. Cross Encoder Reranking
        reranked_candidates = self.cross_encoder.rerank(query, enriched_retrieved)
        
        if not reranked_candidates:
            logger.warning("Reranking returned empty candidates.")
            return []
        
        # 7. Trust and Logistics Scoring
        for cand in reranked_candidates:
            trust_result = self.trust_engine.evaluate(cand)
            cand["trust_score"] = trust_result.get("trust_score", 1.0)
            cand["logistics_score"] = self.logistics_engine.score(cand, parsed_jd)
            
        # 8. Adaptive Fusion Ranking
        final_ranked = self.adaptive_ranker.rank(reranked_candidates)
        
        # 9. Explanation Generation
        explained_results = []
        for cand in final_ranked:
            explanation = self.reason_generator.generate(cand)
            cand.update(explanation)
            explained_results.append(cand)
            
        # Return final ranked list
        return explained_results
