import logging
import uuid
import copy
import threading
import os
import pickle
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from src.parsing.jd_parser import JDParser
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.reranking.cross_encoder import ONNXCrossEncoder
from src.scoring.trust_engine import TrustEngine
from src.scoring.logistics_engine import LogisticsEngine
from src.scoring.feature_assembler import FeatureAssembler
from src.scoring.adaptive_ranker import AdaptiveFusionRanker
from src.scoring.reason_generator import ReasonGenerator
from src.scoring.honeypot_detector import HoneypotDetector
from src.retrieval.fusion import ReciprocalRankFusion
from src.config import settings

logger = logging.getLogger(__name__)

class RankingPipeline:
    """
    Master orchestrator for the candidate ranking flow using a Sectional Multi-Index Architecture.
    """
    def __init__(self):
        self.jd_parser = JDParser()
        
        # Load shared embedding model
        model_name = getattr(settings, 'EMBEDDING_MODEL', "BAAI/bge-small-en-v1.5")
        logger.info(f"Initializing shared SentenceTransformer encoder: {model_name}...")
        self.shared_encoder = SentenceTransformer(model_name)
        
        # 4 Dense Retriever instances (generic code, distinct instances)
        self.career_dense = DenseRetriever(model_or_name=self.shared_encoder)
        self.skills_dense = DenseRetriever(model_or_name=self.shared_encoder)
        self.profile_dense = DenseRetriever(model_or_name=self.shared_encoder)
        self.education_dense = DenseRetriever(model_or_name=self.shared_encoder)
        
        # 4 BM25 Retriever instances
        self.career_bm25 = BM25Retriever()
        self.skills_bm25 = BM25Retriever()
        self.profile_bm25 = BM25Retriever()
        self.education_bm25 = BM25Retriever()
        
        self.fusion = ReciprocalRankFusion()
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
        self.trust_features = {}
        
        self._index_lock = threading.Lock()
        self._indexed_candidate_ids = set()

    def load_artifacts(self, artifacts_dir: str):
        """Loads precomputed FAISS, BM25 indices, candidate cache, and static trust features."""
        logger.info("Loading precomputed sectional offline artifacts...")
        
        # Load dense indexes
        self.career_dense.load(os.path.join(artifacts_dir, "career_dense"))
        self.skills_dense.load(os.path.join(artifacts_dir, "skills_dense"))
        self.profile_dense.load(os.path.join(artifacts_dir, "profile_dense"))
        self.education_dense.load(os.path.join(artifacts_dir, "education_dense"))
        
        # Load BM25 indexes
        self.career_bm25.load(os.path.join(artifacts_dir, "career_bm25.pkl"))
        self.skills_bm25.load(os.path.join(artifacts_dir, "skills_bm25.pkl"))
        self.profile_bm25.load(os.path.join(artifacts_dir, "profile_bm25.pkl"))
        self.education_bm25.load(os.path.join(artifacts_dir, "education_bm25.pkl"))
        
        # Load candidate cache
        with open(os.path.join(artifacts_dir, "candidate_cache.pkl"), "rb") as f:
            self.candidate_cache = pickle.load(f)
            
        # Load precomputed static trust features
        trust_pkl = os.path.join(artifacts_dir, "trust_features.pkl")
        if os.path.exists(trust_pkl):
            with open(trust_pkl, "rb") as f:
                self.trust_features = pickle.load(f)
        else:
            logger.warning("trust_features.pkl not found. Trust features will be evaluated dynamically.")
            self.trust_features = {}
            
        logger.info(f"Loaded {len(self.candidate_cache)} candidates into cache.")

    def run(self, raw_jd: str, all_resumes: List[Dict[str, Any]] = None, top_k: int = 250) -> List[Dict[str, Any]]:
        logger.info("Starting candidate ranking pipeline (Sectional Multi-Index)...")
        
        # Stage 1: JD Parser
        parsed_jd = self.jd_parser.parse(raw_jd)
        jd_chunks = parsed_jd.get("chunks", {})
        
        jd_career = str(jd_chunks.get("career", "")).strip()
        jd_skills = str(jd_chunks.get("skills", "")).strip()
        jd_profile = str(jd_chunks.get("profile", "")).strip()
        jd_education = str(jd_chunks.get("education", "")).strip()
        
        if all_resumes is not None:
            self._build_dynamic_indices(all_resumes)
            
        # Stage 2: Sectional Retrieval
        career_dense_res = {}
        if jd_career:
            career_q_emb = self.shared_encoder.encode([jd_career], normalize_embeddings=True)
            career_dense_res = self.career_dense.search(jd_career, top_k=top_k, precomputed_query_embedding=career_q_emb)
        career_bm25_res = self.career_bm25.search(jd_career, top_k=top_k) if jd_career else {}
        
        skills_dense_res = {}
        if jd_skills:
            skills_q_emb = self.shared_encoder.encode([jd_skills], normalize_embeddings=True)
            skills_dense_res = self.skills_dense.search(jd_skills, top_k=top_k, precomputed_query_embedding=skills_q_emb)
        skills_bm25_res = self.skills_bm25.search(jd_skills, top_k=top_k) if jd_skills else {}
        
        profile_dense_res = {}
        if jd_profile:
            profile_q_emb = self.shared_encoder.encode([jd_profile], normalize_embeddings=True)
            profile_dense_res = self.profile_dense.search(jd_profile, top_k=top_k, precomputed_query_embedding=profile_q_emb)
        profile_bm25_res = self.profile_bm25.search(jd_profile, top_k=top_k) if jd_profile else {}
        
        education_dense_res = {}
        if jd_education:
            education_q_emb = self.shared_encoder.encode([jd_education], normalize_embeddings=True)
            education_dense_res = self.education_dense.search(jd_education, top_k=top_k, precomputed_query_embedding=education_q_emb)
        education_bm25_res = self.education_bm25.search(jd_education, top_k=top_k) if jd_education else {}
        
        # Stage 3: RRF Fusion across all 8 retrieval streams
        rrf_scores = {}
        streams = [
            career_dense_res, career_bm25_res,
            skills_dense_res, skills_bm25_res,
            profile_dense_res, profile_bm25_res,
            education_dense_res, education_bm25_res
        ]
        
        for stream in streams:
            if not stream:
                continue
            ranked = sorted(stream.items(), key=lambda item: item[1], reverse=True)
            for rank, (cand_id, _) in enumerate(ranked, start=1):
                if cand_id not in rrf_scores:
                    rrf_scores[cand_id] = 0.0
                rrf_scores[cand_id] += 1.0 / (60 + rank)
                
        if not rrf_scores:
            logger.warning("All sectional retrieval streams returned empty candidates.")
            return []
            
        # Stage 4: Pruning to fixed TOP_K = 250 candidates
        fused_ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        prune_k = getattr(settings, 'TOP_K_PRUNE', 250)
        top_candidates = fused_ranked[:prune_k]
        
        candidates_to_rerank = []
        for cand_id, rrf_score in top_candidates:
            if cand_id not in self.candidate_cache:
                continue
            full_cand = self.candidate_cache[cand_id].copy()
            full_cand["retrieval_rrf_score"] = float(rrf_score)
            candidates_to_rerank.append(full_cand)
            
        # Stage 5: Sectional Cross Encoder
        ce_results = self.cross_encoder.rerank(parsed_jd, candidates_to_rerank)
        
        for cand in ce_results:
            cid = cand.get("candidate_id")
            
            static_trust = self.trust_features.get(cid, {})
            dynamic_trust = self.trust_engine.evaluate(cand)
            
            cand["chronology_anomaly"] = static_trust.get("chronology_anomaly", dynamic_trust.get("chronology_anomaly", 0.0))
            cand["title_chaser_score"] = static_trust.get("title_chaser_score", dynamic_trust.get("title_chaser_score", 0.0))
            cand["consulting_ratio"] = static_trust.get("consulting_ratio", dynamic_trust.get("consulting_ratio", 0.0))
            cand["skill_inflation"] = static_trust.get("skill_inflation", dynamic_trust.get("skill_inflation", 0.0))
            cand["experience_inflation"] = static_trust.get("experience_inflation", dynamic_trust.get("experience_inflation", 0.0))
            cand["trust_score"] = dynamic_trust.get("trust_score", static_trust.get("trust_score", 1.0))
            
            logistics_result = self.logistics_engine.score(cand, parsed_jd)
            cand.update(logistics_result)
            
            if "is_suspicious" not in cand:
                honeypot_result = self.honeypot_detector.evaluate(cand)
                cand.update(honeypot_result)
            
        # Stage 6: Feature Fusion
        feature_matrix = self.feature_assembler.process_batch(ce_results)
        final_ranked = self.adaptive_ranker.rank(ce_results, feature_matrix, parsed_jd=parsed_jd)
        
        # Stage 7: Reasoning Engine
        explained_results = []
        for rank_idx, cand in enumerate(final_ranked, start=1):
            explanation = self.reason_generator.generate(cand, parsed_jd, rank=rank_idx)
            cand.update(explanation)
            explained_results.append(cand)
            
        return explained_results

    def _build_dynamic_indices(self, candidates: List[Dict[str, Any]]):
        """Dynamic fallback indexing for online testing resumes passed via API in a batched manner."""
        from src.templates.template_builder import TemplateBuilder
        
        cids = []
        career_chunks = []
        skills_chunks = []
        profile_chunks = []
        education_chunks = []
        
        for cand in candidates:
            cid = cand.get("candidate_id") or uuid.uuid4().hex
            cand["candidate_id"] = cid
            chunks = TemplateBuilder.build_candidate_chunks(cand)
            cand_item = cand.copy()
            cand_item.update({
                "chunk_career": chunks["career"],
                "chunk_skills": chunks["skills"],
                "chunk_profile": chunks["profile"],
                "chunk_education": chunks["education"]
            })
            self.candidate_cache[cid] = cand_item
            
            cids.append(cid)
            career_chunks.append(chunks["career"])
            skills_chunks.append(chunks["skills"])
            profile_chunks.append(chunks["profile"])
            education_chunks.append(chunks["education"])
            
        if cids:
            self.career_dense.add_candidates(career_chunks, cids)
            self.career_bm25.add_candidates(career_chunks, cids)
            
            self.skills_dense.add_candidates(skills_chunks, cids)
            self.skills_bm25.add_candidates(skills_chunks, cids)
            
            self.profile_dense.add_candidates(profile_chunks, cids)
            self.profile_bm25.add_candidates(profile_chunks, cids)
            
            self.education_dense.add_candidates(education_chunks, cids)
            self.education_bm25.add_candidates(education_chunks, cids)
