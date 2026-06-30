import os
import sys
import json
import pickle
import logging
from sentence_transformers import SentenceTransformer

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.templates.template_builder import TemplateBuilder
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.scoring.trust_engine import TrustEngine
from src.scoring.honeypot_detector import HoneypotDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("precompute")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out_dir", type=str, default="artifacts", help="Output directory for artifacts")
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    candidates = []
    logger.info(f"Loading candidates from: {args.candidates}")
    
    if args.candidates.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(args.candidates)
        candidates = df.to_dict(orient="records")
    else:
        with open(args.candidates, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
                
    logger.info(f"Loaded {len(candidates)} candidates.")
    
    candidate_cache = {}
    career_docs, skills_docs, profile_docs, education_docs = [], [], [], []
    candidate_ids = []
    
    trust_engine = TrustEngine()
    honeypot_detector = HoneypotDetector()
    trust_features = {}
    
    logger.info("Building chunks and trust features...")
    for cand in candidates:
        cid = cand.get("candidate_id")
        if not cid:
            continue
            
        chunks = TemplateBuilder.build_candidate_chunks(cand)
        
        cand_cache_item = cand.copy()
        cand_cache_item.update({
            "chunk_career": chunks["career"],
            "chunk_skills": chunks["skills"],
            "chunk_profile": chunks["profile"],
            "chunk_education": chunks["education"]
        })
        
        career_docs.append(chunks["career"])
        skills_docs.append(chunks["skills"])
        profile_docs.append(chunks["profile"])
        education_docs.append(chunks["education"])
        candidate_ids.append(cid)
        
        trust_result = trust_engine.evaluate(cand)
        honeypot_result = honeypot_detector.evaluate(cand)
        
        cand_cache_item.update(honeypot_result)
        candidate_cache[cid] = cand_cache_item
        
        trust_features[cid] = {
            "chronology_anomaly": 1.0 if trust_result.get("chronology_anomaly") or honeypot_result.get("is_suspicious") else 0.0,
            "title_chaser_score": trust_result.get("title_chaser_score", 0.0),
            "consulting_ratio": trust_result.get("consulting_ratio", 0.0),
            "skill_inflation": trust_result.get("skill_inflation", 0.0),
            "experience_inflation": trust_result.get("experience_inflation", 0.0),
            "trust_score": trust_result.get("trust_score", 1.0)
        }
        
    logger.info("Initializing shared embedding model...")
    shared_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    sections = ["career", "skills", "profile", "education"]
    docs_map = {
        "career": career_docs,
        "skills": skills_docs,
        "profile": profile_docs,
        "education": education_docs
    }
    
    for section in sections:
        logger.info(f"Building FAISS index for {section}...")
        dense = DenseRetriever(model_or_name=shared_model)
        dense.add_candidates(docs_map[section], candidate_ids)
        dense.save(os.path.join(args.out_dir, f"{section}_dense"))
        
        logger.info(f"Building BM25 index for {section}...")
        bm25 = BM25Retriever()
        bm25.add_candidates(docs_map[section], candidate_ids)
        bm25.save(os.path.join(args.out_dir, f"{section}_bm25.pkl"))
        
    logger.info("Saving candidate cache...")
    with open(os.path.join(args.out_dir, "candidate_cache.pkl"), "wb") as f:
        pickle.dump(candidate_cache, f)
        
    logger.info("Saving precomputed trust features...")
    with open(os.path.join(args.out_dir, "trust_features.pkl"), "wb") as f:
        pickle.dump(trust_features, f)
        
    logger.info("Precomputation finished successfully!")

if __name__ == "__main__":
    main()
