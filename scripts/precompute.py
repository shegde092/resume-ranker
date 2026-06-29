import os
import sys
import json
import pickle
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline.ranking_pipeline import RankingPipeline

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
    
    if args.candidates.endswith('.parquet'):
        import pandas as pd
        logger.info(f"Loading from parquet: {args.candidates}")
        df = pd.read_parquet(args.candidates)
        candidates = df.to_dict(orient="records")
    else:
        logger.info(f"Loading from jsonl: {args.candidates}")
        with open(args.candidates, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
                    
    logger.info(f"Loaded {len(candidates)} candidates.")
    
    pipeline = RankingPipeline()
    logger.info("Building indices (FAISS & BM25)...")
    pipeline._build_indices(candidates)
    
    logger.info("Saving dense retriever artifacts...")
    pipeline.dense_retriever.save(os.path.join(args.out_dir, "dense"))
    
    logger.info("Saving BM25 artifacts...")
    pipeline.bm25_retriever.save(os.path.join(args.out_dir, "bm25.pkl"))
    
    logger.info("Saving candidate cache...")
    candidate_cache = {c.get("candidate_id"): c for c in candidates if c.get("candidate_id")}
    with open(os.path.join(args.out_dir, "candidate_cache.pkl"), "wb") as f:
        pickle.dump(candidate_cache, f)
        
    logger.info("Precomputation complete. Artifacts saved.")

if __name__ == "__main__":
    main()
