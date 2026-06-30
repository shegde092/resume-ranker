import os
# Force strict offline mode for Hugging Face hub / sentence transformers
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import json
import csv
import argparse
import logging
from typing import List, Dict, Any

from src.pipeline.ranking_pipeline import RankingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rank")

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranking Submission Script")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", type=str, required=True, help="Path to output submission.csv")
    parser.add_argument("--jd", type=str, required=True, help="Job description string or path to file")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Path to precomputed artifacts directory")
    args = parser.parse_args()

    jd = args.jd
    if os.path.exists(jd):
        with open(jd, 'r', encoding='utf-8') as f:
            jd = f.read().strip()
            
    logger.info("Initializing Ranking Pipeline...")
    pipeline = RankingPipeline()
    
    logger.info(f"Loading offline artifacts from {args.artifacts}...")
    if not os.path.exists(args.artifacts):
        logger.error(f"Artifacts directory not found: {args.artifacts}. Please run scripts/precompute.py first.")
        sys.exit(1)
        
    pipeline.load_artifacts(args.artifacts)
    
    # Defensive check: read candidates from file and verify they are present in cache
    unseen_candidates = []
    if os.path.exists(args.candidates):
        logger.info(f"Ingesting candidates file '{args.candidates}' to check for unseen profiles...")
        try:
            is_jsonl = True
            if args.candidates.endswith('.json'):
                try:
                    with open(args.candidates, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            is_jsonl = False
                            for cand in data:
                                cid = cand.get("candidate_id")
                                if cid and cid not in pipeline.candidate_cache:
                                    unseen_candidates.append(cand)
                except Exception:
                    pass
            if is_jsonl:
                with open(args.candidates, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            cand = json.loads(line)
                            cid = cand.get("candidate_id")
                            if cid and cid not in pipeline.candidate_cache:
                                unseen_candidates.append(cand)
        except Exception as e:
            logger.warning(f"Could not read candidates file for incremental verification: {e}")
            
    if unseen_candidates:
        logger.error(
            f"Error: Detected {len(unseen_candidates)} candidates not present in the precomputed cache. "
            "Offline precomputation is a mandatory step. Please run 'python scripts/precompute.py' first."
        )
        sys.exit(1)
        
    logger.info("Running online ranking (sectional multi-index)...")
    results = pipeline.run(raw_jd=jd, all_resumes=None, top_k=min(2000, len(pipeline.candidate_cache)))
    
    # Pre-round scores to 4 decimal places to ensure tie-breaks align with written values
    for cand in results:
        cand["rounded_score"] = round(float(cand.get("final_score", 0.0)), 4)
        
    # Sort by rounded_score (descending), break ties using candidate_id (ascending)
    results.sort(key=lambda x: (-x["rounded_score"], x.get("candidate_id", "")))
    top_100 = results[:100]
    
    logger.info(f"Exporting top {len(top_100)} candidates to {args.out}")
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank, cand in enumerate(top_100, start=1):
            candidate_id = cand.get("candidate_id", f"CAND_{rank:07d}")
            score = cand.get("rounded_score", 0.0)
            reasoning = cand.get("reasoning", cand.get("reason", ""))
            writer.writerow([candidate_id, rank, score, reasoning])
            
    logger.info("Done.")

if __name__ == "__main__":
    main()
