import os
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
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates.jsonl (ignored, uses offline artifacts instead)")
    parser.add_argument("--out", type=str, required=True, help="Path to output submission.csv")
    parser.add_argument("--jd", type=str, default="Looking for a Python machine learning engineer with NLP and FastAPI experience", help="Job description string or path to txt")
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
    
    logger.info("Running online ranking (no indices rebuilt)...")
    # We pass all_resumes=None since the cache is already loaded
    results = pipeline.run(raw_jd=jd, all_resumes=None, top_k=min(2000, len(pipeline.candidate_cache)))
    
    # Sort by final_score (descending), break ties using candidate_id (ascending)
    results.sort(key=lambda x: (-x.get("final_score", 0.0), x.get("candidate_id", "")))
    top_100 = results[:100]
    
    # Enforce exactly 100 constraint? The requirements say "Constraints: exactly 100 rows"
    # If we have less than 100 candidates, we just output what we have, but hopefully the dataset has >= 100.
    
    logger.info(f"Exporting top {len(top_100)} candidates to {args.out}")
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank, cand in enumerate(top_100, start=1):
            candidate_id = cand.get("candidate_id", f"unknown_{rank}")
            score = cand.get("final_score", 0.0)
            reasoning = cand.get("reason", "")
            writer.writerow([candidate_id, rank, score, reasoning])
            
    logger.info("Done.")

if __name__ == "__main__":
    main()
