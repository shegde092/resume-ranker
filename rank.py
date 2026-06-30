import os
import sys
import json
import csv
import argparse
import logging
import math
from typing import List, Dict, Any

from src.pipeline.ranking_pipeline import RankingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rank")



def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranking Submission Script")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidate dataset (.json or .jsonl)")
    parser.add_argument("--out", type=str, required=True, help="Path to output submission.csv")
    parser.add_argument("--jd", type=str, required=True, help="Job description string or path to txt/doc/json")
    args = parser.parse_args()

    # Load JD
    jd = args.jd
    if os.path.exists(jd):
        with open(jd, 'r', encoding='utf-8') as f:
            jd = f.read().strip()
            
    # Load Candidates
    logger.info(f"Loading candidates from {args.candidates}...")
    candidates = []
    if args.candidates.endswith('.json'):
        with open(args.candidates, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
            if isinstance(candidates, dict):
                candidates = [candidates]
    else:
        with open(args.candidates, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
                    
    logger.info(f"Loaded {len(candidates)} candidates.")

    logger.info("Initializing Ranking Pipeline...")
    pipeline = RankingPipeline()
    
    logger.info("Running ranking (building indices dynamically)...")
    results = pipeline.run(raw_jd=jd, all_resumes=candidates, top_k=min(2000, len(candidates)))
    
    # Sort by final_score (descending), break ties using candidate_id (ascending)
    results.sort(key=lambda x: (-x.get("final_score", 0.0), x.get("candidate_id", "")))
    top_100 = results[:100]
    
    def calculate_fit(score):
        score = max(0.0, min(1.0, float(score)))
        fit = 100.0 / (1.0 + math.exp(-8.0 * (score - 0.5)))
        return round(fit, 2)
    
    logger.info(f"Exporting top {len(top_100)} candidates to {args.out}")
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "fit_percentage", "reasoning"])
        
        for rank, cand in enumerate(top_100, start=1):
            candidate_id = cand.get("candidate_id", f"unknown_{rank}")
            score = cand.get("final_score", 0.0)
            fit_percentage = calculate_fit(score)
            reasoning = cand.get("reasoning", cand.get("reason", ""))
            writer.writerow([
                candidate_id,
                rank,
                fit_percentage,
                reasoning
            ])
            
    logger.info("Done.")

if __name__ == "__main__":
    main()
