import os
import sys
import time
import json
import logging
import random
import tracemalloc
from typing import List, Dict, Any

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline.ranking_pipeline import RankingPipeline

# Set up logging for validation script (we want it relatively clean)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("validator")
logger.setLevel(logging.INFO)

# --- Synthetic Data Generation (since no real dataset found) ---

SKILL_POOLS = {
    "ml": ["python", "pytorch", "tensorflow", "machine learning", "nlp", "llms", "pandas", "scikit-learn"],
    "backend": ["java", "go", "python", "node.js", "kubernetes", "docker", "postgres", "redis", "aws"],
    "data_science": ["python", "r", "sql", "tableau", "statistics", "ab testing", "databricks"],
    "frontend": ["javascript", "typescript", "react", "vue", "css", "html", "figma"]
}

ROLES = ["ML Engineer", "Data Scientist", "Backend Engineer", "Frontend Developer", "DevOps Engineer"]

def generate_synthetic_dataset(num_candidates: int) -> List[Dict[str, Any]]:
    candidates = []
    for i in range(num_candidates):
        role_type = random.choice(list(SKILL_POOLS.keys()))
        job_title = random.choice(ROLES)
        
        # Pick 3-6 random skills from the pool
        pool = SKILL_POOLS[role_type]
        skills = random.sample(pool, k=min(len(pool), random.randint(3, 6)))
        
        cand = {
            "candidate_id": f"cand_{i}",
            "profile": {"headline": f"Senior {job_title}"},
            "skills": skills,
            "career_history": [
                {
                    "title": job_title,
                    "description": f"Worked as {job_title} utilizing {', '.join(skills)}."
                }
            ]
        }
        candidates.append(cand)
    return candidates


def run_test(pipeline: RankingPipeline, jd: str, candidates: List[Dict[str, Any]], title: str):
    logger.info(f"\n--- Testing JD: {title} ---")
    
    start_time = time.time()
    results = pipeline.run(raw_jd=jd, all_resumes=candidates, top_k=2000)
    latency = time.time() - start_time
    
    if not results:
        raise ValueError(f"Pipeline returned empty results for JD: {title}")
        
    print(f"JD: {title}")
    print(f"Latency: {latency:.2f}s")
    print("Top Candidates:")
    
    for i, res in enumerate(results[:10]):
        score = res.get("final_score")
        if score is None or score != score:  # checks for None or NaN
            raise ValueError(f"Candidate {res.get('candidate_id')} received NaN or missing score!")
        print(f"{i+1}. {res['candidate_id']} -> {score:.4f}")
        

def stress_test(pipeline: RankingPipeline):
    logger.info("\n=== Starting Batch Stress Test ===")
    counts = [10, 100, 1000]
    
    jd = "Looking for a seasoned Backend Engineer with Go, Kubernetes, and AWS."
    
    for count in counts:
        logger.info(f"Generating {count} synthetic candidates...")
        dataset = generate_synthetic_dataset(count)
        
        tracemalloc.start()
        start_time = time.time()
        
        try:
            results = pipeline.run(raw_jd=jd, all_resumes=dataset, top_k=2000)
        except Exception as e:
            logger.error(f"Failed on count {count}: {e}")
            raise
            
        latency = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 10**6
        
        print(f"\nCount: {count}")
        print(f"Latency: {latency:.2f}s")
        print(f"Peak Memory: {peak_mb:.2f} MB")
        print(f"Top Score: {results[0]['final_score']:.4f}" if results else "No results")

if __name__ == "__main__":
    logger.info("Initializing Pipeline (loading models)...")
    pipeline = RankingPipeline()
    
    # 1. Validation with specific JDs on a fixed pool of 100
    validation_dataset = generate_synthetic_dataset(100)
    
    jds = {
        "ML Engineer": "Looking for a Python machine learning engineer with NLP and PyTorch experience",
        "Data Scientist": "Need a Data Scientist strong in SQL, Python, statistics, and AB testing",
        "Backend Engineer": "Backend dev required with strong Go, Python, Postgres, and Docker skills"
    }
    
    for title, jd in jds.items():
        run_test(pipeline, jd, validation_dataset, title)
        
    # 2. Stress Testing
    stress_test(pipeline)
    
    print("\n[SUCCESS] All validation and stress tests passed successfully!")
