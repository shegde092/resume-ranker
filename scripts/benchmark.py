import os
import sys
import time
import logging
import random
import tracemalloc
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline.ranking_pipeline import RankingPipeline

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark")

SKILL_POOLS = {
    "ml": ["python", "pytorch", "tensorflow", "machine learning", "nlp", "llms", "pandas", "scikit-learn"],
    "backend": ["java", "go", "python", "node.js", "kubernetes", "docker", "postgres", "redis", "aws"],
}
ROLES = ["ML Engineer", "Backend Engineer"]

def generate_synthetic_dataset(num_candidates: int) -> List[Dict[str, Any]]:
    candidates = []
    for i in range(num_candidates):
        role_type = random.choice(list(SKILL_POOLS.keys()))
        job_title = random.choice(ROLES)
        pool = SKILL_POOLS[role_type]
        skills = random.sample(pool, k=min(len(pool), random.randint(3, 6)))
        
        cand = {
            "candidate_id": f"cand_{i}",
            "profile": {"headline": f"Senior {job_title}"},
            "skills": skills,
            "career_history": [{"title": job_title, "description": f"Worked as {job_title} utilizing {', '.join(skills)}."}],
            "years_of_experience": random.randint(1, 15)
        }
        candidates.append(cand)
    return candidates

def run_benchmark():
    print("Initializing Pipeline...")
    pipeline = RankingPipeline()
    jd = "Looking for a Python machine learning engineer with NLP and PyTorch experience"
    
    counts = [100, 1000, 10000]
    
    print("\nStarting Benchmarks...\n")
    print(f"{'Count':<10} | {'Offline Time':<15} | {'Online Time':<15} | {'Peak Memory (MB)':<15}")
    print("-" * 65)
    
    for count in counts:
        dataset = generate_synthetic_dataset(count)
        
        # 1. Benchmark Offline (Indexing)
        start_offline = time.time()
        pipeline._build_indices(dataset)
        offline_time = time.time() - start_offline
        
        # Build cache manually since we bypassed run(all_resumes)
        pipeline.candidate_cache = {c.get("candidate_id"): c for c in dataset}
        
        tracemalloc.start()
        start_online = time.time()
        
        # 2. Benchmark Online (Retrieval + Reranking)
        results = pipeline.run(raw_jd=jd, all_resumes=None, top_k=2000)
        
        online_time = time.time() - start_online
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / (1024 * 1024)
        
        print(f"{count:<10} | {offline_time:<15.2f} | {online_time:<15.2f} | {peak_mb:<15.2f}")
        
        # Clear indices for next run
        pipeline = RankingPipeline()
        
if __name__ == "__main__":
    run_benchmark()
