import os
import sys
import time
import csv
import psutil
import argparse
import random
import logging
from typing import List, Dict, Any

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline.ranking_pipeline import RankingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

SKILL_POOLS = {
    "ml": ["python", "pytorch", "tensorflow", "machine learning", "nlp", "llms", "pandas", "scikit-learn", "vector search", "hybrid search", "xgboost", "keras", "opencv", "nltk", "spacy"],
    "backend": ["java", "go", "python", "node.js", "kubernetes", "docker", "postgres", "redis", "aws", "gcp", "azure", "graphql", "grpc", "spring boot", "django", "flask", "fastapi"],
}
ROLES = ["ML Engineer", "Backend Engineer"]

def generate_rich_synthetic_dataset(num_candidates: int) -> List[Dict[str, Any]]:
    """
    Generates synthetic candidates following the exact nested Redrob candidate JSON schema
    with 3-8 career entries, 20-60 skills, 200-500 words descriptions, and signals.
    """
    words = [
        "implemented", "designed", "architected", "optimized", "deployed", "scaled", "secured", "monitored",
        "kubernetes", "docker", "microservices", "pipelines", "databases", "analytics", "real-time", "streaming",
        "performance", "efficiency", "distributed", "system", "infrastructure", "platform", "cloud", "aws", "gcp",
        "engineering", "development", "integration", "migration", "automation", "CI/CD", "observability", "metrics",
        "collaborated", "managed", "led", "team", "stakeholders", "deliverables", "agile", "scrum", "architecture",
        "python", "spark", "kafka", "redis", "postgresql", "mongodb", "elasticsearch", "airflow", "terraform", "ansible"
    ]
    
    candidates = []
    for i in range(num_candidates):
        role_type = random.choice(list(SKILL_POOLS.keys()))
        job_title = random.choice(ROLES)
        pool = SKILL_POOLS[role_type]
        
        # 20 to 60 skills
        num_skills = random.randint(20, 60)
        skills_subset = random.sample(pool, k=min(len(pool), num_skills))
        if len(skills_subset) < num_skills:
            for s_idx in range(num_skills - len(skills_subset)):
                skills_subset.append(f"Skill_{s_idx}")
                
        skills_list = []
        for s in skills_subset:
            skills_list.append({
                "name": s,
                "proficiency": random.choice(["beginner", "intermediate", "advanced"]),
                "endorsements": random.randint(0, 50),
                "duration_months": random.randint(6, 60)
            })
            
        # 3 to 8 career history entries
        num_career = random.randint(3, 8)
        career_list = []
        for c_idx in range(num_career):
            is_current = (c_idx == 0)
            desc_words = [random.choice(words) for _ in range(random.randint(200, 500))]
            description = " ".join(desc_words)
            career_list.append({
                "company": f"Company {c_idx}",
                "title": f"Engineer Level {num_career - c_idx}",
                "start_date": f"20{20 - c_idx}-01-01",
                "end_date": None if is_current else f"20{20 - c_idx + 1}-01-01",
                "duration_months": random.randint(12, 36),
                "is_current": is_current,
                "industry": "IT Services",
                "company_size": "10001+",
                "description": description
            })
            
        # Education
        edu_list = [
            {
                "institution": "University of Tech",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2012,
                "end_year": 2016,
                "grade": "8.5 CGPA",
                "tier": "tier_3"
            }
        ]
        
        cand = {
            "candidate_id": f"CAND_{i:07d}",
            "profile": {
                "anonymized_name": f"Anonymized Candidate {i}",
                "headline": f"Experienced {job_title} | Cloud & Infrastructure",
                "summary": "Experienced software professional with demonstrated history of working in industry.",
                "location": random.choice(["Mumbai", "Pune", "Noida", "Toronto", "Austin"]),
                "country": random.choice(["India", "Canada", "USA"]),
                "years_of_experience": float(random.randint(3, 15)),
                "current_title": job_title,
                "current_company": "Company 0",
                "current_company_size": "10001+",
                "current_industry": "IT Services"
            },
            "career_history": career_list,
            "education": edu_list,
            "skills": skills_list,
            "certifications": [
                {
                    "name": "Cloud Architect",
                    "issuer": "AWS",
                    "year": 2024
                }
            ],
            "languages": [
                {
                    "language": "English",
                    "proficiency": "professional"
                }
            ],
            "redrob_signals": {
                "profile_completeness_score": float(random.randint(70, 100)),
                "signup_date": "2023-01-01",
                "last_active_date": "2026-06-01",
                "open_to_work_flag": random.choice([True, False]),
                "profile_views_received_30d": random.randint(10, 200),
                "applications_submitted_30d": random.randint(1, 10),
                "recruiter_response_rate": random.uniform(0.3, 0.9),
                "avg_response_time_hours": random.uniform(5, 48),
                "skill_assessment_scores": {},
                "connection_count": random.randint(50, 800),
                "endorsements_received": random.randint(10, 100),
                "notice_period_days": random.choice([30, 60, 90]),
                "expected_salary_range_inr_lpa": {
                    "min": float(random.randint(10, 20)),
                    "max": float(random.randint(20, 50))
                },
                "preferred_work_mode": random.choice(["remote", "hybrid", "onsite"]),
                "willing_to_relocate": random.choice([True, False]),
                "github_activity_score": float(random.randint(10, 100)),
                "search_appearance_30d": random.randint(20, 300),
                "saved_by_recruiters_30d": random.randint(0, 15),
                "interview_completion_rate": random.uniform(0.5, 0.95),
                "offer_acceptance_rate": random.uniform(0.4, 0.9),
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True
            }
        }
        candidates.append(cand)
    return candidates

def get_peak_ram(process) -> float:
    try:
        import resource
        # ru_maxrss is peak memory in KB on Linux
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak_kb > 0:
            return peak_kb / 1024.0
    except (ImportError, AttributeError):
        pass
    
    # Fallback to Windows peak working set (peak_wset)
    mem_info = process.memory_info()
    peak_bytes = getattr(mem_info, "peak_wset", mem_info.rss)
    return peak_bytes / (1024 * 1024)

def run_synthetic_benchmark(count: int):
    print("\n" + "="*60)
    print(f"RUNNING NESTED SYNTHETIC SCALABILITY BENCHMARK (COUNT={count})")
    print("="*60)
    
    process = psutil.Process(os.getpid())
    
    print("Generating rich candidate dataset...")
    dataset = generate_rich_synthetic_dataset(count)
    
    pipeline = RankingPipeline()
    
    print(f"Building dynamic indices for {count} candidates...")
    start_offline = time.time()
    pipeline._build_dynamic_indices(dataset)
    offline_duration = time.time() - start_offline
    print(f"Offline indexing duration: {offline_duration:.2f} seconds.")
    
    print("Running online ranking...")
    jd = "Looking for a Python machine learning engineer with NLP and PyTorch experience in Mumbai."
    start_online = time.time()
    _ = pipeline.run(raw_jd=jd, all_resumes=None, top_k=250)
    online_duration = time.time() - start_online
    
    ram_mb = get_peak_ram(process)
    print(f"Online pipeline duration:  {online_duration:.4f} seconds")
    print(f"Peak Working Set RAM:       {ram_mb:.2f} MB")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Redrob Compliant Benchmark Runner")
    parser.add_argument("--mode", type=str, default="production", choices=["production", "synthetic"], help="Benchmark execution mode")
    parser.add_argument("--count", type=int, default=1000, help="Number of synthetic candidates to benchmark")
    parser.add_argument("--jd", type=str, default="Looking for a Python machine learning engineer with NLP and PyTorch experience in Mumbai.", help="Job description string or path to file")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Path to precomputed artifacts directory")
    parser.add_argument("--out", type=str, default="benchmark_output.csv", help="Path to output CSV")
    args = parser.parse_args()
    
    if args.mode == "synthetic":
        run_synthetic_benchmark(args.count)
        return
        
    jd = args.jd
    if os.path.exists(jd):
        with open(jd, 'r', encoding='utf-8') as f:
            jd = f.read().strip()
            
    print("="*60)
    print("REDROB SUBMISSION BENCHMARK RUNNER (PRODUCTION MODE)")
    print("="*60)
    
    process = psutil.Process(os.getpid())
    ram_initial = process.memory_info().rss / (1024 * 1024)
    print(f"Initial Process RAM:       {ram_initial:.2f} MB")
    
    print("Initializing ranking pipeline...")
    start_init = time.time()
    pipeline = RankingPipeline()
    init_duration = time.time() - start_init
    print(f"Pipeline Init Latency:     {init_duration:.4f} seconds")
    
    print(f"Loading precomputed artifacts from '{args.artifacts}'...")
    start_load = time.time()
    if not os.path.exists(args.artifacts):
        print(f"Error: Artifacts directory not found: {args.artifacts}")
        sys.exit(1)
    pipeline.load_artifacts(args.artifacts)
    load_duration = time.time() - start_load
    
    ram_after_load = process.memory_info().rss / (1024 * 1024)
    print(f"Artifacts Load Latency:    {load_duration:.4f} seconds")
    print(f"RAM After Load:            {ram_after_load:.2f} MB")
    
    print("\nExecuting full online pipeline (JD parse -> retrieval -> RRF -> CE -> fusion)...")
    start_online = time.time()
    results = pipeline.run(raw_jd=jd, all_resumes=None, top_k=min(2000, len(pipeline.candidate_cache)))
    
    results.sort(key=lambda x: (-x.get("final_score", 0.0), x.get("candidate_id", "")))
    top_100 = results[:100]
    
    print(f"Writing top {len(top_100)} candidates to '{args.out}'...")
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, cand in enumerate(top_100, start=1):
            writer.writerow([
                cand.get("candidate_id", "UNKNOWN"),
                rank,
                cand.get("final_score", 0.0),
                cand.get("reason", "")
            ])
            
    online_duration = time.time() - start_online
    ram_peak = get_peak_ram(process)
    
    print("\n" + "="*60)
    print("PRODUCTION COMPLIANCE AUDIT RESULTS")
    print("="*60)
    print(f"Full Online Pipeline Latency:  {online_duration:.4f} seconds")
    print(f"Total Initial-to-Peak RAM:     {ram_peak:.2f} MB")
    print(f"Net RAM Peak Growth (RSS):     {ram_peak - ram_initial:.2f} MB")
    print(f"Cross-Encoder Rerank Pool:     Fixed at TOP_K_PRUNE=250 (regardless of dataset size)")
    print(f"Under 300s Wall-Clock Limit:   {'YES' if online_duration <= 300.0 else 'NO'}")
    print(f"Under 16GB Memory Limit:       {'YES' if (ram_peak / 1024.0) <= 16.0 else 'NO'}")
    print("="*60)
    
    if os.path.exists(args.out):
        os.remove(args.out)

if __name__ == "__main__":
    main()
