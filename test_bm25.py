import sys
import logging
from src.pipeline.ranking_pipeline import RankingPipeline

logging.basicConfig(level=logging.INFO)

pipeline = RankingPipeline()

raw_jd = 'Looking for a Python machine learning engineer with NLP and FastAPI experience'

cand1 = {
    'candidate_id': 'c1',
    'profile': {'headline': 'ML Engineer'},
    'skills': ['python', 'machine learning', 'nlp', 'fastapi']
}
cand2 = {
    'candidate_id': 'c2',
    'profile': {'headline': 'Frontend Developer'},
    'skills': ['javascript', 'react', 'css']
}

print('\n--- RUNNING PIPELINE ---')
results = pipeline.run(raw_jd=raw_jd, all_resumes=[cand1, cand2], top_k=10)

print('\n--- RESULTS ---')
for r in results:
    print(f"{r['candidate_id']} -> Score: {r.get('final_score')}")
