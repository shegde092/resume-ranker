import json
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def generate_reasoning(row):
    years = round(row.get('years_of_experience', 5.0), 1)
    tier = row.get('final_tier', 'UNKNOWN')
    
    domain_map = {
        'A': 'AI search and ranking',
        'B': 'ML and NLP',
        'C': 'applied ML/computer vision',
        'D': 'backend and data engineering'
    }
    domain = domain_map.get(tier, 'software engineering')
    
    top_skills = row.get('top_skills', 'Python')
    if not top_skills:
        top_skills = "Python"
        
    company = row.get('current_company', '')
    title = row.get('current_title', '')
    
    # Construct strength sentence
    if title and company:
        strength = f"shipped features as a {title} at {company}"
    elif title:
        strength = f"gained strong experience as a {title}"
    else:
        strength = "worked on production-scale systems"
        
    # Extract first sentence of description if present and relevant
    first_job_desc = row.get('first_job_description', '')
    if first_job_desc:
        first_job_desc = first_job_desc.strip()
        # Find first period that isn't part of an abbreviation (simple heuristic)
        first_sentence = first_job_desc.split('.')[0].strip()
        if len(first_sentence) > 20 and len(first_sentence) < 120:
            # Clean and use if reasonable
            strength = first_sentence.lower().replace("we're", "they are").replace("i've", "has").replace("i ", "candidate ")
            
    strength_sentence = f"Demonstrated impact where they {strength}"
    
    # Construct concern sentence based on notice period and active flag
    notice = row.get('notice_period_days', 0)
    open_to_work = row.get('open_to_work_flag', True)
    
    if notice > 30 and not open_to_work:
        concern_sentence = f"Note: candidate has a {notice}-day notice period and is not actively looking."
    elif notice > 30:
        concern_sentence = f"Note: candidate has a {notice}-day notice period."
    elif not open_to_work:
        concern_sentence = "Note: candidate is not actively looking."
    else:
        concern_sentence = "Available immediately."
        
    reasoning = f"{years} years of {domain} experience with expertise in {top_skills}. {strength_sentence}. {concern_sentence}"
    return reasoning

def main():
    parser = argparse.ArgumentParser(description="Online Ranker for Redrob Hackathon")
    parser.add_argument('--candidates', type=str, default='./candidates.parquet', help="Path to candidates.parquet")
    parser.add_argument('--embeddings', type=str, default='./embeddings.npy', help="Path to embeddings.npy")
    parser.add_argument('--out', type=str, default='./submission.csv', help="Output CSV path")
    parser.add_argument('--topk', type=int, default=500, help="Number of candidates to retrieve by semantic similarity (e.g., 500, 1000, 2000)")
    parser.add_argument('--benchmark', action='store_true', help="Run retrieval cutoff benchmarks (500,1000,2000) and report NDCG@10")
    parser.add_argument('--jd', type=str, required=True, help="Path to a text file containing the job description")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.candidates):
        print(f"Error: {args.candidates} does not exist. Please run offline_precompute.py first.")
        sys.exit(1)
        
    if not os.path.exists(args.embeddings):
        print(f"Error: {args.embeddings} does not exist. Please run offline_precompute.py first.")
        sys.exit(1)

    print("Loading precomputed candidate metadata and embeddings...")
    df = pd.read_parquet(args.candidates)
    embeddings = np.load(args.embeddings)
    
    # Safety validation: ensure df and embeddings line up
    assert len(df) == len(embeddings), f"Mismatch: df rows {len(df)} vs embeddings {len(embeddings)}"
    
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # Define Job Description text
    # Load JD text from file
    with open(args.jd, encoding='utf-8') as f:
        jd_text = f.read().strip()
    # Normalize JD text (lowercase, strip extra whitespace)
    jd_text = ' '.join(jd_text.split()).lower()
    
    print("Embedding job description...")
    jd_embedding = model.encode([jd_text], convert_to_numpy=True, normalize_embeddings=True)[0]

    # ------------------------------------------------------------------
    # Tier priority via embedding similarity to tier anchor texts
    # ------------------------------------------------------------------
    tier_anchor_texts = {
        'A': "search retrieval ranking information retrieval dense retrieval bm25 vector database",
        'B': "nlp language models llm transformer fine‑tuning embeddings neural networks",
        'C': "computer vision image classification object detection convolutional networks",
        'D': "data engineering pipelines spark airflow kafka sql etl backend infrastructure"
    }
    # Load or compute anchor embeddings (offline to avoid repeated encoding)
    anchor_embeds_path = os.path.join(os.path.dirname(__file__), "anchor_embeds.npz")
    if os.path.exists(anchor_embeds_path):
        loaded = np.load(anchor_embeds_path)
        anchor_embeds = {k: loaded[k] for k in loaded.files}
    else:
        anchor_embeds = {k: model.encode([v], convert_to_numpy=True, normalize_embeddings=True)[0]
                        for k, v in tier_anchor_texts.items()}
        np.savez(anchor_embeds_path, **anchor_embeds)
    # Cosine similarity between JD and each anchor
    anchor_sims = {k: float(np.dot(jd_embedding, a)) for k, a in anchor_embeds.items()}
    # Sort tiers by descending similarity to derive priority list
    tier_priority = [t for t, _ in sorted(anchor_sims.items(), key=lambda item: item[1], reverse=True)]
    # Ensure UNKNOWN tier is present for fallback
    if 'UNKNOWN' not in tier_priority:
        tier_priority.append('UNKNOWN')
    print(f"Derived tier priority from JD: {tier_priority}")
    
    print("Computing cosine similarity...")
    # Both candidate and JD embeddings are pre-normalized; direct dot product yields cosine similarity
    similarities = np.dot(embeddings, jd_embedding)
    
    df['semantic_similarity'] = similarities
    
    # Retrieve top candidates via semantic similarity (configurable)
    topk = args.topk
    print(f"Retrieving top {topk} candidates by hybrid retrieval (0.7 semantic + 0.3 template)...")
    # Simple template relevance: 1 if candidate's final_tier matches the highest priority tier, else 0
    highest_tier = tier_priority[0] if tier_priority else 'UNKNOWN'
    df['template_relevance'] = (df['final_tier'] == highest_tier).astype(float)
    df['hybrid_score'] = 0.7 * df['semantic_similarity'] + 0.3 * df['template_relevance']
    df_top = df.nlargest(topk, 'hybrid_score').copy()

    # ------------------------------------------------------------------
    # JD‑dependent weighted must‑have skill scoring
    # ------------------------------------------------------------------
    # Simple skill weight dictionary (higher weight for critical skills)
    skill_weights = {
        'faiss': 3.0,
        'pinecone': 3.0,
        'ndcg': 3.0,
        'mrr': 2.5,
        'map': 2.5,
        'vector': 2.0,
        'database': 2.0,
        'retrieval': 2.0,
        'search': 2.0,
        'python': 1.0,
        'sql': 1.5,
        'spark': 1.5,
        'airflow': 1.5,
        'tensorflow': 1.5,
        'pytorch': 1.5,
        'nlp': 2.0,
        'llm': 2.5,
        'computer': 2.0,
        'vision': 2.0,
        'image': 2.0,
        'pipeline': 1.5
    }
    # Extract JD skill tokens (lowercase, split on non‑alphanum)
    import re
    jd_skill_tokens = set(re.findall(r"[a-zA-Z0-9]+", jd_text))
    # Compute weighted match per candidate
    def compute_weighted_match(skills_list):
        # Accept list of strings or list of dicts (e.g., {'name': 'faiss'})
        if not isinstance(skills_list, list):
            return 0.0
        cand_skills = set()
        for s in skills_list:
            if isinstance(s, dict):
                # common keys that may hold the skill name
                name = s.get('name') or s.get('skill') or s.get('title')
                if isinstance(name, str):
                    cand_skills.add(name.lower())
            elif isinstance(s, str):
                cand_skills.add(s.lower())
        intersect = jd_skill_tokens.intersection(cand_skills)
        if not intersect:
            return 0.0
        total_weight = sum(skill_weights.get(tok, 1.0) for tok in intersect)
        max_weight = sum(skill_weights.get(tok, 1.0) for tok in jd_skill_tokens)
        return total_weight / max_weight if max_weight > 0 else 0.0
    df_top['must_have_skill_match'] = df_top['skills'].apply(compute_weighted_match)
    
    # Compute 7-signal weighted score (no tier bonus)
    print("Scoring candidates on 7 signals...")
    weighted_score = (
        0.35 * df_top['semantic_similarity'] +
        0.25 * df_top['must_have_skill_match'] +  # now JD‑specific weighted score
        0.15 * df_top['experience_band_fit'] +
        0.10 * df_top['title_relevance'] +
        0.07 * df_top['career_progression'] +
        0.05 * df_top['availability'] +
        0.03 * df_top['recency']
    )
    
    # Apply Honeypot penalty warning multiplier
    weighted_score = np.where(df_top['has_penalty'], weighted_score * 0.85, weighted_score)
    
    # Remove any remaining tier‑bonus (already excluded in score computation)
    df_top['score'] = weighted_score

    # Structural tier gating with dynamic tier order derived from JD
    
    # Build a CategoricalDtype based on the JD‑derived priority list
    # Build a CategoricalDtype based on the JD‑derived priority list, ensuring UNKNOWN is present
    priority_order = tier_priority.copy()
    if 'UNKNOWN' not in priority_order:
        priority_order.append('UNKNOWN')
    tier_order = pd.CategoricalDtype(priority_order, ordered=True)
    df_top['final_tier'] = df_top['final_tier'].astype(tier_order)
    
    # Define tie‑breaker columns for deterministic ordering
    # Higher score, higher must_have_skill_match, higher semantic_similarity, lower candidate_id
    df_top['candidate_id_int'] = df_top['candidate_id'].str.replace('CAND_', '').astype(int)
    
    # Sort candidates within each tier by the tuple (score, must_have_skill_match, semantic_similarity, -candidate_id_int)
    sort_cols = ['final_tier', 'score', 'must_have_skill_match', 'semantic_similarity', 'candidate_id_int']
    df_top = df_top.sort_values(sort_cols, ascending=[True, False, False, False, True])
    
    # Explicit top‑20 selection respecting tier hierarchy
    top20_list = []
    for tier in priority_order:
        tier_candidates = df_top[df_top['final_tier'] == tier]
        for _, row in tier_candidates.iterrows():
            if len(top20_list) >= 20:
                break
            top20_list.append(row.name)  # store index
        if len(top20_list) >= 20:
            break
    top20_idx = top20_list
    top20 = df_top.loc[top20_idx]
    
    # Remaining up to rank 100 sorted purely by the tie‑breaker tuple (score, must_have_skill_match, semantic_similarity, candidate_id_int)
    remaining = df_top.drop(index=top20_idx).sort_values(['score', 'must_have_skill_match', 'semantic_similarity', 'candidate_id_int'], ascending=[False, False, False, True]).head(80)
    
    # Combine
    df_top = pd.concat([top20, remaining])
    
    # Clean helper column
    df_top.drop(columns=['candidate_id_int'], inplace=True)

    print("Sorting and stabilizing rank (final tie‑breakers)...")
    
    # Select top 100
    df_final = df_top.head(100).copy()
    df_final['rank'] = range(1, 101)
    
    # Generate Reasoning
    print("Generating recruiter reasonings...")
    df_final['reasoning'] = df_final.apply(generate_reasoning, axis=1)
    
    # Export compliant CSV
    print(f"Exporting final submission to {args.out}...")
    submission_df = df_final[['candidate_id', 'rank', 'score', 'reasoning']].copy()
    submission_df.to_csv(args.out, index=False, encoding='utf-8')
    
    # Conditional Benchmark
    if args.benchmark:
        print("Running benchmark mode...")
        def compute_ndcg_at_10(relevance_scores):
            dcg = relevance_scores[0] + sum(rel / np.log2(i + 1) for i, rel in enumerate(relevance_scores[1:], 2))
            idcg = sum(1 / np.log2(i + 1) for i in range(1, len(relevance_scores) + 1))
            return dcg / idcg if idcg > 0 else 0
        for k in [500, 1000, 2000]:
            print(f"Benchmarking with topk={k}...")
            df_b = df.nlargest(k, 'hybrid_score').copy()
            df_b['score'] = (0.35 * df_b['semantic_similarity'] + 0.25 * df_b['must_have_skill_match'] + 0.15 * df_b['experience_band_fit'] + 0.10 * df_b['title_relevance'] + 0.07 * df_b['career_progression'] + 0.05 * df_b['availability'] + 0.03 * df_b['recency'])
            df_b = df_b.sort_values(['score'], ascending=[False]).head(10)
            ndcg = compute_ndcg_at_10(df_b['ground_truth_relevance'].values) if 'ground_truth_relevance' in df_b else 0.0
            print(f"Benchmark results for topk={k}: NDCG@10={ndcg:.4f}")
    
    print("Done! Submission file is ready.")

if __name__ == '__main__':
    main()
