import json
import os
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def dcg(relevances, k):
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size == 0:
        return 0.0
    return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))

def ndcg(relevances, k):
    actual_dcg = dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal_relevances, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg

def compute_map(relevances, rel_threshold=2):
    binary_rel = [1 if r >= rel_threshold else 0 for r in relevances]
    total_rel = sum(binary_rel)
    if total_rel == 0:
        return 0.0
        
    ap = 0.0
    rel_found = 0
    for idx, rel in enumerate(binary_rel):
        if rel == 1:
            rel_found += 1
            precision_at_i = rel_found / (idx + 1)
            ap += precision_at_i
            
    return ap / total_rel

def evaluate_ranking(df, weights, sort_mode='bonus'):
    # Compute weighted score
    w_score = (
        weights[0] * df['semantic_similarity'] +
        weights[1] * df['must_have_skill_match'] +
        weights[2] * df['experience_band_fit'] +
        weights[3] * df['title_relevance'] +
        weights[4] * df['career_progression'] +
        weights[5] * df['availability'] +
        weights[6] * df['recency']
    )
    
    # Honeypot penalty
    w_score = np.where(df['has_penalty'], w_score * 0.85, w_score)
    
    if sort_mode == 'bonus':
        # Tier Priority Bonus
        tier_bonuses = {'A': 0.30, 'B': 0.20, 'C': 0.10, 'D': 0.00}
        bonus = df['tier'].map(tier_bonuses).fillna(0.0)
        df['score'] = w_score + bonus
        
        # Sort by score descending (cross-tier competition allowed)
        df_sorted = df.sort_values(
            by=['score', 'tier', 'must_have_skill_match', 'semantic_similarity', 'candidate_id'],
            ascending=[False, True, False, False, True]
        )
    else:  # sort_mode == 'concat'
        df['score'] = w_score
        # Sort strictly by Tier letter first, then score descending
        df_sorted = df.sort_values(
            by=['tier', 'score', 'must_have_skill_match', 'semantic_similarity', 'candidate_id'],
            ascending=[True, False, False, False, True]
        )
        
    relevances = df_sorted['relevance'].tolist()
    
    n_10 = ndcg(relevances, 10)
    n_20 = ndcg(relevances, 20)
    m_ap = compute_map(relevances)
    p_10 = sum(1 for r in relevances[:10] if r >= 2) / 10.0
    
    return n_10, n_20, m_ap, p_10

def main():
    gold_json = './gold_set_labels.json'
    parquet_path = './candidates.parquet'
    embeddings_path = './embeddings.npy'
    
    if not os.path.exists(gold_json):
        print(f"Error: {gold_json} not found. Run curate_gold_set.py first.")
        sys.exit(1)
        
    if not os.path.exists(parquet_path) or not os.path.exists(embeddings_path):
        print(f"Error: Precomputed files not found. Run offline_precompute.py first.")
        sys.exit(1)
        
    # Load gold set labels
    with open(gold_json, encoding='utf-8') as f:
        gold_labels = json.load(f)
        
    print(f"Loaded {len(gold_labels)} gold set labels.")
    
    # Load precomputed data
    df = pd.read_parquet(parquet_path)
    embeddings = np.load(embeddings_path)
    
    # Compute semantic similarity on the fly
    print("Computing semantic similarity for gold set...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    jd_text = (
        "Senior AI Engineer — Founding Team. Company: Redrob AI. "
        "Embeddings-based retrieval systems, Sentence transformers, OpenAI embeddings, vector database, hybrid search, "
        "Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, evaluation frameworks, NDCG, MRR, MAP, "
        "Python, LoRA, QLoRA, PEFT, learning to rank, XGBoost, marketplace products, Pune/Noida, India, notice period sub-30 days."
    )
    jd_embedding = model.encode([jd_text], convert_to_numpy=True, normalize_embeddings=True)[0]
    
    similarities = np.dot(embeddings, jd_embedding)
    df['semantic_similarity'] = similarities
    
    # Filter only candidates in the gold set
    df_gold = df[df['candidate_id'].isin(gold_labels.keys())].copy()
    df_gold['relevance'] = df_gold['candidate_id'].map(gold_labels)
    
    print(f"Found {len(df_gold)} of the gold set candidates in the precomputed parquet database.")
    
    # Baseline weights
    # [semantic, must-have, exp_fit, title_rel, career_prog, availability, recency]
    baseline_weights = [0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03]
    signal_names = [
        "Semantic Similarity (0.35)",
        "Must-Have Skill Match (0.25)",
        "Experience Band Fit (0.15)",
        "Title Relevance (0.10)",
        "Career Progression (0.07)",
        "Availability (0.05)",
        "Recency (0.03)"
    ]
    
    # Evaluate mode 1: Tier Bonus
    n_10_b, n_20_b, m_ap_b, p_10_b = evaluate_ranking(df_gold, baseline_weights, sort_mode='bonus')
    
    # Evaluate mode 2: Strict Tier Concatenation
    n_10_c, n_20_c, m_ap_c, p_10_c = evaluate_ranking(df_gold, baseline_weights, sort_mode='concat')
    
    print("\n" + "="*50)
    print(f"{'STRATEGY PERFORMANCE COMPARISON':^50}")
    print("="*50)
    print(f"METRIC       | TIER BONUS  | STRICT CONCAT")
    print(f"-------------+-------------+--------------")
    print(f"NDCG@10      | {n_10_b:.4f}      | {n_10_c:.4f}")
    print(f"NDCG@20      | {n_20_b:.4f}      | {n_20_c:.4f}")
    print(f"MAP          | {m_ap_b:.4f}      | {m_ap_c:.4f}")
    print(f"P@10         | {p_10_b:.4f}      | {p_10_c:.4f}")
    print("="*50)
    
    # Let's perform ablation study using the Tier Bonus (or whichever is the baseline)
    print("\n" + "="*50)
    print(f"{'ABLATION STUDY (TIER BONUS - NDCG@10 change)':^50}")
    print("="*50)
    for idx, name in enumerate(signal_names):
        temp_weights = baseline_weights.copy()
        temp_weights[idx] = 0.0
        w_sum = sum(temp_weights)
        if w_sum > 0:
            temp_weights = [w / w_sum for w in temp_weights]
            
        ab_n10, _, _, _ = evaluate_ranking(df_gold, temp_weights, sort_mode='bonus')
        diff = ab_n10 - n_10_b
        print(f"Ablating {name:<30} -> NDCG@10: {ab_n10:.4f} (Delta: {diff:+.4f})")
    print("="*50)

    # Perform ablation study using Strict Tier Concatenation
    print("\n" + "="*50)
    print(f"{'ABLATION STUDY (STRICT CONCAT - NDCG@10 change)':^50}")
    print("="*50)
    for idx, name in enumerate(signal_names):
        temp_weights = baseline_weights.copy()
        temp_weights[idx] = 0.0
        w_sum = sum(temp_weights)
        if w_sum > 0:
            temp_weights = [w / w_sum for w in temp_weights]
            
        ab_n10, _, _, _ = evaluate_ranking(df_gold, temp_weights, sort_mode='concat')
        diff = ab_n10 - n_10_c
        print(f"Ablating {name:<30} -> NDCG@10: {ab_n10:.4f} (Delta: {diff:+.4f})")
    print("="*50)

if __name__ == '__main__':
    main()
