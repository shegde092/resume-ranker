# Walkthrough - Intelligent Candidate Discovery & Ranking Pipeline

This document walks through the implementation, verification, and validation details of the candidate discovery and ranking system for the **Senior AI Engineer — Founding Team** role.

## Changes Made

### 1. Unified Compact Career Representation
- Incorporated the exact benchmark representation structure from the user's script:
  `f"Title: {title}. Headline: {headline}. Summary: {summary}. Career: {career_summary}. Skills: {skills_summary}."`
  where `career_summary` parses the top 2 job history titles and descriptions (clamped to 300 characters each).
- Generated normalized embeddings directly from the local `all-MiniLM-L6-v2` encoder by passing `normalize_embeddings=True` in `offline_precompute.py`.
- Simplified the online retrieval stage in `rank.py` to use a direct matrix-multiplication dot product `np.dot(embeddings, jd_embedding)` which yields cosine similarity mathematically.

### 2. Strict Tier Concatenation sorting
- Modified the sorting stabilizer in `rank.py` to enforce **Strict Tier Concatenation** (sorting by `['tier', 'score', ...]` with ascending `[True, False, ...]`), guaranteeing that Tier A candidates are ranked above Tier B candidates regardless of individual scoring variances.

---

## Strategy Performance Comparison

We evaluated both the **Tier Priority Bonus** and **Strict Tier Concatenation** strategies against our gold-labeled validation set (55 curated profiles).

```text
==================================================
         STRATEGY PERFORMANCE COMPARISON          
==================================================
METRIC       | TIER BONUS  | STRICT CONCAT
-------------+-------------+--------------
NDCG@10      | 0.8328      | 0.8328
NDCG@20      | 0.8617      | 0.8296
MAP          | 0.8012      | 0.8133
P@10         | 0.6000      | 0.6000
==================================================
```
* Both strategies are highly competitive. **Tier Bonus** achieves a higher NDCG@20 of **0.8617** (allowing exceptional Tier B candidates to outperform marginal Tier A candidates at rank boundaries), while **Strict Concat** delivers slightly higher Mean Average Precision of **0.8133**.
* In accordance with user guidance, the final submission uses **Strict Tier Concatenation** for structural safety.

---

## Ablation Study & Metric Analysis

### 1. Strict Concat Ablation Results
Ablating each signal (setting its weight to 0.0 and evaluating NDCG@10 change) yields:
```text
==================================================
 ABLATION STUDY (STRICT CONCAT - NDCG@10 change)  
==================================================
Ablating Semantic Similarity (0.35)     -> NDCG@10: 0.8002 (Delta: -0.0326)
Ablating Must-Have Skill Match (0.25)   -> NDCG@10: 0.6949 (Delta: -0.1378)
Ablating Experience Band Fit (0.15)     -> NDCG@10: 0.9169 (Delta: +0.0842)
Ablating Title Relevance (0.10)         -> NDCG@10: 0.8328 (Delta: +0.0000)
...
```

### 2. Signal Verification & Rationale
* **Semantic Similarity (-0.0326):** Removing the semantic similarity signal now degrades the NDCG score as expected. This validates that the new career-rich text representation is highly informative and the cosine similarity behaves as a clean, healthy ranking driver.
* **Must-Have Skill Match (-0.1378):** Confirming its role as the primary lexical rank filter.
* **Experience Band Fit (+0.0842):** Removing the experience fit signal increases NDCG. This is expected: the gold validation set includes exceptionally strong candidates who have more than 9 years of experience. The JD targets a Senior AI Engineer (5-9 years experience), so our experience fit signal penalizes candidates with >9 years experience. When the experience signal is ablated, these veterans are no longer penalized, naturally raising the NDCG score on the hand-labeled validation set.

---

## Verification & Validation Status
* **Unit Tests (`test_pipeline.py`):** 7/7 tests passed.
* **Official Validator:** Running `validate_submission.py` on the strict-concat `submission.csv` outputs: `Submission is valid`.
