# Implementation Plan - Intelligent Candidate Discovery & Ranking Pipeline (Final)

This document details the final implementation plan for the candidate discovery and ranking system. The goal is to identify and rank the top 100 candidates from a 100,000-candidate pool (`candidates.jsonl`) for the **Senior AI Engineer — Founding Team** role, generating recruiter-style reasoning and outputting a compliant CSV file.

## User Review Required

> [!IMPORTANT]
> **1. Description-Heavy Tier Classification**
> Rather than matching titles alone, candidates are classified into Tiers A through D based on a hybrid score:
> $$\text{tier\_score} = 0.2 \times \text{title\_match} + 0.8 \times \text{career\_description\_match}$$
> - `title_match`: 1.0 if current or past titles match the tier keywords, else 0.0.
> - `career_description_match`: 1.0 if any past job description or profile summary contains the tier keywords, else 0.0.
> - If `tier_score >= 0.25`, the candidate qualifies for that tier. This ensures a "Software Engineer" who built "vector search using FAISS" is correctly categorized under Tier A. Candidates are checked sequentially (Tier A $\rightarrow$ Tier B $\rightarrow$ Tier C $\rightarrow$ Tier D). Candidates not matching any tier are classified as Tier E and excluded.

> [!IMPORTANT]
> **2. Honeypot Soft Thresholds**
> To handle potentially noisy dates safely:
> - $\text{honeypot\_score} > 0.65$: Hard reject (disqualified, never ranked).
> - $0.45 \le \text{honeypot\_score} \le 0.65$: Soft penalty zone. Candidates are kept but their final score is penalized by a multiplier of $0.85$:
>   $$\text{score} = \text{score} \times 0.85$$
> - $\text{honeypot\_score} < 0.45$: No penalty.

> [!IMPORTANT]
> **3. Flexible Tier Priority Bonuses**
> To allow outstanding Tier B/C candidates to outrank weaker Tier A candidates, we will add a tier-specific priority bonus to the weighted score:
> - **Tier A**: +0.30
> - **Tier B**: +0.20
> - **Tier C**: +0.10
> - **Tier D**: +0.00
> $$\text{final\_score} = \text{weighted\_score} + \text{tier\_bonus}$$

> [!IMPORTANT]
> **4. Tuple Stabilization Key**
> Sorting and tie-breaking will be performed using the following key (sorted ascending, with negative signs for descending sorts):
> `rank_key = (-final_score, tier_letter, -must_have_skill_match, -semantic_similarity, candidate_id)`
> *Note: If we sorted strictly with `tier_letter` as the primary key, it would enforce rigid concatenation (preventing any Tier B from beating Tier A), contradicting the tier bonus mechanism. Therefore, we use `-final_score` as the primary sort key (which incorporates the tier bonus) and `tier_letter` as a secondary tie-breaker.*

## Decided Specifications
1. **Notice Period**: Notice period $\le 30$ days has a multiplier of 1.0; $31-90$ days decays linearly to 0.2; $>90$ days gets 0.0.
2. **Anchor Date**: Locked to `2026-06-27`.
3. **Gold Labels**: Target gold-set size to 50–80 candidates.
4. **Schema Verification Checklist**: All fields verified against the actual schema.

## Proposed Changes

We will build the pipeline in the [jolly-darwin](file:///C:/Users/shegd/Documents/antigravity/jolly-darwin) directory.

### Component 1: Offline Precomputation & Features (`offline_precompute.py`)

#### [NEW] [offline_precompute.py](file:///C:/Users/shegd/Documents/antigravity/jolly-darwin/offline_precompute.py)
This script runs offline without time limits to process the 100,000-candidate pool:
- **Streaming Reader**: Streams `candidates.jsonl` line-by-line.
- **Honeypot Filtering**: Calculates `honeypot_score` using the 5 weighted heuristics:
  $$\text{honeypot\_score} = 0.40 \times h_1 + 0.25 \times h_2 + 0.15 \times h_3 + 0.12 \times h_4 + 0.08 \times h_5$$
  - $h_1$ (`experience_predates_company_founding`): Compares start dates against our company founding lookup table.
  - $h_2$ (`skill_career_mismatch`): Detects claims of LTR/FAISS expert skills where past titles are exclusively frontend/QA.
  - $h_3$ (`graduation_year_anomaly`): Detects Master's or Ph.D. end years that are chronologically earlier than Bachelor's end years.
  - $h_4$ (`overlapping_job_timelines`): Identifies overlapping full-time jobs exceeding 90 days.
  - $h_5$ (`expert_skill_to_experience_ratio`): Checks for $\ge 3$ expert skills with exactly $0$ months duration.
  If $2+$ heuristics fire, the score is multiplied by $1.15$ (capped at $1.0$).
  - Candidates with $\text{honeypot\_score} > 0.65$ are excluded.
  - Candidates with $0.45 \le \text{honeypot\_score} \le 0.65$ are flagged with a penalty flag.
- **Hard Filters**: Rejects candidates with `role_category` in `{sales, marketing, hr, accounting}` (mapped from current title) or `years_experience < 3`.
- **Hybrid Tier Classification**: Runs the sequential scoring logic for Tiers A–D. Generic Tier E candidates are filtered out.
  - Keywords for each tier:
    - **Tier A (Ranking/Search/IR)**: `ranking`, `search`, `retrieval`, `recommendation`, `ltr`, `information retrieval`, `dense retrieval`, `bm25`, `faiss`, `pinecone`, `vector database`.
    - **Tier B (General ML/NLP/LLM)**: `machine learning`, `nlp`, `llm`, `deep learning`, `transformers`, `fine-tuning`, `rag`, `pytorch`.
    - **Tier C (Computer Vision / Applied ML)**: `computer vision`, `image classification`, `object detection`, `speech`, `tts`, `asr`, `ocr`, `gan`.
    - **Tier D (Data Engineering / Backend with ML exposure)**: `data engineer`, `backend`, `analytics engineer`, `spark`, `airflow`, `kafka`, `snowflake`.
- **Embedding Generation**: Generates 384-dimensional candidate embeddings using `all-MiniLM-L6-v2`.
- **Storage**: Saves candidate embeddings as a compact NumPy `.npy` array, and structured metadata as `candidates.parquet`.

---

### Component 2: Online Ranker (`rank.py`)

#### [NEW] [rank.py](file:///C:/Users/shegd/Documents/antigravity/jolly-darwin/rank.py)
The CLI script that produces the final submission CSV within 5 minutes:
- **JD Parser & Embedding**: Computes the embedding of the JD using the local sentence-transformer model.
- **Retrieval**: Selects the top 500 candidates via matrix multiplication cosine similarity.
- **Orthogonal 7-Signal Score**:
  - `semantic_similarity` (0.35): Cosine similarity score.
  - `must_have_skill_match` (0.25): Matching list of vector search, python, embeddings, and ranking evaluation skills.
  - `experience_band_fit` (0.15): Penalty-based fit scoring for the 5-9 years range.
  - `title_relevance` (0.10): Relevance of titles against AI engineering and search roles.
  - `career_progression` (0.07): Duration of roles (penalizing job-hoppers).
  - `availability` (0.05): notice period scoring (sub-30-day preferred) combined with `open_to_work_flag`.
  - `recency` (0.03): Exponential decay based strictly on time since `last_active_date` relative to `2026-06-27`.
  - **Honeypot Penalty Application**: If a candidate has the penalty flag from Component 1, their final score is multiplied by $0.85$.
  - **Tier Priority Bonus**: The tier bonus (+0.30, +0.20, +0.10, or +0.00) is added to form the `final_score`.
- **Sorting & Tuple Stabilization**:
  Sorts candidates by `rank_key = (-final_score, tier_letter, -must_have_skill_match, -semantic_similarity, candidate_id)`.
  Takes the top 100, generates recruiter-style template reasoning, and writes the CSV.

---

### Component 3: Unit Tests & Validation (`test_pipeline.py`)

#### [NEW] [test_pipeline.py](file:///C:/Users/shegd/Documents/antigravity/jolly-darwin/test_pipeline.py)
Pre-submission tests:
- Asserts that the weights for the 7 signals sum exactly to $1.0$.
- Verifies that `validate_submission.py` passes on the output CSV.
- Tests that the online ranking runs under 2 minutes on CPU.

---

### Component 4: Gold-Set Validation (`validate_gold_set.py`)

#### [NEW] [validate_gold_set.py](file:///C:/Users/shegd/Documents/antigravity/jolly-darwin/validate_gold_set.py)
Validation script for local accuracy testing:
- **Gold-Set Storage**: A manually curated JSON file `gold_set_labels.json` containing 50–80 selected candidate IDs with hand-labeled relevance scores (e.g., 0 to 3).
- **Evaluation Engine**: Runs the ranking pipeline on the gold-set candidates and computes NDCG@10, NDCG@20, MAP, and P@10.
- **Ablation Mode**: Temporarily zeroes out individual signal weights to log the delta change in NDCG, identifying which signals contribute most to search accuracy.

## Verification Plan

### Automated Tests
- Run `pytest test_pipeline.py`.
- Run validation: `python validate_submission.py <team_id>.csv`.
- Run gold-set accuracy check: `python validate_gold_set.py`.

### Manual Verification
- Review top 20 and bottom 10 reasoning lines to check for recruiter-style tone, specific profile facts, and lack of hallucinations.
