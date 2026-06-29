# Redrob Candidate Ranking System

## Overview
This system evaluates large pools of candidates against job descriptions and efficiently selects the top 100 profiles using a hybrid pipeline (Dense FAISS + BM25 Sparse + Cross-Encoder Reranking + Rule-based Anomaly Detection).

## Execution Instructions

**1. Offline Precomputation (Mandatory First Step)**
You must run the precomputation script offline first to index the full candidate dataset. This generates the necessary dense/sparse index artifacts.

```bash
python scripts/precompute.py --candidates ./candidates.jsonl --out_dir artifacts
```
This will process the entire `candidates.jsonl` file and generate:
- `artifacts/dense.faiss`
- `artifacts/dense_candidates.json`
- `artifacts/bm25.pkl`
- `artifacts/candidate_cache.pkl`

**2. Online Ranking**
Once the offline artifacts are generated, execute the final ranking step:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
The `--candidates` argument here is mostly ignored since the online ranker will rapidly ingest the offline artifacts. The system will retrieve the top candidates, run the cross-encoder, score anomalies, break ties deterministically, and generate `submission.csv` containing the exact Top 100 in ~10 seconds.
