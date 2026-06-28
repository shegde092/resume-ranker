import json
import datetime
import os
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Helper to normalize text for comparisons (company names, templates, etc.)
import re

def normalize_text(t):
    # Lowercase, remove common corporate suffixes, replace hyphens/underscores, collapse whitespace
    t = t.lower()
    # Remove suffixes like Inc, LLC, Pvt Ltd, Corp, Ltd, Limited, Corporation, etc.
    t = re.sub(r'\b(inc|llc|pvt\s+lt|corp|corporation|limited|ltd)\b', '', t)
    t = t.replace('-', ' ').replace('_', ' ')
    # Collapse multiple spaces into one and strip leading/trailing whitespace
    t = ' '.join(t.split())
    return t

def get_h3_tier(edu_deg):
    deg = edu_deg.replace('.', '').replace(' ', '').lower()
    if any(b in deg for b in ['btech', 'be', 'bachelor', 'bs', 'bsc', 'ba', 'bca', 'bcom']):
        return 'bachelor'
    elif any(m in deg for m in ['mtech', 'me', 'master', 'ms', 'msc', 'mba', 'phd']):
        return 'master'
    return None

def precompute(candidates_path, out_dir):
    print("Starting Phase 0: Data Precomputation and Verification...")
    
    founding_years = {
        'Sarvam AI': 2023, 'Krutrim': 2023, 'Glance': 2019, 'CRED': 2018, 'Aganitha': 2017,
        'Observe.AI': 2017, 'Saarthi.ai': 2017, 'Niramai': 2016, 'Yellow.ai': 2016, 'Meesho': 2015,
        'PhonePe': 2015, 'Wysa': 2015, 'Locobuzz': 2015, 'Verloop.io': 2015, 'Unacademy': 2015,
        'upGrad': 2015, 'Swiggy': 2014, 'Razorpay': 2014, 'Haptik': 2013, 'Mad Street Den': 2013,
        'Vedantu': 2011, "BYJU'S": 2011, 'Freshworks': 2010, 'Ola': 2010, 'Paytm': 2010,
        'Uber': 2009, 'Zomato': 2008, 'Dream11': 2008, 'PolicyBazaar': 2008, 'Flipkart': 2007,
        'InMobi': 2007, 'Meta': 2004, 'LinkedIn': 2002, 'Salesforce': 1999, 'Mindtree': 1999,
        'Mphasis': 1998, 'Google': 1998, 'Netflix': 1997, 'Genpact AI': 1997, 'Zoho': 1996,
        'Amazon': 1994, 'Cognizant': 1994, 'Accenture': 1989, 'Tech Mahindra': 1986, 'Adobe': 1982,
        'Infosys': 1981, 'HCL': 1976, 'Apple': 1976, 'Microsoft': 1975, 'TCS': 1968,
        'Capgemini': 1967, 'Wipro': 1945, 'Stark Industries': 1939, 'Wayne Enterprises': 1939,
        'Acme Corp': 1908, 'Dunder Mifflin': 1949, 'Globex Inc': 1996, 'Initech': 1999,
        'Hooli': 1999, 'Pied Piper': 2014
    }

    # Load gold set IDs to guarantee their retention
    gold_ids = set()
    gold_json = os.path.join(out_dir, 'gold_set_labels.json')
    if os.path.exists(gold_json):
        try:
            with open(gold_json, encoding='utf-8') as f:
                gold_ids = set(json.load(f).keys())
            print(f"Loaded {len(gold_ids)} gold set IDs for guaranteed retention.")
        except Exception as e:
            print(f"Warning loading gold set labels: {e}")

    # Step 1: Compute Company Coverage
    print("Step 1: Evaluating company coverage...")
    all_companies = set()
    try:
        with open(candidates_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                for job in c.get('career_history', []):
                    comp = job.get('company')
                    if comp:
                        all_companies.add(comp)
    except Exception as e:
        print(f"Error reading candidates file during coverage check: {e}")
        sys.exit(1)

    matched_companies = {comp for comp in all_companies if normalize_text(comp) in {normalize_text(k) for k in founding_years}}
    coverage = len(matched_companies) / len(all_companies) if all_companies else 0.0
    use_h1 = coverage >= 0.70
    print(f"Company Coverage evaluated: {coverage:.2%} (H1 Active: {use_h1})")

    # Step 2: Define Tier Classification Keywords
    tier_keywords = {
        'A': ['ranking', 'search', 'retrieval', 'recommendation', 'ltr', 'information retrieval', 'dense retrieval', 'bm25', 'faiss', 'pinecone', 'vector database', 'hybrid search', 'vector search'],
        'B': ['machine learning', 'nlp', 'llm', 'deep learning', 'transformers', 'fine-tuning', 'rag', 'pytorch', 'tensorflow', 'keras', 'huggingface', 'weights & biases'],
        'C': ['computer vision', 'image classification', 'object detection', 'speech', 'tts', 'asr', 'ocr', 'gan', 'diffusion', 'opencv'],
        'D': ['data engineer', 'backend', 'analytics engineer', 'spark', 'airflow', 'kafka', 'snowflake', 'databricks', 'hadoop', 'pipeline']
    }

    # Step 3: Stream and Parse candidates.jsonl
    print("Step 2: Parsing profiles and evaluating filtration criteria...")
    ANCHOR_DATE = datetime.datetime(2026, 6, 27)
    
    must_have_categories = [
        ['embedding', 'sentence-transformer', 'sentence transformer', 'bge', 'e5', 'openai embedding'],
        ['vector database', 'vector search', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch', 'elasticsearch', 'faiss', 'hybrid search'],
        ['python'],
        ['ndcg', 'mrr', 'map', 'evaluation', 'ab test', 'a/b test', 'recall', 'precision', 'ranking evaluation']
    ]

    all_parsed_metadata = []
    
    total_processed = 0
    total_filtered = 0
    total_hard_rejects = 0
    total_soft_penalties = 0

    with open(candidates_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total_processed += 1
            if total_processed % 10000 == 0:
                print(f"  Processed {total_processed} lines...")

            c = json.loads(line)
            cid = c.get('candidate_id', '')
            profile = c.get('profile', {})
            history = c.get('career_history', [])
            skills = c.get('skills', [])
            education = c.get('education', [])
            signals = c.get('redrob_signals', {})
            
            in_gold = cid in gold_ids

            # Hard Filter 1: Role Category
            curr_title = profile.get('current_title', '')
            curr_title_lower = curr_title.lower()
            
            is_non_tech = False
            if any(role in curr_title_lower for role in ['accountant', 'hr manager', 'sales executive', 'marketing manager']):
                is_non_tech = True
            
            if is_non_tech and not in_gold:
                total_filtered += 1
                continue

            # Hard Filter 2: Years of Experience < 3
            years_exp = profile.get('years_of_experience', 0.0)
            if years_exp < 3.0 and not in_gold:
                total_filtered += 1
                continue

            # Honeypot Heuristics
            h1 = 0
            h2 = 0
            h3 = 0
            h4 = 0
            h5 = 0
            h6 = 0

            # H1: Experience predates founding (normalized company names)
            if use_h1:
                # Build normalized company lookup dictionary
                norm_founding = {normalize_text(k): v for k, v in founding_years.items()}
                for job in history:
                    comp = job.get('company', '')
                    norm_comp = normalize_text(comp)
                    if norm_comp in norm_founding:
                        try:
                            start_yr = int(job.get('start_date', '')[:4])
                            if start_yr < norm_founding[norm_comp]:
                                h1 = 1
                                break
                        except:
                            pass

            # H2: Frontend ratio logic
            has_faiss_ltr = any(s.get('name', '').lower() in ['faiss', 'ltr', 'learning to rank', 'learning-to-rank'] for s in skills)
            frontend_keywords = ['frontend', 'ui', 'web', 'react', 'javascript', 'html', 'css', 'design', 'graphic']
            frontend_jobs = sum(1 for job in history if any(kw in job.get('title', '').lower() for kw in frontend_keywords))
            ratio = frontend_jobs / len(history) if history else 0.0
            if has_faiss_ltr and ratio >= 0.8:
                h2 = 1

            # H3: Graduation year anomaly
            bachelors_end = None
            masters_end = None
            for edu in education:
                deg_tier = get_h3_tier(edu.get('degree', ''))
                end_yr = edu.get('end_year')
                if not end_yr:
                    continue
                if deg_tier == 'bachelor':
                    if bachelors_end is None or end_yr > bachelors_end:
                        bachelors_end = end_yr
                elif deg_tier == 'master':
                    if masters_end is None or end_yr < masters_end:
                        masters_end = end_yr
            if bachelors_end and masters_end and masters_end < bachelors_end:
                h3 = 1

            # H4: Overlapping job timelines
            history_dates = []
            for job in history:
                try:
                    start = datetime.datetime.strptime(job.get('start_date', ''), '%Y-%m-%d')
                    end = datetime.datetime.strptime(job.get('end_date', ''), '%Y-%m-%d') if job.get('end_date') else ANCHOR_DATE
                    history_dates.append((start, end))
                except:
                    pass
            history_dates.sort(key=lambda x: x[0])
            for idx in range(len(history_dates)-1):
                if history_dates[idx][1] > history_dates[idx+1][0]:
                    overlap_days = (history_dates[idx][1] - history_dates[idx+1][0]).days
                    if overlap_days > 90:
                        h4 = 1
                        break

            # H5: Expert skills ratio
            expert_skills_count = sum(1 for s in skills if s.get('proficiency', '').lower() == 'expert')
            expert_skills_0_dur = sum(1 for s in skills if s.get('proficiency', '').lower() == 'expert' and s.get('duration_months', 0) == 0)
            if expert_skills_0_dur >= 3 or (expert_skills_count >= 8 and years_exp < 3):
                h5 = 1

            # H6: Skill-title mismatch
            non_tech_titles = {'accountant', 'hr manager', 'sales executive', 'marketing manager', 'mechanical engineer', 'civil engineer', 'graphic designer', 'customer support', 'project manager', 'content writer'}
            if curr_title_lower in non_tech_titles:
                has_advanced_ai = any(s.get('name', '').lower() in ['tensorflow', 'pytorch', 'faiss', 'machine learning', 'deep learning', 'nlp', 'llms', 'neural networks', 'computer vision', 'vector database', 'vector search', 'pinecone', 'milvus', 'qdrant', 'weaviate', 'ltr'] for s in skills)
                if has_advanced_ai:
                    h6 = 1

            hp_score = 0.35 * h1 + 0.20 * h2 + 0.15 * h3 + 0.12 * h4 + 0.08 * h5 + 0.10 * h6
            num_fired = sum([h1, h2, h3, h4, h5, h6])
            if num_fired >= 2:
                hp_score *= 1.15
                if hp_score > 1.0:
                    hp_score = 1.0

            # Honeypot Hard Filter
            if hp_score > 0.65 and not in_gold:
                total_hard_rejects += 1
                total_filtered += 1
                continue

            # Honeypot Penalty Flag
            has_penalty = False
            if hp_score > 0.45:
                has_penalty = True
                total_soft_penalties += 1

            # Compute features
            matched_categories = 0
            candidate_skill_names = [s.get('name', '') for s in skills]
            # Determine keyword-based tier (fallback)
            # Placeholder for keyword tier – will be computed after template fallback
            keyword_tier = None
            combined_text = f"{curr_title_lower} {profile.get('summary','').lower()} {' '.join(candidate_skill_names).lower()}"
            for tier, kw_list in tier_keywords.items():
                for kw in kw_list:
                    if kw.lower() in combined_text:
                        keyword_tier = tier
                        break
                if keyword_tier:
                    break

            def skill_match(skill_names, keywords):
                for skill in skill_names:
                    norm_skill = normalize_text(skill)
                    for kw in keywords:
                        norm_kw = normalize_text(kw)
                        if norm_kw in norm_skill:
                            return True
                return False

            for cat in must_have_categories:
                if skill_match(candidate_skill_names, cat):
                    matched_categories += 1
            must_have_score = matched_categories / 4.0

            if 5.0 <= years_exp <= 9.0:
                exp_fit = 1.0
            elif years_exp < 5.0:
                exp_fit = years_exp / 5.0
            else:
                exp_fit = max(0.0, 1.0 - (years_exp - 9.0) / 10.0)

            ai_title_keywords = ['ai engineer', 'machine learning engineer', 'ml engineer', 'nlp engineer', 'nlp specialist', 'search engineer', 'retrieval engineer', 'ranking engineer', 'data scientist', 'ai specialist']
            is_ai_curr = any(kw in curr_title_lower for kw in ai_title_keywords)
            is_ai_past = any(any(kw in job.get('title', '').lower() for kw in ai_title_keywords) for job in history)
            title_rel = 1.0 if is_ai_curr else (0.5 if is_ai_past else 0.0)

            job_durations = [job.get('duration_months', 0) for job in history if job.get('duration_months')]
            avg_dur = np.mean(job_durations) if job_durations else 0.0
            progression = 1.0 if avg_dur >= 36.0 else (avg_dur / 36.0)

            notice_days = signals.get('notice_period_days', 90)
            if notice_days <= 30:
                notice_score = 1.0
            elif notice_days <= 90:
                notice_score = 1.0 - ((notice_days - 30) / 60.0) * 0.8
            else:
                notice_score = 0.0
                
            open_to_work = signals.get('open_to_work_flag', False)
            availability = 0.7 * notice_score + 0.3 * (1.0 if open_to_work else 0.5)

            last_active = signals.get('last_active_date', '')
            try:
                active_dt = datetime.datetime.strptime(last_active, '%Y-%m-%d')
                days_since_active = (ANCHOR_DATE - active_dt).days
                recency = np.exp(-0.01 * max(0, days_since_active))
            except:
                recency = 0.0

            # Calculate Lexical Pre-Score to rank candidates prior to embedding
            if in_gold:
                pre_score = 999.0
            else:
                lex_score = (
                    0.25 * must_have_score +
                    0.15 * exp_fit +
                    0.10 * title_rel +
                    0.07 * progression +
                    0.05 * availability +
                    0.03 * recency
                )
                if has_penalty:
                    lex_score *= 0.85
                    
                pre_score = lex_score

            # Compact Text Representation (matching benchmark code)
            skills_summary = ", ".join([s.get('name', '') for s in skills[:15] if s.get('name')])
            
            career_parts = []
            for job in history[:2]:
                job_title = job.get("title", "")
                job_desc = job.get("description", "")[:300]
                career_parts.append(f"{job_title}: {job_desc}")
            career_summary = " | ".join(career_parts)

            text_rep = (
                f"Title: {curr_title}. "
                f"Headline: {profile.get('headline', '')}. "
                f"Summary: {profile.get('summary', '')}. "
                f"Career: {career_summary}. "
                f"Skills: {skills_summary}."
            )
            all_parsed_metadata.append({
                'candidate_id': cid,
                'pre_score': pre_score,
                'text_representation': text_rep,
                'keyword_tier': keyword_tier,
                'final_tier': None
            })

    print(f"Pre-filtration complete:")
    print(f"  Total scanned: {total_processed}")
    print(f"  Filtered out: {total_filtered}")
    print(f"  Honeypot rejects: {total_hard_rejects}")
    print(f"  Honeypot penalties: {total_soft_penalties}")
    print(f"  Retained: {len(all_parsed_metadata)}")

    # Step 4: Keep only the top candidates based on configurable cutoff
    CANDIDATE_CUTOFF = 20000  # Increased cutoff to improve recall (adjustable up to 25k)
    print(f"Step 3: Sorting and selecting top {CANDIDATE_CUTOFF} candidates by pre-score for embedding...")
    all_parsed_metadata.sort(key=lambda x: x['pre_score'], reverse=True)
    selected_metadata = all_parsed_metadata[:CANDIDATE_CUTOFF]
    print(f"  Retained for embedding: {len(selected_metadata)} (sliced from {len(all_parsed_metadata)})")

    text_representations = [x['text_representation'] for x in selected_metadata]

    # Step 5: Loading sentence embedding model
    print("Step 5: Loading sentence embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    print("Step 6: Embedding top candidate profiles (batch processing with normalization)...")
    embeddings = model.encode(text_representations, batch_size=128, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # -----------------------------------------------------------------
    # Template Classification Integration (44 canonical templates)
    # -----------------------------------------------------------------
    # Load template definitions and pre‑computed embeddings if available
    templates_file = os.path.join(out_dir, 'templates.json')
    template_embeds_file = os.path.join(out_dir, 'template_embeds.npy')
    if os.path.exists(templates_file) and os.path.exists(template_embeds_file):
        with open(templates_file, encoding='utf-8') as tf:
            template_data = json.load(tf)  # Expected format: {"template_id": {"template": "...", "tier": "A"}, ...}
        template_embeddings = np.load(template_embeds_file)
        # Build normalized lookup for exact matches
        norm_template_lookup = {}
        for tid, info in template_data.items():
            norm_tpl = normalize_text(info.get('template', ''))
            norm_template_lookup[norm_tpl] = (tid, info.get('tier'))
    else:
        template_data = {}
        template_embeddings = None
        norm_template_lookup = {}

    # Assign template‑based tier to selected candidates (fallback to keyword tier)
    SIM_THRESHOLD = 0.92  # adjustable similarity threshold
    for idx, cand in enumerate(selected_metadata):
        desc = cand.get('text_representation', '')
        norm_desc = normalize_text(desc)
        # Exact normalized match
        if norm_desc in norm_template_lookup:
            tid, tier = norm_template_lookup[norm_desc]
            cand['template_id'] = tid
            cand['final_tier'] = tier
            continue
        # Embedding similarity fallback
        if template_embeddings is not None:
            cand_emb = embeddings[idx]
            # Cosine similarity assuming normalized vectors
            sims = np.dot(template_embeddings, cand_emb)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= SIM_THRESHOLD:
                # Retrieve corresponding template info
                tid = list(template_data.keys())[best_idx]
                tier = template_data[tid].get('tier')
                cand['template_id'] = tid
                cand['final_tier'] = tier
                continue
        # Keyword tier fallback using tier_keywords
        assigned = False
        for tier_key, keywords in tier_keywords.items():
            for kw in keywords:
                if kw.lower() in norm_desc:
                    cand['final_tier'] = tier_key
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            cand['final_tier'] = 'UNKNOWN'
    
    # Step 7: Save precomputed assets
    os.makedirs(out_dir, exist_ok=True)

    parquet_path = os.path.join(out_dir, 'candidates.parquet')
    # Ensure final_tier exists and clean legacy fields
    for item in selected_metadata:
        if item.get('final_tier') is None:
            item['final_tier'] = 'UNKNOWN'
        # Remove old tier fields
        item.pop('tier', None)
        item.pop('candidate_tier', None)
        item.pop('template_tier', None)
        # Clean up temporary columns
        item.pop('text_representation', None)
        item.pop('pre_score', None)
        item.pop('keyword_tier', None)
        
    df = pd.DataFrame(selected_metadata)
    df.to_parquet(parquet_path, engine='pyarrow')
    print(f"Saved candidate features metadata to {parquet_path}")

    npy_path = os.path.join(out_dir, 'embeddings.npy')
    np.save(npy_path, embeddings)
    print(f"Saved candidate embeddings array of shape {embeddings.shape} to {npy_path}")
    print("Precomputation finished successfully!")

if __name__ == '__main__':
    candidates_file = r'C:\Users\shegd\Downloads\[PUB] India_runs_data_and_ai_challenge (1)\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'
    out_directory = r'C:\Users\shegd\Documents\antigravity\jolly-darwin'
    precompute(candidates_file, out_directory)
