import json
import datetime
import os

def curate():
    candidates_path = r'C:\Users\shegd\Downloads\[PUB] India_runs_data_and_ai_challenge (1)\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'
    out_json = r'C:\Users\shegd\Documents\antigravity\jolly-darwin\gold_set_labels.json'
    
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

    tier_keywords = {
        'A': ['ranking', 'search', 'retrieval', 'recommendation', 'ltr', 'information retrieval', 'dense retrieval', 'bm25', 'faiss', 'pinecone', 'vector database', 'hybrid search', 'vector search'],
        'B': ['machine learning', 'nlp', 'llm', 'deep learning', 'transformers', 'fine-tuning', 'rag', 'pytorch', 'tensorflow', 'keras', 'huggingface', 'weights & biases'],
        'C': ['computer vision', 'image classification', 'object detection', 'speech', 'tts', 'asr', 'ocr', 'gan', 'diffusion', 'opencv'],
        'D': ['data engineer', 'backend', 'analytics engineer', 'spark', 'airflow', 'kafka', 'snowflake', 'databricks', 'hadoop', 'pipeline']
    }

    must_have_skills = ['vector search', 'python', 'embeddings', 'sentence-transformers', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss', 'ndcg', 'mrr', 'map', 'ranking evaluation']

    gold_set = {}
    
    tier_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'Honeypot': 0}
    
    with open(candidates_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            cid = c.get('candidate_id', '')
            profile = c.get('profile', {})
            history = c.get('career_history', [])
            skills = c.get('skills', [])
            education = c.get('education', [])
            signals = c.get('redrob_signals', {})
            
            curr_title = profile.get('current_title', '')
            curr_title_lower = curr_title.lower()
            years_exp = profile.get('years_of_experience', 0.0)

            # Check Honeypot score to see if it is a honeypot
            h1 = 0
            for job in history:
                comp = job.get('company', '')
                if comp in founding_years:
                    try:
                        start_yr = int(job.get('start_date', '')[:4])
                        if start_yr < founding_years[comp]:
                            h1 = 1
                            break
                    except:
                        pass
            
            # Simple check for honeypots (H1 or overlapping timeline or grad anomaly)
            h3 = 0
            bachelors_end = None
            masters_end = None
            for edu in education:
                deg = edu.get('degree', '').replace('.', '').replace(' ', '').lower()
                end_yr = edu.get('end_year')
                if not end_yr:
                    continue
                if any(b in deg for b in ['btech', 'be', 'bachelor', 'bs', 'bsc', 'ba', 'bca', 'bcom']):
                    if bachelors_end is None or end_yr > bachelors_end:
                        bachelors_end = end_yr
                elif any(m in deg for m in ['mtech', 'me', 'master', 'ms', 'msc', 'mba', 'phd']):
                    if masters_end is None or end_yr < masters_end:
                        masters_end = end_yr
            if bachelors_end and masters_end and masters_end < bachelors_end:
                h3 = 1

            is_honeypot = (h1 == 1 or h3 == 1)

            if is_honeypot:
                if tier_counts['Honeypot'] < 15:
                    gold_set[cid] = 0
                    tier_counts['Honeypot'] += 1
                continue

            # Classify tier
            candidate_tier = None
            for tier_letter in ['A', 'B', 'C', 'D']:
                kws = tier_keywords[tier_letter]
                has_title_kw = any(any(kw in title.lower() for kw in kws) for title in [curr_title] + [job.get('title', '') for job in history])
                has_desc_kw = any(any(kw in text.lower() for kw in kws) for text in [profile.get('summary', '')] + [job.get('description', '') for job in history])
                tier_score = 0.2 * (1.0 if has_title_kw else 0.0) + 0.8 * (1.0 if has_desc_kw else 0.0)
                if tier_score >= 0.25:
                    candidate_tier = tier_letter
                    break

            if not candidate_tier:
                continue

            # Limit counts per tier
            if candidate_tier == 'A' and tier_counts['A'] < 20:
                pass
            elif candidate_tier == 'B' and tier_counts['B'] < 20:
                pass
            elif candidate_tier == 'C' and tier_counts['C'] < 8:
                pass
            elif candidate_tier == 'D' and tier_counts['D'] < 7:
                pass
            else:
                continue

            # Assign relevance score (3: perfect, 2: good, 1: borderline, 0: not fit)
            relevance = 1
            if candidate_tier == 'A':
                relevance = 3
            elif candidate_tier == 'B':
                relevance = 2

            # Experience fit adjustment
            if years_exp < 5.0 or years_exp > 9.0:
                relevance = max(1, relevance - 1)

            # Must-have skills check
            candidate_skill_names = [s.get('name', '').lower() for s in skills]
            has_must_haves = sum(1 for sk in must_have_skills if sk in candidate_skill_names)
            if has_must_haves < 2:
                relevance = max(1, relevance - 1)

            # Hard filter category
            if any(role in curr_title_lower for role in ['accountant', 'hr manager', 'sales executive', 'marketing manager']):
                relevance = 0

            gold_set[cid] = relevance
            tier_counts[candidate_tier] += 1

            if sum(tier_counts.values()) >= 70:
                break

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(gold_set, f, indent=2)
        
    print(f"Curated gold set saved to {out_json}")
    print(f"Distribution: {tier_counts}")

if __name__ == '__main__':
    curate()
