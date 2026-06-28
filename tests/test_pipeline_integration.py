import pytest
from src.pipeline.ranking_pipeline import RankingPipeline

@pytest.fixture(scope="module")
def pipeline():
    return RankingPipeline()

@pytest.fixture
def sample_resumes():
    return [
        {
            "candidate_id": "CAND_001",
            "career_history": [{"title": "Software Engineer", "description": "Python, SQL"}],
            "skills": ["Python", "SQL", "AWS"],
            "profile": {"headline": "Senior Developer", "years_of_experience": "5"},
            "education": [{"degree": "B.E Computer Science"}],
            "trust_score": 0.9
        },
        {
            "candidate_id": "CAND_002",
            "career_history": [{"title": "Data Scientist", "description": "Machine Learning, PyTorch"}],
            "skills": ["PyTorch", "Python", "ML"],
            "profile": {"headline": "Data Scientist", "years_of_experience": "3"},
            "education": [{"degree": "Ph.D Computer Science"}],
            "trust_score": 0.8
        },
        {
            "candidate_id": "CAND_003",
            "career_history": [{"title": "Frontend Developer", "description": "React, CSS"}],
            "skills": ["React", "CSS", "JavaScript"],
            "profile": {"headline": "UI Developer", "years_of_experience": "2"},
            "education": [{"degree": "B.Sc"}],
            "trust_score": 0.3  # Low trust to test penalty
        }
    ]

@pytest.fixture
def sample_jd():
    return "Looking for a Python Software Engineer with AWS experience and 5+ years of experience."

def test_jd_parsing(pipeline, sample_jd):
    """Test 1: JD parsing returns valid structured output"""
    parsed = pipeline.jd_parser.parse(sample_jd)
    assert isinstance(parsed, dict)
    assert "query" in parsed

def test_retrieval(pipeline, sample_jd, sample_resumes):
    """Test 2: Retrieval returns non-empty candidate shortlist"""
    query = sample_jd
    pipeline._build_indices(sample_resumes)
    dense = pipeline.dense_retriever.search(query, top_k=10)
    bm25 = pipeline.bm25_retriever.search(query, top_k=10)
    fused = pipeline.fusion.fuse(dense, bm25)
    
    assert len(fused) > 0, "Retrieval should return a non-empty shortlist"
    assert "candidate_id" in fused[0]

def test_cross_encoder_reranking(pipeline, sample_jd, sample_resumes):
    """Test 3: Cross encoder reranking executes successfully"""
    query = sample_jd
    for cand in sample_resumes:
        cand.update(pipeline.block_selector.select_blocks(cand, query))
    
    reranked = pipeline.cross_encoder.rerank(query, sample_resumes)
    assert len(reranked) > 0, "Reranking should return a non-empty list"
    assert "ce_score_avg" in reranked[0], "Cross encoder should append ce_score_avg"

def test_adaptive_fusion(pipeline, sample_resumes):
    """Test 4: Adaptive fusion returns sorted scores"""
    # Mock CE and retrieval features for isolated fusion test
    for i, c in enumerate(sample_resumes):
        c["score"] = 0.05
        c["career_fit_ce"] = 0.9
        c["skill_fit_ce"] = 0.8
        c["profile_fit_ce"] = 0.7
        c["education_fit_ce"] = 0.6
        c["ce_score_avg"] = 0.75
        c["trust_score"] = 0.9 if i != 2 else 0.2
        c["logistics_score"] = 1.0
        
    ranked = pipeline.adaptive_ranker.rank(sample_resumes)
    assert len(ranked) == 3
    assert ranked[0]["final_score"] >= ranked[-1]["final_score"], "Scores should be sorted descending"

def test_explanation_engine(pipeline):
    """Test 5: Explanation engine attaches reason and confidence"""
    cand = {
        "candidate_id": "CAND_TEST",
        "final_score": 0.91,
        "ce_score_avg": 0.85,
        "trust_score": 0.9,
        "years_of_experience": "5"
    }
    
    explained = pipeline.reason_generator.generate(cand)
    assert explained["candidate_id"] == "CAND_TEST"
    assert explained["final_score"] == 0.91
    assert explained["confidence"] in ["high", "medium", "low"]
    assert "reason" in explained

def test_full_pipeline_orchestration(pipeline, sample_jd, sample_resumes):
    """Master Pipeline Integration Test"""
    results = pipeline.run(sample_jd, sample_resumes)
    
    assert len(results) > 0, "Pipeline must return non-empty results"
    assert results[0]["final_score"] >= results[-1]["final_score"], "Results must be sorted by descending final_score"
    
    for r in results:
        assert "candidate_id" in r
        assert "final_score" in r
        assert "confidence" in r
        assert "reason" in r
