import logging
import uuid
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.pipeline.ranking_pipeline import RankingPipeline
from src.scoring.rank_memory import RankMemory
from scripts.download_model import download_file, MODEL_URL, MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Candidate Ranking Service",
    description="FastAPI endpoint for candidate evaluation and ranking",
    version="1.0.0"
)

# Global Singletons
pipeline = None
memory = None

@app.on_event("startup")
def startup_event():
    global pipeline, memory
    
    # Ensure model exists before initializing pipeline
    if not os.path.exists(MODEL_PATH):
        logger.info("ONNX model not found. Downloading...")
        download_file(MODEL_URL, MODEL_PATH)
        
    logger.info("Initializing RankingPipeline (loading models into memory)...")
    pipeline = RankingPipeline()
    memory = RankMemory()
    logger.info("Service is ready to accept requests.")

class RankRequest(BaseModel):
    raw_jd: str = Field(..., description="Raw text of the Job Description")
    all_resumes: List[Dict[str, Any]] = Field(..., description="List of candidate parsed resumes as JSON")
    top_k: int = Field(default=2000, description="Max number of candidates to process")

@app.get("/health")
def health_check():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    return {"status": "ok", "message": "Service is fully operational"}

@app.post("/api/v1/rank")
def rank_candidates(request: RankRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
        
    try:
        session_id = uuid.uuid4().hex
        logger.info(f"Starting ranking session {session_id} for {len(request.all_resumes)} resumes.")
        
        # Execute pipeline
        results = pipeline.run(
            raw_jd=request.raw_jd,
            all_resumes=request.all_resumes,
            top_k=request.top_k
        )
        
        # Persist results to SQLite
        if memory:
            memory.log_session(session_id, results)
            
        # Limit output response size
        max_results = getattr(settings, 'MAX_RESULTS_RESPONSE', 100)
        limited_results = results[:max_results]
            
        return {
            "session_id": session_id,
            "count": len(limited_results),
            "total_processed": len(results),
            "results": limited_results
        }
    except Exception as e:
        logger.error(f"Ranking failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
