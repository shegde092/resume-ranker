import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Application Config
    APP_NAME: str = "Resilient AI Candidate Ranking System"
    DEBUG: bool = False
    
    # Latency Budgets (seconds)
    BUDGET_RETRIEVAL: int = 30
    BUDGET_BLOCK_SELECTION: int = 60
    BUDGET_CROSS_ENCODER: int = 120
    BUDGET_FUSION: int = 10
    BUDGET_EXPLANATION: int = 30
    
    # Data Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    DB_PATH: str = os.getenv("DB_PATH", f"{DATA_DIR}/rank_memory.sqlite")
    
    # Model Configs
    MAX_CANDIDATES_RETRIEVAL: int = 2000
    MAX_CANDIDATES_CE: int = 800
    MAX_CANDIDATES_RERANK: int = 500
    CHUNK_SIZE_TOKENS: int = 200
    
    # Phase 1 Fusion Strategy Weights
    WEIGHT_CE: float = 0.40
    WEIGHT_RRF: float = 0.30
    WEIGHT_TRUST: float = 0.20
    WEIGHT_LOGISTICS: float = 0.10
    
    # Target Architecture Configuration
    TOP_K_PRUNE: int = 250
    WEIGHT_CE_SKILLS: float = 0.40
    WEIGHT_CE_CAREER: float = 0.30
    WEIGHT_CE_PROFILE: float = 0.20
    WEIGHT_CE_EDUCATION: float = 0.10
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()

# Ensure directories exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
