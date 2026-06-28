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
    MAX_CANDIDATES_RERANK: int = 500
    CHUNK_SIZE_TOKENS: int = 200
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()

# Ensure directories exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
