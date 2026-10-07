from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM providers -------------------------------------------------
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_REASONING_EFFORT: str = "medium"
    LLM_TIMEOUT_S: float = 60.0
    # USD per 1M tokens - ESTIMATES for the cost line in traces; edit to match your billing.
    GEMINI_PRICE_IN: float = 0.30
    GEMINI_PRICE_OUT: float = 2.50
    GROQ_PRICE_IN: float = 0.15
    GROQ_PRICE_OUT: float = 0.75

    # --- Embeddings (Jina) ----------------------------------------------
    JINA_API_KEY: str = ""
    JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"
    JINA_MODEL: str = "jina-embeddings-v5-omni-small"
    JINA_TASK_DOC: str = "retrieval.passage"    # task used when indexing documents
    JINA_TASK_QUERY: str = "retrieval.query"    # task used for user queries
    JINA_SEND_DIMENSIONS: bool = False          # true -> also send "dimensions": EMBEDDING_DIM (if the model supports truncation)
    JINA_BATCH_SIZE: int = 64
    EMBEDDING_DIM: int = 1024                   # must equal the vector length Jina returns

    # --- Qdrant ---------------------------------------------------------
    QDRANT_URL: str = ""            # empty -> embedded mode at QDRANT_PATH
    QDRANT_API_KEY: str = ""
    QDRANT_PATH: str = "./data/qdrant"   # ":memory:" is also accepted
    QDRANT_COLLECTION: str = "ukvi_policy"

    # --- Retrieval ------------------------------------------------------
    TOP_K: int = 5
    CANDIDATE_K: int = 30
    RERANK_ENABLED: bool = False
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    MIN_RERANK_SCORE: float = 0.30   # abstain threshold when reranker is on
    MIN_DENSE_SCORE: float = 0.45    # abstain threshold (cosine) when reranker is off

    # --- Graph ----------------------------------------------------------
    CLASSIFY_MIN_CONFIDENCE: float = 0.60
    FUZZY_THRESHOLD: int = 85
    FUZZY_AMBIGUITY_GAP: int = 5
    MAX_REGENERATIONS: int = 2

    # --- Storage / logs -------------------------------------------------
    DB_PATH: str = "./data/emigrant.db"
    LOG_DIR: str = "./logs"
    RAW_DIR: str = "./data/raw"

    # --- Eval -----------------------------------------------------------
    RAGAS_JUDGE: str = "groq"        # judge != generator to limit self-preference bias


@lru_cache
def get_settings() -> Settings:
    return Settings()
