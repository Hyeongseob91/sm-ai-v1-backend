# Config 모듈
# 애플리케이션 전역 설정

from .config_model import (
    OPENAI_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    AVAILABLE_MODELS,
    VLLM_BASE_URL,
    VLLM_MODEL,
    EMBEDDING_MODEL,
)
from .config_prompts import (
    PROMPTS_DIR,
    CHATBOT_PROMPTS_DIR,
    RAG_PROMPTS_DIR,
    AGENTIC_PROMPTS_DIR,
)
from .config_db import (
    WEAVIATE_URL,
    WEAVIATE_INDEX_NAME,
    FAISS_INDEX_DIR,
)

__all__ = [
    # Model config
    "OPENAI_API_KEY",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "AVAILABLE_MODELS",
    "VLLM_BASE_URL",
    "VLLM_MODEL",
    "EMBEDDING_MODEL",
    # Prompts config
    "PROMPTS_DIR",
    "CHATBOT_PROMPTS_DIR",
    "RAG_PROMPTS_DIR",
    "AGENTIC_PROMPTS_DIR",
    # DB config
    "WEAVIATE_URL",
    "WEAVIATE_INDEX_NAME",
    "FAISS_INDEX_DIR",
]
