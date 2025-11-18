"""
LLM 모델 및 API 설정
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


# =============================================================================
# API Keys
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# =============================================================================
# LLM Model 설정
# =============================================================================

# 기본 모델
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

# 사용 가능한 모델 목록
AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
]


# =============================================================================
# vLLM 설정 (로컬 LLM)
# =============================================================================

VLLM_ENABLED = os.getenv("VLLM_ENABLED", "false").lower() == "true"
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")


# =============================================================================
# Embedding Model 설정
# =============================================================================

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))


# =============================================================================
# 경로 설정
# =============================================================================

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 캐시 디렉토리
CACHE_DIR = PROJECT_ROOT / ".cache"
FILES_DIR = CACHE_DIR / "files"
EMBEDDINGS_DIR = CACHE_DIR / "embeddings"

# 디렉토리 생성
CACHE_DIR.mkdir(exist_ok=True)
FILES_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)


# =============================================================================
# API 설정
# =============================================================================

API_PREFIX = "/api/v1"
API_TITLE = "SM-AI Backend"
API_VERSION = "0.2.0"

# CORS 설정
CORS_ORIGINS = [
    "http://localhost:8501",  # Streamlit
    "http://localhost:3000",  # React
    "http://localhost:8080",
]


# =============================================================================
# Logging 설정
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# =============================================================================
# LangSmith 설정 (Optional)
# =============================================================================

LANGSMITH_ENABLED = os.getenv("LANGSMITH_ENABLED", "false").lower() == "true"
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "sm-ai")

if LANGSMITH_ENABLED and LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
