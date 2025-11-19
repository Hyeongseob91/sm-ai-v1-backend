"""
LLM 모델 및 API 설정
"""

import os
from pathlib import Path
from enum import Enum
from typing import Dict, Any
from dotenv import load_dotenv


# .env 파일 로드
load_dotenv()


# =============================================================================
# LLM 공급자 열거형
# =============================================================================
class LLMProvider(Enum):
    """
    LLM 공급자 열거형

    지원하는 LLM 공급자를 정의합니다.
    MODEL_CONFIG에서 각 모델의 공급자를 지정할 때 사용됩니다.
    """
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    VLLM = "vllm"


# =============================================================================
# API Keys
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


# =============================================================================
# API Model 설정
# =============================================================================

# 1. 기본값 설정
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

# 2. 사용자 Selectbox에 표시할 모델 목록
AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5.1",
    # "claude-3-5-sonnet-20241022",
    # "claude-3-opus-20240229",
    # "gemini-2.5-pro",
    # "gemini-2.5-flash",
]

# 3. 사용자 모델 선택시 Routing 처리 공급자 매핑
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    # OpenAI Models
    "gpt-4o": {"provider": LLMProvider.OPENAI},
    "gpt-4o-mini": {"provider": LLMProvider.OPENAI},
    "gpt-4-turbo": {"provider": LLMProvider.OPENAI},
    "gpt-4.1": {"provider": LLMProvider.OPENAI},
    "gpt-5": {"provider": LLMProvider.OPENAI},
    "gpt-5-mini": {"provider": LLMProvider.OPENAI},
    "gpt-5.1": {"provider": LLMProvider.OPENAI},

    # Anthropic Models
    "claude-3-5-sonnet-20241022": {"provider": LLMProvider.ANTHROPIC},
    "claude-3-opus-20240229": {"provider": LLMProvider.ANTHROPIC},

    # Google Models
    "gemini-2.5-pro": {"provider": LLMProvider.GOOGLE},
    "gemini-2.5-flash": {"provider": LLMProvider.GOOGLE},

    # vLLM Models
    "vllm:gemma-32b": {"provider": LLMProvider.VLLM},
    "vllm:qwen3-32b": {"provider": LLMProvider.VLLM},
}


# =============================================================================
# Local Model 설정 (vLLM)
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