"""
데이터베이스 및 벡터 스토어 설정
"""

import os
from pathlib import Path


# =============================================================================
# Weaviate 설정
# =============================================================================

WEAVIATE_ENABLED = os.getenv("WEAVIATE_ENABLED", "false").lower() == "true"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
WEAVIATE_INDEX_NAME = os.getenv("WEAVIATE_INDEX_NAME", "SoundmindDocuments")


# =============================================================================
# FAISS 설정
# =============================================================================

_BASE_DIR = Path(__file__).parent.parent.parent

# FAISS 인덱스 저장 경로
FAISS_INDEX_DIR = _BASE_DIR / ".cache" / "faiss_indexes"
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ChromaDB 설정
# =============================================================================

CHROMA_ENABLED = os.getenv("CHROMA_ENABLED", "false").lower() == "true"
CHROMA_PERSIST_DIR = _BASE_DIR / ".cache" / "chroma"
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Redis 설정 (세션 캐시, Optional)
# =============================================================================

REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))


# =============================================================================
# SQLite 설정 (메타데이터 저장, Optional)
# =============================================================================

SQLITE_ENABLED = os.getenv("SQLITE_ENABLED", "false").lower() == "true"
SQLITE_DB_PATH = _BASE_DIR / ".cache" / "metadata.db"


# =============================================================================
# 벡터 DB 타입 선택
# =============================================================================

def get_vector_store_type() -> str:
    """
    활성화된 벡터 스토어 타입 반환

    Returns:
        'faiss', 'weaviate', 'chroma'
    """
    if WEAVIATE_ENABLED:
        return "weaviate"
    elif CHROMA_ENABLED:
        return "chroma"
    else:
        return "faiss"  # 기본값


def get_vector_store_config() -> dict:
    """
    현재 활성화된 벡터 스토어 설정 반환

    Returns:
        벡터 스토어 설정 딕셔너리
    """
    store_type = get_vector_store_type()

    if store_type == "weaviate":
        return {
            "type": "weaviate",
            "url": WEAVIATE_URL,
            "api_key": WEAVIATE_API_KEY,
            "index_name": WEAVIATE_INDEX_NAME,
        }
    elif store_type == "chroma":
        return {
            "type": "chroma",
            "persist_directory": str(CHROMA_PERSIST_DIR),
        }
    else:
        return {
            "type": "faiss",
            "index_directory": str(FAISS_INDEX_DIR),
        }
