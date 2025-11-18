"""
SM-AI Backend - FastAPI 애플리케이션 진입점

Uvicorn으로 실행:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.config_model import (
    API_TITLE,
    API_VERSION,
    API_PREFIX,
    CORS_ORIGINS,
    LOG_LEVEL,
    LOG_FORMAT,
)
from src.api.router import get_api_router
from src.models.api_schema import HealthResponse


# =============================================================================
# Logging 설정
# =============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)

logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan (시작/종료 이벤트)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 이벤트"""
    # Startup
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")

    # 도구 초기화
    try:
        from src.systems.chat.tools.tool_registry import initialize_default_tools
        initialize_default_tools()
        logger.info("Initialized default tools")
    except Exception as e:
        logger.warning(f"Failed to initialize tools: {e}")

    yield

    # Shutdown
    logger.info("Shutting down application")


# =============================================================================
# FastAPI 앱 생성
# =============================================================================

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="SM-AI Backend API - LLM 기반 대화 및 RAG 시스템",
    lifespan=lifespan,
)


# =============================================================================
# CORS 미들웨어
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 라우터 등록
# =============================================================================

# API v1 라우터
api_router = get_api_router()
app.include_router(api_router, prefix=API_PREFIX)


# =============================================================================
# Root Endpoints
# =============================================================================

@app.get("/", tags=["root"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": f"Welcome to {API_TITLE}",
        "version": API_VERSION,
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """상태 체크"""
    return HealthResponse(
        status="healthy",
        version=API_VERSION
    )


@app.get("/info", tags=["info"])
async def app_info():
    """애플리케이션 정보"""
    from src.config.config_model import (
        DEFAULT_MODEL,
        VLLM_ENABLED,
        EMBEDDING_MODEL,
    )
    from src.config.config_db import get_vector_store_type

    return {
        "title": API_TITLE,
        "version": API_VERSION,
        "default_model": DEFAULT_MODEL,
        "vllm_enabled": VLLM_ENABLED,
        "embedding_model": EMBEDDING_MODEL,
        "vector_store": get_vector_store_type(),
    }


# =============================================================================
# 직접 실행
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
