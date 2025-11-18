"""
API Router - 라우터 등록 및 관리
"""

from fastapi import APIRouter


def create_router() -> APIRouter:
    """
    기본 API 라우터 생성

    Returns:
        APIRouter: 설정된 라우터
    """
    router = APIRouter()
    return router


def get_api_router() -> APIRouter:
    """
    모든 API 라우터를 포함한 라우터 반환

    Returns:
        APIRouter: 통합 라우터
    """
    from .chat_endpoints import router as chat_router
    from .rag_endpoints import router as rag_router

    api_router = APIRouter()

    # 라우터 등록
    api_router.include_router(
        chat_router,
        prefix="/chat",
        tags=["chat"]
    )

    api_router.include_router(
        rag_router,
        prefix="/rag",
        tags=["rag"]
    )

    return api_router
