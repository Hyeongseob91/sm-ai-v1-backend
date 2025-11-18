"""
RAG API Endpoints
RAG 시스템 관련 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import AsyncIterator, Dict, Any, Optional
import logging
import os
import uuid
from pathlib import Path

from src.models.api_schema import (
    RAGQueryRequest,
    RAGResponse,
    RAGConfigUpdate,
    RAGUploadResponse,
)
from src.core.graph_factory import (
    create_rag_system,
    get_available_prompts,
)
from src.config.config_model import FILES_DIR

logger = logging.getLogger(__name__)

router = APIRouter()

# 세션별 RAG 인스턴스 저장
rag_store: Dict[str, Any] = {}


# =============================================================================
# Document Upload Endpoints
# =============================================================================

@router.post("/upload", response_model=RAGUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    문서 업로드 및 인덱싱 생성

    PDF, TXT, MD 파일 지원
    """
    try:
        # 파일 타입 확인
        filename = file.filename
        ext = Path(filename).suffix.lower()

        if ext not in [".pdf", ".txt", ".md"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Supported: .pdf, .txt, .md"
            )

        # 파일 저장
        file_id = uuid.uuid4().hex[:8]
        save_path = FILES_DIR / f"{session_id}_{file_id}{ext}"

        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"File saved: {save_path}")

        # RAG 시스템 생성
        rag = create_rag_system(session_id=session_id)
        rag_store[session_id] = rag

        # 문서 인덱싱
        result = await rag.ingest_document(str(save_path))

        return RAGUploadResponse(
            session_id=session_id,
            filename=filename,
            chunk_count=result.get("chunk_count", 0),
            message=f"Successfully processed {filename}"
        )

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Query Endpoints
# =============================================================================

@router.post("/query")
async def rag_query(request: RAGQueryRequest):
    """
    RAG 질의응답 (스트리밍)
    """
    session_id = request.session_id

    # RAG 인스턴스 확인
    if session_id not in rag_store:
        raise HTTPException(
            status_code=404,
            detail=f"No document uploaded for session: {session_id}"
        )

    rag = rag_store[session_id]

    try:
        # 설정 업데이트
        if request.retrieval_k or request.final_k:
            await rag.update_config(
                retrieval_k=request.retrieval_k,
                final_k=request.final_k,
                use_hybrid_search=request.use_hybrid_search,
                use_reranking=request.use_reranking
            )

        # 스트리밍 응답
        async def generate() -> AsyncIterator[str]:
            try:
                async for chunk in rag.query_stream(request.question):
                    if chunk:
                        yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Query stream error: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/sync", response_model=RAGResponse)
async def rag_query_sync(request: RAGQueryRequest):
    """
    RAG 질의응답 (동기)
    """
    session_id = request.session_id

    if session_id not in rag_store:
        raise HTTPException(
            status_code=404,
            detail=f"No document uploaded for session: {session_id}"
        )

    rag = rag_store[session_id]

    try:
        result = await rag.query(request.question)

        return RAGResponse(
            session_id=session_id,
            answer=result.answer,
            sources=result.sources
        )

    except Exception as e:
        logger.error(f"RAG sync query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Config Endpoints
# =============================================================================

@router.put("/config/{session_id}")
async def update_rag_config(session_id: str, config: RAGConfigUpdate):
    """
    RAG 설정 업데이트
    """
    if session_id not in rag_store:
        raise HTTPException(
            status_code=404,
            detail=f"No RAG instance for session: {session_id}"
        )

    rag = rag_store[session_id]

    try:
        await rag.update_config(**config.dict(exclude_none=True))
        return {"message": "Config updated", "session_id": session_id}
    except Exception as e:
        logger.error(f"Config update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Prompt Endpoints
# =============================================================================

@router.get("/prompts")
async def get_rag_prompts():
    """
    사용 가능한 RAG 프롬프트 목록
    """
    try:
        prompts = get_available_prompts("rag")
        return {
            "prompts": prompts,
            "count": len(prompts)
        }
    except Exception as e:
        logger.error(f"Get RAG prompts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Session Endpoints
# =============================================================================

@router.get("/session/{session_id}/document")
async def get_session_document(session_id: str):
    """
    세션의 문서 정보 조회
    """
    if session_id not in rag_store:
        return {
            "session_id": session_id,
            "document_loaded": False,
            "message": "No document uploaded"
        }

    rag = rag_store[session_id]

    return {
        "session_id": session_id,
        "document_loaded": True,
        "document_name": getattr(rag, 'document_name', None),
        "chunk_count": getattr(rag, 'chunk_count', 0),
    }


@router.delete("/session/{session_id}")
async def delete_rag_session(session_id: str):
    """
    RAG 세션 삭제
    """
    if session_id in rag_store:
        rag = rag_store[session_id]

        # 세션 정리
        if hasattr(rag, 'clear'):
            await rag.clear()

        del rag_store[session_id]

        return {"message": f"RAG session {session_id} deleted", "success": True}
    else:
        return {"message": f"Session {session_id} not found", "success": False}


# =============================================================================
# Stats Endpoints
# =============================================================================

@router.get("/stats")
async def get_rag_stats():
    """
    RAG 시스템 통계
    """
    return {
        "active_sessions": len(rag_store),
        "sessions": list(rag_store.keys())
    }
