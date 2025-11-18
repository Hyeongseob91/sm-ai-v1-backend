"""
Chat API Endpoints
대화 관련 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncIterator
import logging

from src.models.api_schema import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)
from src.core.graph_factory import (
    create_chatbot_chain,
    get_available_prompts,
    get_prompt_info,
)
from src.core.session_manager import (
    clear_session,
    session_exists,
    get_session_info,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Chat Endpoints
# =============================================================================

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    스트리밍 대화 응답

    SSE (Server-Sent Events) 방식으로 응답 스트리밍
    """
    try:
        # Chain 생성
        chain = create_chatbot_chain(
            prompt_file=request.prompt_file,
            model=request.model,
            task=request.task or "",
            temperature=request.temperature or 0.0
        )

        # 스트리밍 응답 생성
        async def generate() -> AsyncIterator[str]:
            try:
                async for chunk in chain.astream(
                    {"question": request.message},
                    config={"configurable": {"session_id": request.session_id}}
                ):
                    if chunk:
                        yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Prompt file not found: {e}")
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    """
    일반 대화 응답 (non-streaming)
    """
    try:
        # Chain 생성
        chain = create_chatbot_chain(
            prompt_file=request.prompt_file,
            model=request.model,
            task=request.task or "",
            temperature=request.temperature or 0.0
        )

        # 응답 생성
        response = await chain.ainvoke(
            {"question": request.message},
            config={"configurable": {"session_id": request.session_id}}
        )

        return ChatResponse(
            session_id=request.session_id,
            message=response,
            role="assistant"
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Prompt file not found: {e}")
    except Exception as e:
        logger.error(f"Chat message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Prompt Endpoints
# =============================================================================

@router.get("/prompts")
async def get_chat_prompts():
    """
    사용 가능한 대화 프롬프트 목록
    """
    try:
        prompts = get_available_prompts("chatbot")
        return {
            "prompts": prompts,
            "count": len(prompts)
        }
    except Exception as e:
        logger.error(f"Get prompts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prompts/{prompt_name}")
async def get_chat_prompt_info(prompt_name: str):
    """
    특정 프롬프트 정보 조회
    """
    try:
        info = get_prompt_info("chatbot", prompt_name)
        return info
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_name}")
    except Exception as e:
        logger.error(f"Get prompt info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Session Endpoints
# =============================================================================

@router.delete("/session/{session_id}")
async def delete_chat_session(session_id: str):
    """
    대화 세션 초기화
    """
    success = clear_session(session_id)
    if success:
        return {"message": f"Session {session_id} cleared", "success": True}
    else:
        return {"message": f"Session {session_id} not found", "success": False}


@router.get("/session/{session_id}/exists")
async def check_session_exists(session_id: str):
    """
    세션 존재 여부 확인
    """
    exists = session_exists(session_id)
    return {"session_id": session_id, "exists": exists}


@router.get("/session/{session_id}")
async def get_session_details(session_id: str):
    """
    세션 상세 정보 조회
    """
    info = get_session_info(session_id)
    if info:
        return info
    else:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
