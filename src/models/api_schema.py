"""
API 요청/응답 스키마 정의
FastAPI 엔드포인트에서 사용하는 Pydantic 모델
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# =============================================================================
# Chat API 스키마
# =============================================================================

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    session_id: str = Field(..., description="세션 식별자")
    message: str = Field(..., description="사용자 메시지")
    model: str = Field(default="gpt-4o", description="사용할 LLM 모델")
    prompt_file: str = Field(..., description="프롬프트 파일 경로")
    task: Optional[str] = Field(default="", description="추가 작업 지시")
    temperature: Optional[float] = Field(default=0.0, description="생성 온도")


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    session_id: str = Field(..., description="세션 식별자")
    message: str = Field(..., description="AI 응답 메시지")
    role: str = Field(default="assistant", description="메시지 역할")


# =============================================================================
# RAG API 스키마
# =============================================================================

class RAGUploadRequest(BaseModel):
    """RAG 문서 업로드 요청 모델"""
    session_id: str = Field(..., description="세션 식별자")
    # 파일은 UploadFile로 별도 처리


class RAGUploadResponse(BaseModel):
    """RAG 문서 업로드 응답 모델"""
    session_id: str
    filename: str
    chunk_count: int
    message: str


class RAGQueryRequest(BaseModel):
    """RAG 질의 요청 모델"""
    session_id: str = Field(..., description="세션 식별자")
    question: str = Field(..., description="질문 내용")
    model: str = Field(default="gpt-4o", description="사용할 LLM 모델")
    prompt_file: str = Field(default="01-pdf-rag", description="프롬프트 파일")
    temperature: Optional[float] = Field(default=0.0, description="생성 온도")

    # RAG 설정
    retrieval_k: Optional[int] = Field(default=10, description="검색할 문서 수")
    final_k: Optional[int] = Field(default=5, description="최종 사용할 문서 수")
    use_hybrid_search: Optional[bool] = Field(default=True, description="하이브리드 검색 사용")
    use_reranking: Optional[bool] = Field(default=True, description="재순위화 사용")


class RAGResponse(BaseModel):
    """RAG 응답 모델"""
    session_id: str
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None


class RAGConfigUpdate(BaseModel):
    """RAG 설정 업데이트 모델"""
    retrieval_k: Optional[int] = None
    final_k: Optional[int] = None
    use_hybrid_search: Optional[bool] = None
    use_reranking: Optional[bool] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


# =============================================================================
# 공통 응답 스키마
# =============================================================================

class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str = "healthy"
    version: str = "0.1.0"


class ErrorResponse(BaseModel):
    """에러 응답"""
    error: str
    detail: Optional[str] = None
