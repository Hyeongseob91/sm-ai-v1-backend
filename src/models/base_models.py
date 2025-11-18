"""
기본 데이터 모델 정의
시스템 전반에서 사용되는 공통 모델
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class MessageRole(str, Enum):
    """메시지 역할"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SystemType(str, Enum):
    """시스템 타입"""
    CHAT = "chat"
    RAG = "rag"
    AGENTIC = "agentic"


# =============================================================================
# Document 모델
# =============================================================================

class Document(BaseModel):
    """문서 모델"""
    page_content: str = Field(..., description="문서 내용")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")

    # 선택적 필드
    source: Optional[str] = Field(default=None, description="문서 출처")
    page: Optional[int] = Field(default=None, description="페이지 번호")

    class Config:
        extra = "allow"


class DocumentChunk(BaseModel):
    """문서 청크 모델"""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: Optional[float] = None


# =============================================================================
# Message 모델
# =============================================================================

class Message(BaseModel):
    """대화 메시지 모델"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Session 모델
# =============================================================================

class SessionState(BaseModel):
    """세션 상태 모델"""
    session_id: str
    system_type: SystemType
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # 대화 기록
    messages: List[Message] = Field(default_factory=list)

    # RAG 관련 상태
    document_loaded: bool = False
    document_name: Optional[str] = None
    chunk_count: int = 0

    # 추가 상태
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_message(self, role: MessageRole, content: str) -> None:
        """메시지 추가"""
        self.messages.append(Message(role=role, content=content))
        self.updated_at = datetime.now()

    def get_history(self, limit: Optional[int] = None) -> List[Message]:
        """대화 기록 조회"""
        if limit:
            return self.messages[-limit:]
        return self.messages

    def clear_history(self) -> None:
        """대화 기록 초기화"""
        self.messages = []
        self.updated_at = datetime.now()


# =============================================================================
# RAG 관련 모델
# =============================================================================

class RetrievalResult(BaseModel):
    """검색 결과 모델"""
    documents: List[DocumentChunk]
    query: str
    total_found: int
    retrieval_time: float  # 초 단위


class RAGContext(BaseModel):
    """RAG 컨텍스트 모델"""
    question: str
    context: str
    sources: List[Dict[str, Any]]

    @classmethod
    def from_documents(cls, question: str, documents: List[DocumentChunk]) -> "RAGContext":
        """문서들로부터 컨텍스트 생성"""
        context_parts = []
        sources = []

        for i, doc in enumerate(documents):
            context_parts.append(f"[문서 {i+1}]\n{doc.content}")
            sources.append({
                "index": i + 1,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", None),
                "score": doc.score,
            })

        return cls(
            question=question,
            context="\n\n".join(context_parts),
            sources=sources
        )
