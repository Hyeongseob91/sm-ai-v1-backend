"""
Graph Factory - 시스템 인스턴스 생성 라우터

사용자 선택에 따라 적절한 시스템(RAG/Chat)을 생성하고 반환하는 Factory 모듈
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# RAG System 생성
# =============================================================================

def create_rag_system(session_id: str, **kwargs):
    """
    RAG System 생성

    Args:
        session_id: 세션 ID
        **kwargs: RAG 설정 인수

    Returns:
        RAGSystemChain 인스턴스

    Example:
        >>> rag = create_rag_system(
        ...     session_id="user123",
        ...     prompt_name="01-pdf-rag",
        ...     use_hybrid_search=True,
        ... )
        >>> await rag.ingest_document("path/to/doc.pdf")
        >>> response = await rag.query("질문")
    """
    from src.systems.rag.rag_system_chain import RAGSystemChain

    rag = RAGSystemChain(session_id=session_id, **kwargs)
    logger.info(f"Created RAG system for session: {session_id}")

    return rag


# =============================================================================
# Chat System (Agentic) 생성
# =============================================================================

def create_chat_system(session_id: str, **kwargs):
    """
    Agentic Chat System 생성

    Args:
        session_id: 세션 ID
        **kwargs: Chat 설정 인수

    Returns:
        ChatSystemChain 인스턴스

    Example:
        >>> chat = create_chat_system(
        ...     session_id="user123",
        ...     persona="01-agentic-rag-default",
        ... )
        >>> response = await chat.chat("분석해줘")
    """
    from src.systems.chat.chat_system_chain import ChatSystemChain

    chat = ChatSystemChain(session_id=session_id, **kwargs)
    logger.info(f"Created Chat system for session: {session_id}")

    return chat


# =============================================================================
# Graph Factory 클래스
# =============================================================================

class GraphFactory:
    """
    통합 Graph/Chain 팩토리

    시스템 타입에 따라 적절한 인스턴스 생성
    """

    @staticmethod
    def create(
        system_type: str,
        session_id: str,
        **kwargs
    ) -> Any:
        """
        시스템 타입에 따른 인스턴스 생성

        Args:
            system_type: 'rag', 'chat'
            session_id: 세션 ID
            **kwargs: 추가 설정

        Returns:
            해당 시스템 인스턴스
        """
        if system_type == "rag":
            return create_rag_system(session_id=session_id, **kwargs)

        elif system_type == "chat":
            return create_chat_system(session_id=session_id, **kwargs)

        else:
            raise ValueError(
                f"Unknown system type: {system_type}. "
                f"Must be one of ['rag', 'chat']"
            )

    @staticmethod
    def get_available_systems() -> list:
        """사용 가능한 시스템 타입 목록"""
        return ["rag", "chat"]
