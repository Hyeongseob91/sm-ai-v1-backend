"""
Session Manager - 세션 및 대화 기록 관리
"""

from typing import Dict, Optional, List
from datetime import datetime
import logging

from langchain_community.chat_message_histories import ChatMessageHistory

logger = logging.getLogger(__name__)


# =============================================================================
# 세션 저장소 (메모리 기반)
# =============================================================================

session_store: Dict[str, ChatMessageHistory] = {}
session_metadata: Dict[str, dict] = {}


# =============================================================================
# 세션 관리 함수
# =============================================================================

def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    세션 ID로 대화 히스토리 가져오기

    Args:
        session_id: 세션 ID

    Returns:
        ChatMessageHistory: 대화 히스토리 객체
    """
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
        session_metadata[session_id] = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0
        }
        logger.debug(f"Created new session: {session_id}")

    return session_store[session_id]


def clear_session(session_id: str) -> bool:
    """
    세션 초기화

    Args:
        session_id: 세션 ID

    Returns:
        bool: 성공 여부
    """
    if session_id in session_store:
        session_store[session_id].clear()

        if session_id in session_metadata:
            session_metadata[session_id]["updated_at"] = datetime.now().isoformat()
            session_metadata[session_id]["message_count"] = 0

        logger.info(f"Cleared session: {session_id}")
        return True

    return False


def delete_session(session_id: str) -> bool:
    """
    세션 완전 삭제

    Args:
        session_id: 세션 ID

    Returns:
        bool: 성공 여부
    """
    deleted = False

    if session_id in session_store:
        del session_store[session_id]
        deleted = True

    if session_id in session_metadata:
        del session_metadata[session_id]
        deleted = True

    if deleted:
        logger.info(f"Deleted session: {session_id}")

    return deleted


def get_all_sessions() -> List[str]:
    """
    모든 세션 ID 목록 반환

    Returns:
        list: 세션 ID 리스트
    """
    return list(session_store.keys())


def session_exists(session_id: str) -> bool:
    """
    세션 존재 여부 확인

    Args:
        session_id: 세션 ID

    Returns:
        bool: 존재 여부
    """
    return session_id in session_store


def get_session_info(session_id: str) -> Optional[dict]:
    """
    세션 정보 조회

    Args:
        session_id: 세션 ID

    Returns:
        세션 메타데이터 또는 None
    """
    if session_id not in session_store:
        return None

    history = session_store[session_id]
    metadata = session_metadata.get(session_id, {})

    return {
        "session_id": session_id,
        "message_count": len(history.messages),
        "created_at": metadata.get("created_at"),
        "updated_at": metadata.get("updated_at"),
    }


def get_session_messages(
    session_id: str,
    limit: Optional[int] = None
) -> List[dict]:
    """
    세션의 메시지 목록 조회

    Args:
        session_id: 세션 ID
        limit: 최대 메시지 수 (None이면 전체)

    Returns:
        메시지 딕셔너리 리스트
    """
    if session_id not in session_store:
        return []

    history = session_store[session_id]
    messages = history.messages

    if limit:
        messages = messages[-limit:]

    return [
        {
            "role": msg.type,
            "content": msg.content
        }
        for msg in messages
    ]


def update_session_metadata(session_id: str, **kwargs) -> None:
    """
    세션 메타데이터 업데이트

    Args:
        session_id: 세션 ID
        **kwargs: 업데이트할 메타데이터
    """
    if session_id not in session_metadata:
        session_metadata[session_id] = {}

    session_metadata[session_id].update(kwargs)
    session_metadata[session_id]["updated_at"] = datetime.now().isoformat()
