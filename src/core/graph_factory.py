"""
Graph Factory - LangGraph/LangChain 체인 생성

사용자 선택에 따라 적절한 시스템(Chatbot/RAG/Agentic)을 생성하고 반환
"""

import yaml
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory

from src.core.session_manager import get_session_history
from src.config.config_model import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from src.core.prompts_service import PromptsService

logger = logging.getLogger(__name__)


# =============================================================================
# Chatbot Chain 생성
# =============================================================================

def create_chatbot_chain(
    prompt_file: str,
    model: str = DEFAULT_MODEL,
    task: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
):
    """
    Chatbot Chain 생성

    Args:
        prompt_file: 프롬프트 파일 경로
        model: LLM 모델명
        task: 역할 설정 (선택사항)
        temperature: 생성 온도

    Returns:
        RunnableWithMessageHistory: 대화 히스토리를 포함한 체인
    """
    # YAML 파일에서 프롬프트 로드
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_data = yaml.safe_load(f)

    # 템플릿 추출
    template_text = prompt_data.get("template", "")

    # 시스템 메시지 구성
    if "#Question:" in template_text:
        base_prompt = template_text.split("#Question:")[0].strip()
    else:
        base_prompt = template_text.replace("{question}", "").strip()

    # Task 역할 추가
    if task and task.strip():
        system_message = (
            f"{base_prompt}\n\n"
            f"system role: {task}\n"
            f"위 역할에 맞게 전문적으로 대답해주세요."
        )
    else:
        system_message = base_prompt

    # 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # LLM 생성
    from src.core.llm_service import create_llm_router
    llm = create_llm_router(model=model, temperature=temperature)

    # Chain 구성
    chain = prompt | llm | StrOutputParser()

    # 히스토리 포함 Chain
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    logger.info(f"Created chatbot chain with model={model}")
    return chain_with_history


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
            system_type: 'chatbot', 'rag', 'chat'
            session_id: 세션 ID
            **kwargs: 추가 설정

        Returns:
            해당 시스템 인스턴스
        """
        if system_type == "chatbot":
            prompt_file = kwargs.pop("prompt_file", None)
            if not prompt_file:
                raise ValueError("prompt_file is required for chatbot")
            return create_chatbot_chain(prompt_file=prompt_file, **kwargs)

        elif system_type == "rag":
            return create_rag_system(session_id=session_id, **kwargs)

        elif system_type == "chat":
            return create_chat_system(session_id=session_id, **kwargs)

        else:
            raise ValueError(
                f"Unknown system type: {system_type}. "
                f"Must be one of ['chatbot', 'rag', 'chat']"
            )

    @staticmethod
    def get_available_systems() -> list:
        """사용 가능한 시스템 타입 목록"""
        return ["chatbot", "rag", "chat"]
