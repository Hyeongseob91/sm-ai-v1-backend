"""
Chat System Chain
Agentic AI 기반 도구 사용 대화 시스템
"""

from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import logging

from src.core.llm_service import LLMService
from src.core.session_manager import get_session_history
from src.config.config_model import DEFAULT_MODEL, DEFAULT_TEMPERATURE

from .constants import (
    MAX_ITERATIONS,
    AGENT_SYSTEM_PROMPT,
    DEFAULT_TOOLS,
)
from .models.function_call_model import (
    AgentAction,
    ActionType,
    FunctionCallResult,
    AgentState,
)
from .agents import AgentPlanner, ToolExecutor
from .tools import ToolRegistry, initialize_default_tools
from .prompts import PersonaLoader

logger = logging.getLogger(__name__)


@dataclass
class ChatConfig:
    """Chat 시스템 설정"""
    # LLM 설정
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE

    # Agent 설정
    max_iterations: int = MAX_ITERATIONS
    use_planning: bool = False  # Plan-then-Execute vs ReAct

    # 도구 설정
    tools: List[str] = None  # None이면 기본 도구 사용

    # 페르소나
    persona: Optional[str] = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = DEFAULT_TOOLS.copy()


class ChatSystemChain:
    """
    Agentic Chat 시스템 메인

    LLM과 Planner로, Tools를 Executor로 사용하는 대화 시스템
    """

    def __init__(
        self,
        session_id: str,
        config: Optional[ChatConfig] = None,
        **kwargs
    ):
        self.session_id = session_id
        self.config = config or ChatConfig(**kwargs)

        # 컴포넌트 초기화
        self._llm_service = None
        self._planner = AgentPlanner()
        self._executor = ToolExecutor()

        # 도구 초기화
        self._initialize_tools()

        # 상태
        self._iteration_count = 0
        self._conversation_history: List[Dict[str, str]] = []

    def _initialize_tools(self) -> None:
        """도구 초기화"""
        # 기본 도구가 등록되어 있지 않으면 초기화
        if not ToolRegistry.get_names():
            initialize_default_tools()

        # 설정된 도구만 활성화
        self._planner.set_available_tools(self.config.tools)

        logger.debug(f"Initialized tools: {self.config.tools}")

    async def chat(self, message: str) -> str:
        """
        대화 메시지 처리 (동기)

        Args:
            message: 사용자 메시지

        Returns:
            AI 응답
        """
        # LLM 초기화
        if self._llm_service is None:
            self._llm_service = LLMService(
                model=self.config.model,
                temperature=self.config.temperature
            )
            self._planner.set_llm(self._llm_service.get_llm())

        # 대화 기록에 추가
        self._conversation_history.append({
            "role": "user",
            "content": message
        })

        # 시스템 프롬프트 구성
        system_prompt = self._build_system_prompt()

        # ReAct 루프 실행
        response = await self._react_loop(message, system_prompt)

        # 응답을 대화 기록에 추가
        self._conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    async def chat_stream(self, message: str) -> AsyncIterator[str]:
        """
        대화 메시지 처리 (스트리밍)

        Args:
            message: 사용자 메시지

        Yields:
            응답 청크
        """
        # LLM 초기화
        if self._llm_service is None:
            self._llm_service = LLMService(
                model=self.config.model,
                temperature=self.config.temperature,
                streaming=True
            )
            self._planner.set_llm(self._llm_service.get_llm())

        # 대화 기록에 추가
        self._conversation_history.append({
            "role": "user",
            "content": message
        })

        # 시스템 프롬프트 구성
        system_prompt = self._build_system_prompt()

        # 스트리밍 응답
        full_response = ""
        async for chunk in self._react_loop_stream(message, system_prompt):
            full_response += chunk
            yield chunk

        # 응답을 대화 기록에 추가
        self._conversation_history.append({
            "role": "assistant",
            "content": full_response
        })

    async def _react_loop(
        self,
        message: str,
        system_prompt: str
    ) -> str:
        """
        ReAct (Reasoning + Acting) 루프

        Args:
            message: 사용자 메시지
            system_prompt: 시스템 프롬프트

        Returns:
            최종 응답
        """
        self._iteration_count = 0
        self._executor.reset()

        tool_results = []

        while self._iteration_count < self.config.max_iterations:
            self._iteration_count += 1

            # 액션 결정
            action = await self._planner.decide_action(
                user_message=message,
                conversation_history=self._conversation_history,
                tool_results=tool_results
            )

            # 응답 액션이면 종료 및 반환
            if action.type == ActionType.RESPONSE:
                return action.response or ""

            # 도구 실행
            if action.type == ActionType.TOOL_CALL:
                result = await self._executor.execute_action(action)
                tool_results.append({
                    "name": result.name,
                    "result": result.result,
                    "success": result.success,
                    "error": result.error
                })

                # 실행 실패 시 에러 반환
                if not result.success:
                    return f"도구 실행 중 오류가 발생했습니다: {result.error}"

            # 에러 액션
            if action.type == ActionType.ERROR:
                return f"오류가 발생했습니다: {action.response}"

        # 최대 반복 횟수 초과
        return "최대 반복 횟수를 초과했습니다. 다시 시도해주세요."

    async def _react_loop_stream(
        self,
        message: str,
        system_prompt: str
    ) -> AsyncIterator[str]:
        """
        ReAct 루프 (스트리밍)
        """
        self._iteration_count = 0
        self._executor.reset()

        tool_results = []

        while self._iteration_count < self.config.max_iterations:
            self._iteration_count += 1

            # 액션 결정
            action = await self._planner.decide_action(
                user_message=message,
                conversation_history=self._conversation_history,
                tool_results=tool_results
            )

            # 응답 액션이면 종료 스트리밍
            if action.type == ActionType.RESPONSE:
                # 이미 결정된 응답이 있으면 그것을 사용
                if action.response:
                    yield action.response
                return

            # 도구 실행
            if action.type == ActionType.TOOL_CALL:
                yield f"[도구 실행: {action.tool_name}]\n"

                result = await self._executor.execute_action(action)
                tool_results.append({
                    "name": result.name,
                    "result": result.result,
                    "success": result.success,
                    "error": result.error
                })

                if not result.success:
                    yield f"\n오류: {result.error}"
                    return

            if action.type == ActionType.ERROR:
                yield f"오류: {action.response}"
                return

        yield "최대 반복 횟수를 초과했습니다."

    def _build_system_prompt(self) -> str:
        """시스템 프롬프트 구성"""
        base_prompt = AGENT_SYSTEM_PROMPT

        # 페르소나 로드
        if self.config.persona:
            try:
                from .prompts import load_persona, get_prompt_template
                persona = load_persona(self.config.persona)
                persona_prompt = get_prompt_template(persona)
                base_prompt = f"{persona_prompt}\n\n{base_prompt}"
            except Exception as e:
                logger.warning(f"Failed to load persona: {e}")

        return base_prompt

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """대화 기록 반환"""
        return self._conversation_history.copy()

    def clear_history(self) -> None:
        """대화 기록 초기화"""
        self._conversation_history = []
        self._executor.reset()
        logger.info(f"Cleared chat history: {self.session_id}")

    async def update_config(self, **kwargs) -> None:
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.config, key) and value is not None:
                setattr(self.config, key, value)

        # LLM 재생성
        if 'model' in kwargs or 'temperature' in kwargs:
            self._llm_service = None

        logger.info(f"Chat config updated: {kwargs}")


def create_chat_system(session_id: str, **kwargs) -> ChatSystemChain:
    """
    Chat 시스템 인스턴스 생성 함수

    Args:
        session_id: 세션 ID
        **kwargs: ChatConfig 인자

    Returns:
        ChatSystemChain 인스턴스
    """
    return ChatSystemChain(session_id=session_id, **kwargs)
