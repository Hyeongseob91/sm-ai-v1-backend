"""
Chat System Chain
Agentic AI 기반 도구 사용 대화 시스템
"""

from typing import List, Dict, Any, Optional, AsyncIterator, Union
from dataclasses import dataclass, field
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
from .mcp import (
    MCPClientManager,
    MCPServerConfig,
    MCPToolAdapter,
    load_mcp_configs,
)
from .mcp.mcp_tool_adapter import create_all_mcp_tool_adapters

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

    # MCP 설정
    mcp_servers: Optional[List[Union[str, MCPServerConfig]]] = None
    mcp_config_path: Optional[str] = None  # MCP 설정 파일 경로

    # 페르소나
    persona: Optional[str] = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = DEFAULT_TOOLS.copy()


# ===================================================================================================
# ChatSystemChain
# ===================================================================================================
class ChatSystemChain:
    """
    Agentic Chat 시스템 메인

    LLM과 Planner로, Tools를 Executor로 사용하는 대화 시스템
    """

    # 1. 생성자
    def __init__(
        self,
        session_id: str,
        config: Optional[ChatConfig] = None,
        **kwargs
    ):
        self.session_id = session_id
        self.config = config or ChatConfig(**kwargs)

        # 1) 컴포넌트 초기화
        self._llm_service = None
        self._planner = AgentPlanner()
        self._executor = ToolExecutor()

        # 2) MCP 클라이언트 매니저
        self._mcp_manager = MCPClientManager()
        self._mcp_initialized = False

        # 3) 도구 초기화 (네이티브 도구만)
        self._initialize_tools()

        # 4) MCP 서버 등록
        self._register_mcp_servers()

        # 5) 상태
        self._iteration_count = 0
        self._conversation_history: List[Dict[str, str]] = []

    # 2. 도구 초기화
    def _initialize_tools(self) -> None:
        """네이티브 도구 초기화"""
        # 기본 도구가 등록되어 있지 않으면 초기화
        if not ToolRegistry.get_names():
            initialize_default_tools()

        logger.debug(f"Initialized native tools: {ToolRegistry.get_names()}")

    # 3. MCP 서버 등록
    def _register_mcp_servers(self) -> None:
        """MCP 서버 등록"""
        if not self.config.mcp_servers and not self.config.mcp_config_path:
            return

        # 설정 파일에서 로드
        if self.config.mcp_config_path:
            configs = load_mcp_configs(self.config.mcp_config_path)
            self._mcp_manager.register_servers(configs)

        # 직접 지정된 서버 등록
        if self.config.mcp_servers:
            for server in self.config.mcp_servers:
                if isinstance(server, str):
                    # 문자열인 경우 설정 파일에서 로드된 것 중 찾기
                    logger.warning(f"MCP server '{server}' specified as string, skipping")
                elif isinstance(server, MCPServerConfig):
                    self._mcp_manager.register_server(server)

        logger.info(f"Registered MCP servers: {self._mcp_manager.registered_servers}")

    # 4. MCP 도구 초기화
    async def _initialize_mcp_tools(self) -> None:
        """MCP 도구 초기화 (비동기)"""
        if self._mcp_initialized:
            return

        if not self._mcp_manager.registered_servers:
            self._mcp_initialized = True
            return

        # 모든 MCP 서버에 연결
        await self._mcp_manager.connect_all()

        # MCP 도구를 ToolRegistry에 등록
        adapters = create_all_mcp_tool_adapters(self._mcp_manager)
        for adapter in adapters:
            ToolRegistry.register(adapter)

        self._mcp_initialized = True
        logger.info(f"Initialized MCP tools: {len(adapters)} tools from {len(self._mcp_manager.connected_servers)} servers")

    # 5. 사용 가능한 도구 목록 업데이트
    def _update_available_tools(self) -> None:
        """사용 가능한 도구 목록 업데이트"""
        all_tools = ToolRegistry.get_names()

        # 설정된 도구가 있으면 필터링, 없으면 전체 사용
        if self.config.tools:
            # 네이티브 도구 중 설정된 것만 포함
            available = [t for t in self.config.tools if t in all_tools]
            # MCP 도구는 모두 포함
            mcp_tools = [t for t in all_tools if t.startswith("mcp_")]
            available.extend(mcp_tools)
        else:
            available = all_tools

        self._planner.set_available_tools(available)
        logger.debug(f"Available tools: {available}")

    # 6. 리소스 정리
    async def cleanup(self) -> None:
        """리소스 정리 (MCP 연결 해제 등)"""
        await self._mcp_manager.disconnect_all()
        logger.info(f"Cleaned up ChatSystemChain: {self.session_id}")


    # 1. Chat 메시지 처리 (동기)
    async def chat(self, message: str) -> str:
        """
        대화 메시지 처리 (동기)

        Args:
            message: 사용자 메시지

        Returns:
            AI 응답
        """
        # MCP 도구 초기화 (lazy loading)
        await self._initialize_mcp_tools()
        self._update_available_tools()

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

    async def _react_loop(self, message: str, system_prompt: str) -> str:
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


    # 2. Chat 메시지 처리 (스트리밍)
    async def chat_stream(self, message: str) -> AsyncIterator[str]:
        """
        대화 메시지 처리 (스트리밍)

        Args:
            message: 사용자 메시지

        Yields:
            응답 청크
        """
        # MCP 도구 초기화 (lazy loading)
        await self._initialize_mcp_tools()
        self._update_available_tools()

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

    async def _react_loop_stream(self, message: str, system_prompt: str) -> AsyncIterator[str]:
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
