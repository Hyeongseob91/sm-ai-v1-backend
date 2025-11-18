"""
Tool Executor
도구 실행 및 결과 관리
"""

from typing import Any, Dict, List, Optional
import logging
import time
import asyncio

from ..models.function_call_model import (
    FunctionCall,
    FunctionCallResult,
    AgentAction,
    ActionType
)
from ..tools.tool_registry import ToolRegistry
from ..constants import MAX_TOOL_CALLS, TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    도구 실행자
    에이전트가 결정한 도구를 실행하고 결과를 반환
    """

    def __init__(self):
        self._execution_count = 0
        self._results_history: List[FunctionCallResult] = []

    async def execute_action(self, action: AgentAction) -> FunctionCallResult:
        """
        에이전트 액션 실행

        Args:
            action: 실행할 AgentAction

        Returns:
            FunctionCallResult
        """
        if action.type != ActionType.TOOL_CALL:
            return FunctionCallResult(
                name="none",
                result="No tool execution required",
                success=True
            )

        # 실행 제한 확인
        if self._execution_count >= MAX_TOOL_CALLS:
            return FunctionCallResult(
                name=action.tool_name or "unknown",
                result=None,
                success=False,
                error=f"Maximum tool calls exceeded ({MAX_TOOL_CALLS})"
            )

        return await self.execute_tool(
            action.tool_name,
            action.tool_input or {}
        )

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> FunctionCallResult:
        """
        도구 실행

        Args:
            tool_name: 실행할 도구 이름
            arguments: 도구 인자

        Returns:
            FunctionCallResult
        """
        start_time = time.time()

        # 도구 조회
        tool = ToolRegistry.get(tool_name)
        if tool is None:
            return FunctionCallResult(
                name=tool_name,
                result=None,
                success=False,
                error=f"Tool not found: {tool_name}",
                execution_time=time.time() - start_time
            )

        try:
            # 타임아웃 적용하여 실행
            result = await asyncio.wait_for(
                tool.execute(**arguments),
                timeout=TIMEOUT_SECONDS
            )

            self._execution_count += 1
            execution_time = time.time() - start_time

            # 결과 기록
            call_result = FunctionCallResult(
                name=tool_name,
                result=result,
                success=True,
                execution_time=execution_time
            )

            self._results_history.append(call_result)
            logger.info(f"Tool '{tool_name}' executed in {execution_time:.2f}s")

            return call_result

        except asyncio.TimeoutError:
            return FunctionCallResult(
                name=tool_name,
                result=None,
                success=False,
                error=f"Tool execution timed out ({TIMEOUT_SECONDS}s)",
                execution_time=TIMEOUT_SECONDS
            )

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return FunctionCallResult(
                name=tool_name,
                result=None,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def execute_function_call(
        self,
        function_call: FunctionCall
    ) -> FunctionCallResult:
        """
        FunctionCall 객체 실행

        Args:
            function_call: 실행할 FunctionCall

        Returns:
            FunctionCallResult
        """
        return await self.execute_tool(
            function_call.name,
            function_call.arguments
        )

    async def execute_batch(
        self,
        function_calls: List[FunctionCall],
        parallel: bool = False
    ) -> List[FunctionCallResult]:
        """
        여러 도구 일괄 실행

        Args:
            function_calls: 실행할 FunctionCall 리스트
            parallel: 병렬 실행 여부

        Returns:
            FunctionCallResult 리스트
        """
        if parallel:
            # 병렬 실행
            tasks = [
                self.execute_function_call(fc)
                for fc in function_calls
            ]
            return await asyncio.gather(*tasks)
        else:
            # 순차 실행
            results = []
            for fc in function_calls:
                result = await self.execute_function_call(fc)
                results.append(result)
            return results

    def get_results_history(self) -> List[FunctionCallResult]:
        """실행 결과 기록 반환"""
        return self._results_history.copy()

    def get_execution_count(self) -> int:
        """실행 횟수 반환"""
        return self._execution_count

    def reset(self) -> None:
        """실행 상태 초기화"""
        self._execution_count = 0
        self._results_history = []
        logger.debug("Tool executor reset")

    def format_results_for_llm(
        self,
        results: List[FunctionCallResult] = None
    ) -> str:
        """LLM에 전달할 형식으로 결과 포맷팅"""
        results = results or self._results_history

        if not results:
            return "No tool results available."

        formatted = []
        for result in results:
            if result.success:
                formatted.append(
                    f"Tool: {result.name}\n"
                    f"Result: {result.result}\n"
                    f"Time: {result.execution_time:.2f}s"
                )
            else:
                formatted.append(
                    f"Tool: {result.name}\n"
                    f"Error: {result.error}"
                )

        return "\n\n---\n\n".join(formatted)

    def get_last_result(self) -> Optional[FunctionCallResult]:
        """마지막 실행 결과 반환"""
        if self._results_history:
            return self._results_history[-1]
        return None
