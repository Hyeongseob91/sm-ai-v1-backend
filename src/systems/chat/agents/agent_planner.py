"""
Agent Planner
LLM을 사용한 계획 수립 및 의사결정
"""

from typing import List, Dict, Any, Optional
import logging
import json

from ..models.function_call_model import (
    AgentAction,
    AgentPlan,
    PlanStep,
    ActionType,
    FunctionCall
)
from ..tools.tool_registry import ToolRegistry
from ..constants import PLANNER_PROMPT, MAX_ITERATIONS

logger = logging.getLogger(__name__)


class AgentPlanner:
    """
    에이전트 계획 수립자
    사용자 요청을 분석하고 실행 계획을 수립
    """

    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain LLM 인스턴스
        """
        self._llm = llm
        self._available_tools = []

    def set_llm(self, llm) -> None:
        """LLM 설정"""
        self._llm = llm

    def set_available_tools(self, tools: List[str]) -> None:
        """사용 가능한 도구 설정"""
        self._available_tools = tools

    async def create_plan(self, user_message: str, context: Optional[str] = None) -> AgentPlan:
        """
        실행 계획 수립

        Args:
            user_message: 사용자 요청
            context: 추가 컨텍스트 (이전 대화 등)

        Returns:
            AgentPlan 객체
        """
        if not self._llm:
            raise RuntimeError("LLM not initialized")

        # 프롬프트 구성
        available_tools_str = ", ".join(self._available_tools) if self._available_tools else "None"

        planning_prompt = f"""
        {PLANNER_PROMPT.format(available_tools=available_tools_str)}

        사용자 요청: {user_message}
        {f"추가 컨텍스트: {context}" if context else ""}

        다음 JSON 형식으로 계획을 출력하세요:
        {{
            "goal": "최종 목표",
            "steps": [
                {{
                    "step_number": 1,
                    "action": "수행할 작업 설명",
                    "tool": "사용할 도구 이름 또는 null",
                    "expected_output": "예상 결과"
                }}
            ]
        }}
        """

        try:
            # LLM 호출
            response = await self._llm.ainvoke(planning_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # JSON 파싱
            plan_dict = self._parse_json_response(response_text)

            # AgentPlan 생성
            steps = [
                PlanStep(
                    step_number=s.get('step_number', i + 1),
                    action=s['action'],
                    tool=s.get('tool'),
                    expected_output=s.get('expected_output', ''),
                    dependencies=s.get('dependencies', [])
                )
                for i, s in enumerate(plan_dict.get('steps', []))
            ]

            plan = AgentPlan(
                goal=plan_dict.get('goal', user_message),
                steps=steps
            )

            logger.info(f"Created plan with {len(steps)} steps")
            return plan

        except Exception as e:
            logger.error(f"Planning error: {e}")
            # 기본 단일 단계 계획 반환
            return AgentPlan(
                goal=user_message,
                steps=[PlanStep(
                    step_number=1,
                    action="직접 응답",
                    tool=None,
                    expected_output="사용자 요청에 대한 직접 응답"
                )]
            )

    async def decide_action(
        self,
        user_message: str,
        conversation_history: List[Dict[str, Any]] = None,
        tool_results: List[Dict[str, Any]] = None
    ) -> AgentAction:
        """
        다음 액션 결정 (ReAct 스타일)

        Args:
            user_message: 사용자 메시지
            conversation_history: 대화 기록
            tool_results: 이전 도구 실행 결과

        Returns:
            AgentAction 객체
        """
        if not self._llm:
            raise RuntimeError("LLM not initialized")

        # 도구 정의 가져오기
        tools = ToolRegistry.get_openai_tools()

        # Function Calling을 지원하는 LLM인 경우
        if hasattr(self._llm, 'bind_tools'):
            llm_with_tools = self._llm.bind_tools(tools)

            # 메시지 구성
            messages = self._build_messages(
                user_message,
                conversation_history,
                tool_results
            )

            # LLM 호출
            response = await llm_with_tools.ainvoke(messages)

            # 응답 파싱
            return self._parse_llm_response(response)

        else:
            # Function Calling 미지원 LLM - 텍스트 기반 처리
            return await self._decide_action_text_based(
                user_message,
                conversation_history,
                tool_results
            )

    def _build_messages(
        self,
        user_message: str,
        conversation_history: List[Dict[str, Any]] = None,
        tool_results: List[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """LLM 메시지 구성"""
        from ..constants import AGENT_SYSTEM_PROMPT

        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

        # 대화 기록 추가
        if conversation_history:
            for msg in conversation_history[-10:]:  # 최근 10개만
                messages.append(msg)

        # 도구 결과 추가
        if tool_results:
            for result in tool_results:
                messages.append({
                    "role": "assistant",
                    "content": f"Tool '{result['name']}' result: {result['result']}"
                })

        # 사용자 메시지
        messages.append({"role": "user", "content": user_message})

        return messages

    def _parse_llm_response(self, response) -> AgentAction:
        """LLM 응답 파싱"""

        # Function Call이 있는 경우
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            return AgentAction(
                type=ActionType.TOOL_CALL,
                tool_name=tool_call['name'],
                tool_input=tool_call.get('args', {})
            )

        # 일반 텍스트 응답
        content = response.content if hasattr(response, 'content') else str(response)
        return AgentAction(
            type=ActionType.RESPONSE,
            response=content
        )

    async def _decide_action_text_based(
        self,
        user_message: str,
        conversation_history: List[Dict[str, Any]] = None,
        tool_results: List[Dict[str, Any]] = None
    ) -> AgentAction:
        """텍스트 기반 액션 결정 (Function Calling 미지원 LLM용)"""

        tools_desc = "\n".join([
            f"- {name}: {ToolRegistry.get(name).description}"
            for name in ToolRegistry.get_names()
        ])

        prompt = f"""
사용 가능한 도구:
{tools_desc}

사용자 요청: {user_message}

다음 중 하나를 JSON으로 응답하세요:
1. 도구 사용: {{"action": "tool", "tool_name": "도구이름", "tool_input": {{"arg": "value"}}}}
2. 직접 응답: {{"action": "response", "content": "응답 내용"}}
"""

        response = await self._llm.ainvoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)

        try:
            action_dict = self._parse_json_response(response_text)

            if action_dict.get('action') == 'tool':
                return AgentAction(
                    type=ActionType.TOOL_CALL,
                    tool_name=action_dict['tool_name'],
                    tool_input=action_dict.get('tool_input', {})
                )
            else:
                return AgentAction(
                    type=ActionType.RESPONSE,
                    response=action_dict.get('content', response_text)
                )

        except Exception:
            return AgentAction(
                type=ActionType.RESPONSE,
                response=response_text
            )

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """텍스트에서 JSON 추출 및 파싱"""
        # JSON 블록 찾기
        import re

        # ```json ... ``` 패턴
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # { ... } 패턴
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        raise ValueError(f"No JSON found in response: {text[:200]}")
