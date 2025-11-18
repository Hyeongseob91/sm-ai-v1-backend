"""
Function Calling 관련 모델 정의
LLM Function Calling을 위한 스키마
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# Function Call 모델
# =============================================================================

class FunctionCall(BaseModel):
    """LLM이 생성하는 Function Call"""
    name: str = Field(..., description="호출할 함수 이름")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="함수 인자")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments
        }


class FunctionCallResult(BaseModel):
    """Function Call 실행 결과"""
    name: str = Field(..., description="실행된 함수 이름")
    result: Any = Field(..., description="실행 결과")
    success: bool = Field(default=True, description="성공 여부")
    error: Optional[str] = Field(default=None, description="에러 메시지")
    execution_time: float = Field(default=0.0, description="실행 시간 (초)")


# =============================================================================
# Tool Definition 모델
# =============================================================================

class ParameterType(str, Enum):
    """파라미터 타입"""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """도구 파라미터 정의"""
    name: str
    type: ParameterType
    description: str
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None


class ToolDefinition(BaseModel):
    """도구 정의 (OpenAI Function 형식)"""
    name: str = Field(..., description="도구 이름")
    description: str = Field(..., description="도구 설명")
    parameters: List[ToolParameter] = Field(default_factory=list)

    def to_openai_format(self) -> Dict[str, Any]:
        """OpenAI Function Calling 형식으로 변환"""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type.value,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


# =============================================================================
# Agent 관련 모델
# =============================================================================

class ActionType(str, Enum):
    """에이전트 액션 타입"""
    TOOL_CALL = "tool_call"
    RESPONSE = "response"
    PLAN = "plan"
    ERROR = "error"


class AgentAction(BaseModel):
    """에이전트 액션"""
    type: ActionType
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    response: Optional[str] = None
    reasoning: Optional[str] = None


class PlanStep(BaseModel):
    """계획 단계"""
    step_number: int
    action: str
    tool: Optional[str] = None
    expected_output: str
    dependencies: List[int] = Field(default_factory=list)


class AgentPlan(BaseModel):
    """에이전트 실행 계획"""
    goal: str = Field(..., description="최종 목표")
    steps: List[PlanStep] = Field(default_factory=list)
    current_step: int = 0
    completed_steps: List[int] = Field(default_factory=list)

    def get_next_step(self) -> Optional[PlanStep]:
        """다음 실행할 단계 반환"""
        for step in self.steps:
            if step.step_number not in self.completed_steps:
                # 의존성 확인
                if all(dep in self.completed_steps for dep in step.dependencies):
                    return step
        return None

    def mark_completed(self, step_number: int) -> None:
        """단계 완료 처리"""
        if step_number not in self.completed_steps:
            self.completed_steps.append(step_number)

    def is_complete(self) -> bool:
        """모든 단계 완료 여부"""
        return len(self.completed_steps) == len(self.steps)


# =============================================================================
# State 모델 (LangGraph용)
# =============================================================================

class AgentState(BaseModel):
    """에이전트 상태 (LangGraph State)"""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    plan: Optional[AgentPlan] = None
    current_action: Optional[AgentAction] = None
    tool_results: List[FunctionCallResult] = Field(default_factory=list)
    final_response: Optional[str] = None
    iteration_count: int = 0
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
