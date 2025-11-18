"""
Tool Registry
사용 가능한 모든 도구를 중앙에서 관리
"""

from typing import Dict, List, Any, Optional, Callable, Type
import logging

from ..models.function_call_model import ToolDefinition

logger = logging.getLogger(__name__)


class BaseTool:
    """모든 도구의 기본 클래스"""

    name: str = "base_tool"
    description: str = "Base tool"

    def get_definition(self) -> ToolDefinition:
        """도구 정의 반환 (서브클래스에서 구현)"""
        raise NotImplementedError

    async def execute(self, **kwargs) -> Any:
        """도구 실행 (서브클래스에서 구현)"""
        raise NotImplementedError

    def to_openai_format(self) -> Dict[str, Any]:
        """OpenAI Function Calling 형식으로 변환"""
        return self.get_definition().to_openai_format()


class ToolRegistry:
    """도구 중앙 레지스트리"""

    _instance = None
    _tools: Dict[str, BaseTool] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._tools = {}
        return cls._instance

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        """도구 등록"""
        cls._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    @classmethod
    def get(cls, name: str) -> Optional[BaseTool]:
        """이름으로 도구 조회"""
        return cls._tools.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, BaseTool]:
        """모든 도구 반환"""
        return cls._tools.copy()

    @classmethod
    def get_names(cls) -> List[str]:
        """등록된 도구 이름 목록"""
        return list(cls._tools.keys())

    @classmethod
    def get_definitions(cls) -> List[ToolDefinition]:
        """모든 도구 정의 반환"""
        return [tool.get_definition() for tool in cls._tools.values()]

    @classmethod
    def get_openai_tools(cls) -> List[Dict[str, Any]]:
        """OpenAI 형식의 도구 목록"""
        return [tool.to_openai_format() for tool in cls._tools.values()]

    @classmethod
    def clear(cls) -> None:
        """레지스트리 초기화"""
        cls._tools.clear()

    @classmethod
    async def execute(cls, name: str, **kwargs) -> Any:
        """도구 실행"""
        tool = cls.get(name)
        if tool is None:
            raise ValueError(f"Tool not found: {name}")

        logger.debug(f"Executing tool: {name} with args: {kwargs}")
        return await tool.execute(**kwargs)


def register_tool(tool: BaseTool) -> BaseTool:
    """도구 등록 편의 함수"""
    ToolRegistry.register(tool)
    return tool


def get_tool(name: str) -> Optional[BaseTool]:
    """도구 조회 편의 함수"""
    return ToolRegistry.get(name)


def tool(name: str, description: str):
    """도구 등록 데코레이터"""
    def decorator(cls: Type[BaseTool]) -> Type[BaseTool]:
        cls.name = name
        cls.description = description
        instance = cls()
        ToolRegistry.register(instance)
        return cls
    return decorator


def initialize_default_tools() -> None:
    """기본 도구들 초기화"""
    from .data_analyzer import DataAnalyzerTool
    from .chart_generator import ChartGeneratorTool
    from .report_formatter import ReportFormatterTool

    register_tool(DataAnalyzerTool())
    register_tool(ChartGeneratorTool())
    register_tool(ReportFormatterTool())

    logger.info("Initialized default tools")
