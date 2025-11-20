"""
MCP Tool Adapter
MCP 도구를 BaseTool 형태로 변환
"""

import logging
from typing import Any, Dict, List, TYPE_CHECKING

from ..tools.tool_registry import BaseTool
from ..models.function_call_model import ToolDefinition, ToolParameter, ParameterType

if TYPE_CHECKING:
    from .mcp_client import MCPClientManager

logger = logging.getLogger(__name__)


class MCPToolAdapter(BaseTool):
    """
    MCP 도구를 BaseTool 형태로 래핑

    ToolRegistry에 등록하여 기존 시스템과 통합 가능
    """

    def __init__(
        self,
        mcp_manager: "MCPClientManager",
        server_name: str,
        tool_info: Dict[str, Any]
    ):
        """
        Args:
            mcp_manager: MCP 클라이언트 관리자
            server_name: MCP 서버 이름
            tool_info: MCP 도구 정보 (name, description, inputSchema)
        """
        self.mcp_manager = mcp_manager
        self.server_name = server_name

        # 원본 도구 이름 저장
        self.original_name = tool_info.get("name", "unknown")

        # BaseTool 속성 설정
        # 네이밍: mcp_{server}_{tool} 형식으로 고유성 보장
        self.name = f"mcp_{server_name}_{self.original_name}"
        self.description = tool_info.get("description", f"MCP tool: {self.original_name}")

        # 입력 스키마 저장
        self._input_schema = tool_info.get("inputSchema", {})

        logger.debug(f"Created MCP tool adapter: {self.name}")

    def get_definition(self) -> ToolDefinition:
        """
        도구 정의 반환

        Returns:
            ToolDefinition 인스턴스
        """
        parameters = []

        properties = self._input_schema.get("properties", {})
        required = self._input_schema.get("required", [])

        for name, schema in properties.items():
            param_type = self._convert_json_type(schema.get("type", "string"))

            param = ToolParameter(
                name=name,
                type=param_type,
                description=schema.get("description", ""),
                required=name in required,
                enum=schema.get("enum"),
                default=schema.get("default")
            )
            parameters.append(param)

        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=parameters
        )

    async def execute(self, **kwargs) -> Any:
        """
        MCP 도구 실행

        Args:
            **kwargs: 도구 인자

        Returns:
            실행 결과
        """
        logger.debug(f"Executing MCP tool: {self.name} with args: {kwargs}")

        try:
            result = await self.mcp_manager.call_tool(
                tool_name=self.original_name,
                arguments=kwargs,
                server_name=self.server_name
            )

            logger.debug(f"MCP tool {self.name} completed successfully")
            return result

        except Exception as e:
            logger.error(f"MCP tool {self.name} failed: {e}")
            raise

    def _convert_json_type(self, json_type: str) -> ParameterType:
        """
        JSON Schema 타입을 ParameterType으로 변환

        Args:
            json_type: JSON Schema 타입 문자열

        Returns:
            ParameterType enum
        """
        type_map = {
            "string": ParameterType.STRING,
            "integer": ParameterType.INTEGER,
            "number": ParameterType.NUMBER,
            "boolean": ParameterType.BOOLEAN,
            "array": ParameterType.ARRAY,
            "object": ParameterType.OBJECT,
        }
        return type_map.get(json_type.lower(), ParameterType.STRING)

    def __repr__(self) -> str:
        return f"MCPToolAdapter(name={self.name}, server={self.server_name})"


def create_mcp_tool_adapters(
    mcp_manager: "MCPClientManager",
    server_name: str
) -> List[MCPToolAdapter]:
    """
    특정 서버의 모든 도구에 대한 어댑터 생성

    Args:
        mcp_manager: MCP 클라이언트 관리자
        server_name: 서버 이름

    Returns:
        MCPToolAdapter 리스트
    """
    tools = mcp_manager.get_server_tools(server_name)
    adapters = []

    for tool_info in tools:
        adapter = MCPToolAdapter(mcp_manager, server_name, tool_info)
        adapters.append(adapter)

    logger.info(f"Created {len(adapters)} tool adapters for server '{server_name}'")
    return adapters


def create_all_mcp_tool_adapters(
    mcp_manager: "MCPClientManager"
) -> List[MCPToolAdapter]:
    """
    모든 연결된 서버의 도구에 대한 어댑터 생성

    Args:
        mcp_manager: MCP 클라이언트 관리자

    Returns:
        모든 MCPToolAdapter 리스트
    """
    all_adapters = []

    for server_name in mcp_manager.connected_servers:
        adapters = create_mcp_tool_adapters(mcp_manager, server_name)
        all_adapters.extend(adapters)

    logger.info(f"Created total {len(all_adapters)} MCP tool adapters")
    return all_adapters
