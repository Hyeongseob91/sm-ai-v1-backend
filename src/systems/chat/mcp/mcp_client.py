"""
MCP Client Manager
MCP 서버 연결 및 도구 호출 관리
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """MCP 관련 기본 에러"""
    pass


class MCPConnectionError(MCPError):
    """MCP 연결 에러"""
    pass


class MCPTimeoutError(MCPError):
    """MCP 타임아웃 에러"""
    pass


class MCPToolNotFoundError(MCPError):
    """MCP 도구를 찾을 수 없음"""
    pass


@dataclass
class MCPServerConfig:
    """MCP 서버 설정"""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    timeout: float = 30.0
    enabled: bool = True


class MCPClientManager:
    """
    MCP 클라이언트 관리자

    여러 MCP 서버와의 연결을 관리하고 도구 호출을 처리합니다.
    """

    def __init__(self):
        self._sessions: Dict[str, Any] = {}
        self._server_configs: Dict[str, MCPServerConfig] = {}
        self._tools_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._tool_to_server: Dict[str, str] = {}  # tool_name -> server_name 매핑
        self._initialized: bool = False
        self._read_streams: Dict[str, Any] = {}
        self._write_streams: Dict[str, Any] = {}

    def register_server(self, config: MCPServerConfig) -> None:
        """
        MCP 서버 등록

        Args:
            config: MCP 서버 설정
        """
        if not config.enabled:
            logger.info(f"MCP server '{config.name}' is disabled, skipping registration")
            return

        self._server_configs[config.name] = config
        logger.info(f"Registered MCP server: {config.name}")

    def register_servers(self, configs: List[MCPServerConfig]) -> None:
        """
        여러 MCP 서버 등록

        Args:
            configs: MCP 서버 설정 리스트
        """
        for config in configs:
            self.register_server(config)

    async def connect(self, server_name: str) -> Any:
        """
        MCP 서버에 연결

        Args:
            server_name: 서버 이름

        Returns:
            ClientSession 인스턴스
        """
        if server_name in self._sessions:
            return self._sessions[server_name]

        config = self._server_configs.get(server_name)
        if not config:
            raise MCPConnectionError(f"Unknown MCP server: {server_name}")

        try:
            # MCP SDK import (lazy loading)
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env
            )

            # stdio 클라이언트 연결
            read_stream, write_stream = await stdio_client(params).__aenter__()
            self._read_streams[server_name] = read_stream
            self._write_streams[server_name] = write_stream

            # 세션 생성 및 초기화
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()

            self._sessions[server_name] = session

            # 도구 목록 캐싱
            tools_response = await session.list_tools()
            self._tools_cache[server_name] = []

            for tool in tools_response.tools:
                tool_dict = {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "inputSchema": getattr(tool, "inputSchema", {})
                }
                self._tools_cache[server_name].append(tool_dict)
                self._tool_to_server[tool.name] = server_name

            logger.info(f"Connected to MCP server '{server_name}' with {len(self._tools_cache[server_name])} tools")
            return session

        except ImportError:
            raise MCPConnectionError(
                "MCP SDK not installed. Run: pip install mcp"
            )
        except Exception as e:
            raise MCPConnectionError(f"Failed to connect to {server_name}: {e}")

    async def connect_all(self) -> None:
        """모든 등록된 MCP 서버에 연결"""
        for server_name in self._server_configs:
            try:
                await self.connect(server_name)
            except MCPConnectionError as e:
                logger.error(f"Failed to connect to {server_name}: {e}")

        self._initialized = True

    async def disconnect(self, server_name: str) -> None:
        """
        서버 연결 해제

        Args:
            server_name: 서버 이름
        """
        if server_name in self._sessions:
            try:
                await self._sessions[server_name].__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing session for {server_name}: {e}")

            del self._sessions[server_name]

            # 도구 매핑 정리
            tools_to_remove = [
                tool for tool, server in self._tool_to_server.items()
                if server == server_name
            ]
            for tool in tools_to_remove:
                del self._tool_to_server[tool]

            if server_name in self._tools_cache:
                del self._tools_cache[server_name]

            logger.info(f"Disconnected from MCP server: {server_name}")

    async def disconnect_all(self) -> None:
        """모든 연결 해제"""
        for server_name in list(self._sessions.keys()):
            await self.disconnect(server_name)
        self._initialized = False

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        server_name: Optional[str] = None
    ) -> Any:
        """
        MCP 도구 호출

        Args:
            tool_name: 도구 이름
            arguments: 도구 인자
            server_name: 서버 이름 (생략 시 자동 탐색)

        Returns:
            도구 실행 결과
        """
        # 서버 이름 결정
        if server_name is None:
            server_name = self._tool_to_server.get(tool_name)
            if not server_name:
                raise MCPToolNotFoundError(f"Tool not found: {tool_name}")

        # 세션 확인
        session = self._sessions.get(server_name)
        if not session:
            session = await self.connect(server_name)

        # 타임아웃 설정
        timeout = self._server_configs[server_name].timeout

        try:
            result = await asyncio.wait_for(
                session.call_tool(name=tool_name, arguments=arguments),
                timeout=timeout
            )

            # 결과 추출
            if hasattr(result, 'content'):
                # TextContent 또는 다른 컨텐츠 타입 처리
                if isinstance(result.content, list):
                    return [self._extract_content(c) for c in result.content]
                return self._extract_content(result.content)

            return result

        except asyncio.TimeoutError:
            raise MCPTimeoutError(
                f"Timeout calling {tool_name} on {server_name} (timeout: {timeout}s)"
            )

    def _extract_content(self, content: Any) -> Any:
        """MCP 컨텐츠에서 실제 값 추출"""
        if hasattr(content, 'text'):
            return content.text
        if hasattr(content, 'data'):
            return content.data
        return str(content)

    async def call_tool_safe(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 3,
        server_name: Optional[str] = None
    ) -> Any:
        """
        재시도 로직이 포함된 안전한 도구 호출

        Args:
            tool_name: 도구 이름
            arguments: 도구 인자
            retries: 재시도 횟수
            server_name: 서버 이름

        Returns:
            도구 실행 결과
        """
        last_error = None
        actual_server = server_name or self._tool_to_server.get(tool_name)

        for attempt in range(retries):
            try:
                return await self.call_tool(tool_name, arguments, actual_server)

            except MCPTimeoutError as e:
                last_error = e
                logger.warning(f"Timeout on attempt {attempt + 1}/{retries} for {tool_name}")

                # 재연결 시도
                if actual_server:
                    await self.disconnect(actual_server)
                    await asyncio.sleep(1 * (attempt + 1))

            except MCPError as e:
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

        raise last_error or MCPError(f"Failed to call {tool_name} after {retries} retries")

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        모든 MCP 서버의 도구 목록 반환

        Returns:
            모든 도구 정보 리스트
        """
        all_tools = []
        for server_name, tools in self._tools_cache.items():
            for tool in tools:
                tool_info = tool.copy()
                tool_info["server"] = server_name
                all_tools.append(tool_info)
        return all_tools

    def get_server_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """
        특정 서버의 도구 목록 반환

        Args:
            server_name: 서버 이름

        Returns:
            도구 정보 리스트
        """
        return self._tools_cache.get(server_name, [])

    def get_tool_server(self, tool_name: str) -> Optional[str]:
        """
        도구가 속한 서버 이름 반환

        Args:
            tool_name: 도구 이름

        Returns:
            서버 이름 또는 None
        """
        return self._tool_to_server.get(tool_name)

    def is_connected(self, server_name: str) -> bool:
        """서버 연결 상태 확인"""
        return server_name in self._sessions

    def is_initialized(self) -> bool:
        """초기화 완료 여부"""
        return self._initialized

    @property
    def connected_servers(self) -> List[str]:
        """연결된 서버 목록"""
        return list(self._sessions.keys())

    @property
    def registered_servers(self) -> List[str]:
        """등록된 서버 목록"""
        return list(self._server_configs.keys())
