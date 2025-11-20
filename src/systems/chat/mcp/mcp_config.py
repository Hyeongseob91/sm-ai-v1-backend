"""
MCP Configuration Loader
YAML 파일에서 MCP 서버 설정을 로드
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import yaml

from .mcp_client import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPConfigLoader:
    """MCP 설정 로더"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 설정 파일 경로 (기본값: config/mcp_servers.yaml)
        """
        if config_path is None:
            # 기본 경로: 프로젝트 루트의 config/mcp_servers.yaml
            base_dir = Path(__file__).parent.parent.parent.parent.parent
            config_path = base_dir / "config" / "mcp_servers.yaml"

        self.config_path = Path(config_path)

    def load(self) -> List[MCPServerConfig]:
        """
        설정 파일에서 MCP 서버 설정 로드

        Returns:
            MCPServerConfig 리스트
        """
        if not self.config_path.exists():
            logger.warning(f"MCP config file not found: {self.config_path}")
            return []

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if not config:
                return []

            return self._parse_config(config)

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse MCP config: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return []

    def _parse_config(self, config: Dict[str, Any]) -> List[MCPServerConfig]:
        """
        설정 딕셔너리를 MCPServerConfig 리스트로 변환

        Args:
            config: YAML에서 로드한 설정

        Returns:
            MCPServerConfig 리스트
        """
        servers = []

        for name, settings in config.get("servers", {}).items():
            try:
                server_config = MCPServerConfig(
                    name=name,
                    command=settings.get("command", ""),
                    args=settings.get("args", []),
                    env=settings.get("env"),
                    timeout=float(settings.get("timeout", 30.0)),
                    enabled=settings.get("enabled", True)
                )

                if not server_config.command:
                    logger.warning(f"Skipping server '{name}': no command specified")
                    continue

                servers.append(server_config)
                logger.debug(f"Loaded MCP server config: {name}")

            except Exception as e:
                logger.warning(f"Failed to parse server config '{name}': {e}")

        return servers

    def get_server_names(self) -> List[str]:
        """
        설정된 서버 이름 목록 반환

        Returns:
            서버 이름 리스트
        """
        configs = self.load()
        return [c.name for c in configs]


def load_mcp_configs(config_path: Optional[str] = None) -> List[MCPServerConfig]:
    """
    MCP 서버 설정 로드 편의 함수

    Args:
        config_path: 설정 파일 경로

    Returns:
        MCPServerConfig 리스트
    """
    loader = MCPConfigLoader(config_path)
    return loader.load()


def create_default_config_template(output_path: Optional[str] = None) -> str:
    """
    기본 MCP 설정 템플릿 생성

    Args:
        output_path: 출력 경로 (지정 시 파일로 저장)

    Returns:
        템플릿 YAML 문자열
    """
    template = """# MCP Server Configuration
# MCP 서버 설정 파일

servers:
  # 파일 시스템 서버 예시
  # filesystem:
  #   command: "npx"
  #   args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
  #   timeout: 30
  #   enabled: true

  # Fetch 서버 예시 (웹 페이지/API 호출)
  # fetch:
  #   command: "npx"
  #   args: ["-y", "@modelcontextprotocol/server-fetch"]
  #   timeout: 45
  #   enabled: true

  # SQLite 서버 예시
  # sqlite:
  #   command: "npx"
  #   args: ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "./data/database.db"]
  #   timeout: 30
  #   enabled: true

  # Python 기반 MCP 서버 예시
  # custom_python:
  #   command: "python"
  #   args: ["-m", "my_mcp_server"]
  #   env:
  #     API_KEY: "your-api-key"
  #   timeout: 60
  #   enabled: true

  # Brave Search 서버 예시
  # brave_search:
  #   command: "npx"
  #   args: ["-y", "@modelcontextprotocol/server-brave-search"]
  #   env:
  #     BRAVE_API_KEY: "your-brave-api-key"
  #   timeout: 30
  #   enabled: false

  # GitHub 서버 예시
  # github:
  #   command: "npx"
  #   args: ["-y", "@modelcontextprotocol/server-github"]
  #   env:
  #     GITHUB_PERSONAL_ACCESS_TOKEN: "your-github-token"
  #   timeout: 30
  #   enabled: false
"""

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(template)

        logger.info(f"Created MCP config template: {output_path}")

    return template
