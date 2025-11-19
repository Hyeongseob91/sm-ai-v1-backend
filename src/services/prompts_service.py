"""
프롬프트 서비스

모든 프롬프트 관련 기능을 통합 관리하는 서비스 클래스입니다.
Chatbot, RAG, Agentic 등 다양한 시스템에서 일관된 인터페이스로 프롬프트를 사용할 수 있습니다.

주요 기능:
    - 프롬프트 파일 목록 조회
    - 프롬프트 파일 경로 조회
    - 프롬프트 내용 로드 (YAML)
    - 프롬프트 메타 정보 조회

사용 예시:
    >>> from src.services.prompts_service import PromptsService
    >>>
    >>> # 프롬프트 목록 조회
    >>> prompts = PromptsService.list_prompts("chatbot")
    >>> print(prompts)  # ['01-general', '02-sns-marketing', ...]
    >>>
    >>> # 프롬프트 내용 로드
    >>> content = PromptsService.load_prompt("chatbot", "01-general")
    >>> print(content['system_prompt'])
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import logging

from src.config.config_prompts import (
    PROMPTS_DIR,
    CHATBOT_PROMPTS_DIR,
    RAG_PROMPTS_DIR,
    AGENTIC_PROMPTS_DIR,
)

logger = logging.getLogger(__name__)


class PromptsService:
    """
    통합 프롬프트 서비스 클래스

    모든 프롬프트 관련 기능을 제공하는 중앙 집중식 서비스입니다.
    프롬프트 타입별 디렉토리 매핑을 단일 위치에서 관리하여
    새로운 프롬프트 타입 추가 시 이 클래스만 수정하면 됩니다.

    Attributes:
        PROMPT_DIRS: 프롬프트 타입별 디렉토리 매핑

    Examples:
        기본 사용:
            >>> prompts = PromptsService.list_prompts("chatbot")
            >>> content = PromptsService.load_prompt("chatbot", prompts[0])

        전체 프롬프트 조회:
            >>> all_prompts = PromptsService.get_all_prompts()
            >>> for prompt_type, files in all_prompts.items():
            ...     print(f"{prompt_type}: {len(files)} files")
    """

    # =========================================================================
    # 프롬프트 타입별 디렉토리 매핑 (단일 정의)
    # =========================================================================

    PROMPT_DIRS: Dict[str, Path] = {
        "chatbot": CHATBOT_PROMPTS_DIR,
        "rag": RAG_PROMPTS_DIR,
        "agentic": AGENTIC_PROMPTS_DIR,
    }


    # =========================================================================
    # 기본 조회 메서드
    # =========================================================================

    @classmethod
    def get_prompt_types(cls) -> List[str]:
        """
        지원하는 프롬프트 타입 목록 반환

        Returns:
            List[str]: 프롬프트 타입 목록 (예: ["chatbot", "rag", "agentic"])
        """
        return list(cls.PROMPT_DIRS.keys())


    @classmethod
    def get_prompt_dir(cls, prompt_type: str) -> Path:
        """
        프롬프트 타입별 디렉토리 경로 반환

        Args:
            prompt_type: 프롬프트 타입 ("chatbot", "rag", "agentic")

        Returns:
            Path: 해당 타입의 프롬프트 디렉토리 경로

        Raises:
            ValueError: 유효하지 않은 prompt_type인 경우
        """
        if prompt_type not in cls.PROMPT_DIRS:
            raise ValueError(
                f"Invalid prompt_type: '{prompt_type}'. "
                f"Valid types: {list(cls.PROMPT_DIRS.keys())}"
            )
        return cls.PROMPT_DIRS[prompt_type]


    # =========================================================================
    # 프롬프트 목록 조회
    # =========================================================================

    @classmethod
    def list_prompts(cls, prompt_type: str) -> List[str]:
        """
        프롬프트 파일 목록 반환 (파일명만, 확장자 제외)

        Args:
            prompt_type: 프롬프트 타입 ("chatbot", "rag", "agentic")

        Returns:
            List[str]: 프롬프트 파일명 목록 (정렬됨)
                      예: ["01-general", "02-sns-marketing"]

        Examples:
            >>> PromptsService.list_prompts("chatbot")
            ['01-general', '02-sns-marketing', '03-text-summary', '04-prompt-maker']
        """
        prompt_dir = cls.get_prompt_dir(prompt_type)

        if not prompt_dir.exists():
            logger.warning(f"Prompt directory not found: {prompt_dir}")
            return []

        # YAML 파일 검색 (.yaml, .yml 모두 지원)
        files = set()
        for ext in [".yaml", ".yml"]:
            files.update(f.stem for f in prompt_dir.glob(f"*{ext}"))

        return sorted(files)


    @classmethod
    def list_prompt_files(cls, prompt_type: str) -> List[str]:
        """
        프롬프트 파일 전체 경로 목록 반환

        Args:
            prompt_type: 프롬프트 타입 ("chatbot", "rag", "agentic")

        Returns:
            List[str]: 프롬프트 파일 전체 경로 목록 (문자열)

        Examples:
            >>> PromptsService.list_prompt_files("chatbot")
            ['C:/path/to/prompts/chatbot/01-general.yaml', ...]
        """
        prompt_dir = cls.get_prompt_dir(prompt_type)

        if not prompt_dir.exists():
            logger.warning(f"Prompt directory not found: {prompt_dir}")
            return []

        files = list(prompt_dir.glob("*.yaml")) + list(prompt_dir.glob("*.yml"))
        return sorted(str(f) for f in files)


    # =========================================================================
    # 프롬프트 경로 및 로드
    # =========================================================================

    @classmethod
    def get_prompt_path(cls, prompt_type: str, prompt_name: str) -> Path:
        """
        프롬프트 파일의 전체 경로 반환

        Args:
            prompt_type: 프롬프트 타입 ("chatbot", "rag", "agentic")
            prompt_name: 프롬프트 파일명 (확장자 포함/미포함 모두 가능)

        Returns:
            Path: 프롬프트 파일의 전체 경로

        Raises:
            FileNotFoundError: 파일을 찾을 수 없는 경우

        Examples:
            >>> path = PromptsService.get_prompt_path("chatbot", "01-general")
            >>> print(path)
            WindowsPath('C:/path/to/prompts/chatbot/01-general.yaml')
        """
        prompt_dir = cls.get_prompt_dir(prompt_type)

        # 확장자 처리
        if prompt_name.endswith(('.yaml', '.yml')):
            file_path = prompt_dir / prompt_name
        else:
            # .yaml 우선 시도
            file_path = prompt_dir / f"{prompt_name}.yaml"
            if not file_path.exists():
                file_path = prompt_dir / f"{prompt_name}.yml"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: '{prompt_name}' in {prompt_dir}. "
                f"Available prompts: {cls.list_prompts(prompt_type)}"
            )

        return file_path


    @classmethod
    def load_prompt(cls, prompt_type: str, prompt_name: str) -> Dict[str, Any]:
        """
        프롬프트 파일 내용 로드 (YAML)

        Args:
            prompt_type: 프롬프트 타입 ("chatbot", "rag", "agentic")
            prompt_name: 프롬프트 파일명

        Returns:
            Dict[str, Any]: YAML 파싱 결과
                          일반적으로 title, description, system_prompt 등 포함

        Examples:
            >>> content = PromptsService.load_prompt("chatbot", "01-general")
            >>> print(content.keys())
            dict_keys(['title', 'description', 'system_prompt'])
        """
        file_path = cls.get_prompt_path(prompt_type, prompt_name)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)

        logger.debug(f"Loaded prompt: {prompt_type}/{prompt_name}")
        return content


    # =========================================================================
    # 프롬프트 정보 조회
    # =========================================================================

    @classmethod
    def get_prompt_info(cls, prompt_type: str, prompt_name: str) -> Dict[str, Any]:
        """
        프롬프트 메타 정보 반환

        Args:
            prompt_type: 프롬프트 타입 ("chatbot", "rag", "agentic")
            prompt_name: 프롬프트 파일명

        Returns:
            Dict[str, Any]: 프롬프트 메타 정보
                - name: 파일명
                - title: 제목
                - description: 설명
                - has_system_prompt: 시스템 프롬프트 포함 여부
                - file_path: 파일 경로

        Examples:
            >>> info = PromptsService.get_prompt_info("chatbot", "01-general")
            >>> print(info['title'])
            '일반 대화'
        """
        content = cls.load_prompt(prompt_type, prompt_name)
        file_path = cls.get_prompt_path(prompt_type, prompt_name)

        return {
            "name": prompt_name,
            "title": content.get("title", prompt_name),
            "description": content.get("description", ""),
            "has_system_prompt": (
                'system_prompt' in content or
                'system' in content or
                'template' in content
            ),
            "file_path": str(file_path),
        }


    @classmethod
    def get_all_prompts(cls) -> Dict[str, List[str]]:
        """
        모든 프롬프트 타입별 파일 목록 반환

        Returns:
            Dict[str, List[str]]: 타입별 프롬프트 목록
                {"chatbot": [...], "rag": [...], "agentic": [...]}

        Examples:
            >>> all_prompts = PromptsService.get_all_prompts()
            >>> for prompt_type, files in all_prompts.items():
            ...     print(f"{prompt_type}: {len(files)} files")
            chatbot: 4 files
            rag: 4 files
            agentic: 2 files
        """
        return {
            prompt_type: cls.list_prompts(prompt_type)
            for prompt_type in cls.PROMPT_DIRS.keys()
        }


    # =========================================================================
    # 템플릿 추출
    # =========================================================================

    @classmethod
    def get_template(cls, prompt_type: str, prompt_name: str) -> str:
        """
        프롬프트에서 시스템 프롬프트/템플릿만 추출

        Args:
            prompt_type: 프롬프트 타입 ("chatbot", "rag", "agentic")
            prompt_name: 프롬프트 파일명

        Returns:
            str: 시스템 프롬프트 또는 템플릿 문자열
                 찾을 수 없으면 빈 문자열

        Note:
            다음 순서로 키를 검색합니다:
            1. system_prompt
            2. system
            3. template
        """
        content = cls.load_prompt(prompt_type, prompt_name)

        # 우선순위: system_prompt > system > template
        if 'system_prompt' in content:
            return content['system_prompt']
        elif 'system' in content:
            return content['system']
        elif 'template' in content:
            return content['template']

        logger.warning(
            f"No template found in prompt: {prompt_type}/{prompt_name}. "
            f"Expected keys: system_prompt, system, or template"
        )
        return ""
