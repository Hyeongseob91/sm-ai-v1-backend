"""
페르소나 프롬프트 로더
YAML 파일에서 페르소나 설정을 로드
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


class PersonaLoader:
    """페르소나 프롬프트 로더"""

    def __init__(self, prompts_dir: str):
        """
        Args:
            prompts_dir: 프롬프트 파일이 있는 디렉토리 경로
        """
        self.prompts_dir = Path(prompts_dir)

    def load(self, persona_name: str) -> Dict[str, Any]:
        """
        페르소나 설정 로드

        Args:
            persona_name: 페르소나 파일 이름 (확장자 제외)

        Returns:
            페르소나 설정 딕셔너리
        """
        # 파일 경로 결정
        if persona_name.endswith('.yaml') or persona_name.endswith('.yml'):
            file_path = self.prompts_dir / persona_name
        else:
            file_path = self.prompts_dir / f"{persona_name}.yaml"

        if not file_path.exists():
            # .yml 확장자로 시도
            file_path = self.prompts_dir / f"{persona_name}.yml"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Persona file not found: {persona_name} "
                f"in {self.prompts_dir}"
            )

        # YAML 로드
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                persona = yaml.safe_load(f)

            logger.info(f"Loaded persona: {persona_name}")
            return persona

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse persona file: {e}")
            raise

    def get_system_prompt(self, persona_name: str) -> str:
        """시스템 프롬프트만 반환"""
        persona = self.load(persona_name)
        return persona.get('system_prompt', persona.get('system', ''))

    def get_available_personas(self) -> List[str]:
        """사용 가능한 페르소나 목록 반환"""
        personas = []

        for file_path in self.prompts_dir.glob("*.yaml"):
            personas.append(file_path.stem)

        for file_path in self.prompts_dir.glob("*.yml"):
            if file_path.stem not in personas:
                personas.append(file_path.stem)

        return sorted(personas)

    def get_persona_info(self, persona_name: str) -> Dict[str, Any]:
        """페르소나 메타 정보 반환"""
        persona = self.load(persona_name)

        return {
            "name": persona_name,
            "title": persona.get('title', persona_name),
            "description": persona.get('description', ''),
            "has_system_prompt": 'system_prompt' in persona or 'system' in persona,
            "has_tools": 'tools' in persona,
            "metadata": persona.get('metadata', {})
        }


def load_persona(
    persona_name: str,
    prompts_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    편의 함수: 페르소나 로드

    Args:
        persona_name: 페르소나 이름
        prompts_dir: 프롬프트 디렉토리 (기본값: src/data/prompts/agentic)
    """
    if prompts_dir is None:
        # 기본 경로
        base_dir = Path(__file__).parent.parent.parent.parent
        prompts_dir = base_dir / "data" / "prompts" / "agentic"

    loader = PersonaLoader(str(prompts_dir))
    return loader.load(persona_name)


def get_prompt_template(persona: Dict[str, Any]) -> str:
    """
    페르소나에서 프롬프트 템플릿 추출

    Returns:
        완전한 프롬프트 템플릿 문자열
    """
    parts = []

    # 시스템 프롬프트
    if 'system_prompt' in persona:
        parts.append(persona['system_prompt'])
    elif 'system' in persona:
        parts.append(persona['system'])

    # 추가 지시사항
    if 'instructions' in persona:
        parts.append("\n\n## Instructions\n" + persona['instructions'])

    # 예시
    if 'examples' in persona:
        parts.append("\n\n## Examples\n" + persona['examples'])

    return "\n".join(parts)
