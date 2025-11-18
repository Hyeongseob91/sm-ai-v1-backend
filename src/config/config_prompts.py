"""
프롬프트 디렉토리 설정
"""

import os
from pathlib import Path


# =============================================================================
# 프롬프트 디렉토리 경로
# =============================================================================

# 기본 경로
_BASE_DIR = Path(__file__).parent.parent

# 프롬프트 루트 디렉토리
PROMPTS_DIR = _BASE_DIR / "data" / "prompts"

# 시스템별 프롬프트 디렉토리
CHATBOT_PROMPTS_DIR = PROMPTS_DIR / "chatbot"
RAG_PROMPTS_DIR = PROMPTS_DIR / "rag"
AGENTIC_PROMPTS_DIR = PROMPTS_DIR / "agentic"


# =============================================================================
# 프롬프트 파일 관리 함수
# =============================================================================

def get_prompt_files(prompt_type: str) -> list:
    """
    특정 타입의 프롬프트 파일 목록 반환

    Args:
        prompt_type: 'chatbot', 'rag', 'agentic'

    Returns:
        프롬프트 파일 이름 리스트 (확장자 제외)
    """
    prompt_dirs = {
        "chatbot": CHATBOT_PROMPTS_DIR,
        "rag": RAG_PROMPTS_DIR,
        "agentic": AGENTIC_PROMPTS_DIR,
    }

    if prompt_type not in prompt_dirs:
        raise ValueError(
            f"Invalid prompt_type: {prompt_type}. "
            f"Valid types: {list(prompt_dirs.keys())}"
        )

    prompt_dir = prompt_dirs[prompt_type]

    if not prompt_dir.exists():
        return []

    # YAML 파일 검색
    files = []
    for ext in [".yaml", ".yml"]:
        files.extend([f.stem for f in prompt_dir.glob(f"*{ext}")])

    return sorted(set(files))


def get_prompt_path(prompt_type: str, prompt_name: str) -> Path:
    """
    프롬프트 파일의 전체 경로 반환

    Args:
        prompt_type: 'chatbot', 'rag', 'agentic'
        prompt_name: 프롬프트 파일 이름 (확장자 없음)

    Returns:
        프롬프트 파일 Path 객체
    """
    prompt_dirs = {
        "chatbot": CHATBOT_PROMPTS_DIR,
        "rag": RAG_PROMPTS_DIR,
        "agentic": AGENTIC_PROMPTS_DIR,
    }

    if prompt_type not in prompt_dirs:
        raise ValueError(f"Invalid prompt_type: {prompt_type}")

    prompt_dir = prompt_dirs[prompt_type]

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
            f"Prompt file not found: {prompt_name} in {prompt_dir}"
        )

    return file_path


def load_prompt_content(prompt_type: str, prompt_name: str) -> dict:
    """
    프롬프트 파일 내용 로드

    Args:
        prompt_type: 'chatbot', 'rag', 'agentic'
        prompt_name: 프롬프트 파일 이름

    Returns:
        YAML 파싱 결과
    """
    import yaml

    file_path = get_prompt_path(prompt_type, prompt_name)

    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_all_prompts() -> dict:
    """
    모든 프롬프트 타입별 파일 목록 반환

    Returns:
        {"chatbot": [...], "rag": [...], "agentic": [...]}
    """
    return {
        "chatbot": get_prompt_files("chatbot"),
        "rag": get_prompt_files("rag"),
        "agentic": get_prompt_files("agentic"),
    }
