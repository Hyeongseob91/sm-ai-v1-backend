"""
LLM Service - 통합 LLM 인터페이스 및 관리
다양한 LLM 공급자(OpenAI, vLLM, Anthropic 등) 지원
"""

from typing import Optional, Any, List
import logging

from src.config.config_model import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_TOKENS,
    OPENAI_API_KEY,
    VLLM_ENABLED,
    VLLM_BASE_URL,
    VLLM_MODEL,
    VLLM_API_KEY,
    ANTHROPIC_API_KEY,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LLM 생성 함수
# =============================================================================

def create_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    streaming: bool = False,
    **kwargs
) -> Any:
    """
    LLM 인스턴스 생성

    Args:
        model: 모델 이름
        temperature: 생성 온도
        max_tokens: 최대 토큰 수
        streaming: 스트리밍 모드
        **kwargs: 추가 설정

    Returns:
        LangChain LLM 인스턴스
    """
    # vLLM 사용 (로컬)
    if model.startswith("vllm:") or (VLLM_ENABLED and model == VLLM_MODEL):
        return _create_vllm(
            model=model.replace("vllm:", "") if model.startswith("vllm:") else VLLM_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **kwargs
        )

    # Anthropic (Claude)
    if model.startswith("claude"):
        return _create_anthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **kwargs
        )

    # OpenAI (기본)
    return _create_openai(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        **kwargs
    )


def _create_openai(
    model: str,
    temperature: float,
    max_tokens: int,
    streaming: bool,
    **kwargs
) -> Any:
    """OpenAI LLM 생성"""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        openai_api_key=OPENAI_API_KEY,
        **kwargs
    )

    logger.debug(f"Created OpenAI LLM: {model}")
    return llm


def _create_vllm(
    model: str,
    temperature: float,
    max_tokens: int,
    streaming: bool,
    **kwargs
) -> Any:
    """vLLM 로컬 LLM 생성"""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        openai_api_key=VLLM_API_KEY,
        openai_api_base=VLLM_BASE_URL,
        **kwargs
    )

    logger.debug(f"Created vLLM: {model} at {VLLM_BASE_URL}")
    return llm


def _create_anthropic(
    model: str,
    temperature: float,
    max_tokens: int,
    streaming: bool,
    **kwargs
) -> Any:
    """Anthropic (Claude) LLM 생성"""
    try:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            anthropic_api_key=ANTHROPIC_API_KEY,
            **kwargs
        )

        logger.debug(f"Created Anthropic LLM: {model}")
        return llm

    except ImportError:
        logger.warning("langchain_anthropic not installed, falling back to OpenAI")
        return _create_openai(
            model=DEFAULT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **kwargs
        )


# =============================================================================
# LLM Service 클래스
# =============================================================================

class LLMService:
    """
    통합 LLM 서비스
    다양한 LLM 공급자를 동일한 인터페이스로 제공
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        streaming: bool = False
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        self._llm = None

    def get_llm(self) -> Any:
        """LLM 인스턴스 반환 (Lazy 초기화)"""
        if self._llm is None:
            self._llm = create_llm(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                streaming=self.streaming
            )
        return self._llm

    async def invoke(self, prompt: str) -> str:
        """동기 호출"""
        llm = self.get_llm()
        response = await llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    async def stream(self, prompt: str):
        """스트리밍 호출"""
        llm = self.get_llm()
        async for chunk in llm.astream(prompt):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                yield content

    def with_tools(self, tools: List[dict]) -> Any:
        """도구 바인딩"""
        llm = self.get_llm()
        if hasattr(llm, 'bind_tools'):
            return llm.bind_tools(tools)
        else:
            logger.warning(f"Model {self.model} does not support tool binding")
            return llm

    def update_config(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> None:
        """설정 업데이트 (LLM 재생성)"""
        if model:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if max_tokens:
            self.max_tokens = max_tokens

        # LLM 재생성
        self._llm = None
        logger.info(f"LLM config updated: model={self.model}")


# =============================================================================
# Embeddings 생성
# =============================================================================

def create_embeddings(
    model: str = "BAAI/bge-m3",
    device: str = "cpu"
) -> Any:
    """
    임베딩 모델 생성

    Args:
        model: 임베딩 모델 이름
        device: 디바이스 (cpu/cuda)

    Returns:
        LangChain Embeddings 인스턴스
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    logger.debug(f"Created embeddings: {model} on {device}")
    return embeddings
