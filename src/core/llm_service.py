"""
LLM Service - 통합 LLM 인터페이스 및 관리

다양한 LLM 공급자(OpenAI, vLLM, Anthropic, Google 등)를
통합된 인터페이스로 제공하는 서비스 모듈입니다.

주요 구성요소:
    - create_llm_router: LLM 인스턴스 생성 라우터
    - Factory Methods: 각 공급자별 LLM 생성 함수
    - LLMService: 통합 LLM 서비스 클래스
    - create_embeddings: 임베딩 모델 생성 함수
"""

from typing import Optional, Any, List
import logging

from langchain_core.language_models import BaseChatModel

from src.config.config_model import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_TOKENS,
    MODEL_CONFIG,
    LLMProvider,
    OPENAI_API_KEY,
    VLLM_BASE_URL,
    VLLM_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Router - 모델 라우팅 함수
# =============================================================================

def create_llm_router(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    streaming: bool = False,
    **kwargs
) -> BaseChatModel:
    """
    LLM 인스턴스 생성 라우터

    MODEL_CONFIG를 기반으로 적절한 공급자의 LLM을 생성합니다.
    설정에 없는 모델의 경우 이름 패턴으로 추론하거나 기본값(OpenAI)을 사용합니다.
    에러 발생 시 OpenAI로 fallback하여 안정성을 보장합니다.

    Args:
        model: 모델 이름 (예: "gpt-4o", "gemini-2.5-pro", "claude-3-5-sonnet-20241022")
               MODEL_CONFIG에 정의된 모델명을 사용하거나, 공급자별 패턴으로 자동 인식
        temperature: 생성 다양성 제어 (0.0=결정적/일관된 응답, 1.0=창의적/다양한 응답)
                     일반적으로 0.0~0.7 사이 값을 권장
        max_tokens: 최대 생성 토큰 수 (응답 길이 제한)
                    모델별 최대값이 다름 (GPT-4: 8192, Claude: 4096 등)
        streaming: 스트리밍 모드 활성화 여부
                   True: 토큰 단위로 실시간 응답 (채팅 UI에 적합)
                   False: 전체 응답 완료 후 반환
        **kwargs: 추가 설정 (모델별 특수 파라미터)
                  예: top_p, frequency_penalty, presence_penalty 등

    Returns:
        BaseChatModel: LangChain LLM 인스턴스
                      invoke(), stream() 등의 메서드 사용 가능

    Raises:
        에러 발생 시 로깅 후 OpenAI fallback 반환 (예외를 직접 raise하지 않음)

    Examples:
        >>> llm = create_llm_router("gpt-4o", temperature=0.7)
        >>> response = await llm.ainvoke("Hello")

        >>> llm = create_llm_router("gemini-2.5-pro", streaming=True)
        >>> async for chunk in llm.astream("Tell me a story"):
        ...     print(chunk.content)
    """

    try:
        # ---------------------------------------------------------------------
        # 1. MODEL_CONFIG에서 공급자 조회
        # ---------------------------------------------------------------------
        config = MODEL_CONFIG.get(model)
        provider = None

        if config:
            # 설정에 정의된 모델
            provider = config["provider"]
            logger.debug(f"Model '{model}' found in MODEL_CONFIG, provider: {provider.value}")
        else:
            # ---------------------------------------------------------------------
            # 2. 설정에 없는 경우 이름 패턴으로 추론 (Fallback)
            # ---------------------------------------------------------------------
            if model.startswith("gpt") or model.startswith("o1"):
                provider = LLMProvider.OPENAI
            elif model.startswith("claude"):
                provider = LLMProvider.ANTHROPIC
            elif model.startswith("gemini"):
                provider = LLMProvider.GOOGLE
            elif model.startswith("vllm:"):
                provider = LLMProvider.VLLM
            else:
                provider = LLMProvider.OPENAI  # 기본값
                logger.warning(
                    f"Unknown model provider for '{model}', defaulting to OpenAI. "
                    f"Consider adding this model to MODEL_CONFIG."
                )

        # ---------------------------------------------------------------------
        # 3. 공급자별 Factory Method 호출
        # ---------------------------------------------------------------------
        if provider == LLMProvider.OPENAI:
            return _create_openai(model, temperature, max_tokens, streaming, **kwargs)

        elif provider == LLMProvider.ANTHROPIC:
            return _create_anthropic(model, temperature, max_tokens, streaming, **kwargs)

        elif provider == LLMProvider.GOOGLE:
            return _create_google(model, temperature, max_tokens, streaming, **kwargs)

        elif provider == LLMProvider.VLLM:
            # vLLM 모델명에서 'vllm:' 접두사 제거
            real_model = model.replace("vllm:", "") if model.startswith("vllm:") else model
            return _create_vllm(real_model, temperature, max_tokens, streaming, **kwargs)

        # Fallback (도달하지 않아야 함)
        return _create_openai(model, temperature, max_tokens, streaming, **kwargs)

    except Exception as e:
        # ---------------------------------------------------------------------
        # 4. 에러 발생 시 OpenAI로 Fallback
        # ---------------------------------------------------------------------
        logger.error(
            f"LLM 생성 실패 - model: {model}, error: {e}. "
            f"OpenAI ({DEFAULT_MODEL})로 fallback합니다."
        )
        return _create_openai(DEFAULT_MODEL, temperature, max_tokens, streaming, **kwargs)


# =============================================================================
# Factory Methods - OpenAI
# =============================================================================

def _create_openai(
    model: str,
    temperature: float,
    max_tokens: int,
    streaming: bool,
    **kwargs
) -> BaseChatModel:
    """
    OpenAI LLM 인스턴스 생성

    OpenAI API를 사용하여 ChatGPT 모델 인스턴스를 생성합니다.
    GPT-4o, GPT-4-turbo, GPT-3.5-turbo 등의 모델을 지원합니다.

    Args:
        model: OpenAI 모델 이름
               지원 모델: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo,
                         o1-preview, o1-mini 등
        temperature: 생성 다양성 (0.0~2.0, 기본값 권장: 0.0~1.0)
        max_tokens: 최대 생성 토큰 수
                    gpt-4o: 최대 16,384 / gpt-3.5-turbo: 최대 4,096
        streaming: 스트리밍 모드 활성화
        **kwargs: 추가 파라미터
                  - top_p: 누적 확률 샘플링 (0.0~1.0)
                  - frequency_penalty: 반복 억제 (-2.0~2.0)
                  - presence_penalty: 새 주제 유도 (-2.0~2.0)

    Returns:
        ChatOpenAI: OpenAI LLM 인스턴스

    Note:
        API 키는 config_model.py의 OPENAI_API_KEY에서 가져옵니다.
        환경변수 OPENAI_API_KEY로 설정하세요.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        openai_api_key=OPENAI_API_KEY,
        **kwargs
    )

    logger.debug(f"Created OpenAI LLM: {model}, temp={temperature}, max_tokens={max_tokens}")
    return llm


# =============================================================================
# Factory Methods - vLLM (로컬 모델)
# =============================================================================

def _create_vllm(
    model: str,
    temperature: float,
    max_tokens: int,
    streaming: bool,
    **kwargs
) -> BaseChatModel:
    """
    vLLM 로컬 LLM 인스턴스 생성

    vLLM 서버에 연결하여 로컬 LLM 인스턴스를 생성합니다.
    OpenAI 호환 API를 사용하므로 ChatOpenAI 클라이언트를 재활용합니다.

    Args:
        model: vLLM에서 서빙 중인 모델 이름
               예: Qwen/Qwen2.5-14B-Instruct, meta-llama/Llama-2-7b-chat-hf
        temperature: 생성 다양성 (모델별 지원 범위가 다를 수 있음)
        max_tokens: 최대 생성 토큰 수 (모델의 컨텍스트 길이 이내로 설정)
        streaming: 스트리밍 모드 활성화
        **kwargs: 추가 파라미터
                  - top_p, top_k: 샘플링 파라미터
                  - repetition_penalty: 반복 억제

    Returns:
        ChatOpenAI: vLLM 서버에 연결된 LLM 인스턴스

    Note:
        - vLLM 서버가 실행 중이어야 합니다.
        - 서버 주소는 VLLM_BASE_URL 환경변수로 설정 (기본: http://localhost:8000/v1)
        - API 키는 VLLM_API_KEY로 설정 (기본: EMPTY)
    """
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

    logger.debug(f"Created vLLM: {model} at {VLLM_BASE_URL}, temp={temperature}")
    
    return llm


# =============================================================================
# Factory Methods - Anthropic (Claude)
# =============================================================================

def _create_anthropic(
    model: str,
    temperature: float,
    max_tokens: int,
    streaming: bool,
    **kwargs
) -> BaseChatModel:
    """
    Anthropic (Claude) LLM 인스턴스 생성

    Anthropic API를 사용하여 Claude 모델 인스턴스를 생성합니다.
    Claude 3 시리즈(Opus, Sonnet, Haiku)를 지원합니다.

    Args:
        model: Anthropic 모델 이름
               지원 모델: claude-3-5-sonnet-20241022, claude-3-opus-20240229,
                         claude-3-sonnet-20240229, claude-3-haiku-20240307
        temperature: 생성 다양성 (0.0~1.0)
                     Claude는 1.0 초과 값을 지원하지 않음
        max_tokens: 최대 생성 토큰 수
                    Claude 3: 최대 4,096
        streaming: 스트리밍 모드 활성화
        **kwargs: 추가 파라미터
                  - top_p: 누적 확률 샘플링
                  - top_k: 상위 k개 토큰만 샘플링

    Returns:
        ChatAnthropic: Anthropic LLM 인스턴스
        ImportError 시 ChatOpenAI (OpenAI fallback)

    Note:
        - langchain_anthropic 패키지가 필요합니다.
        - API 키는 ANTHROPIC_API_KEY 환경변수로 설정하세요.
        - 패키지 미설치 시 OpenAI로 fallback합니다.
    """
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

        logger.debug(f"Created Anthropic LLM: {model}, temp={temperature}")
        return llm

    except ImportError:
        logger.warning(
            "langchain_anthropic 패키지가 설치되지 않았습니다. "
            f"OpenAI ({DEFAULT_MODEL})로 fallback합니다. "
            "설치: pip install langchain-anthropic"
        )
        return _create_openai(DEFAULT_MODEL, temperature, max_tokens, streaming, **kwargs)


# =============================================================================
# Factory Methods - Google (Gemini)
# =============================================================================

def _create_google(
    model: str,
    temperature: float,
    max_tokens: int,
    streaming: bool,
    **kwargs
) -> BaseChatModel:
    """
    Google (Gemini) LLM 인스턴스 생성

    Google AI API를 사용하여 Gemini 모델 인스턴스를 생성합니다.
    Gemini 1.5/2.0 Pro, Flash 등의 모델을 지원합니다.

    Args:
        model: Google 모델 이름
               지원 모델: gemini-2.5-pro, gemini-2.5-flash,
                         gemini-1.5-pro, gemini-1.5-flash
        temperature: 생성 다양성 (0.0~1.0)
        max_tokens: 최대 생성 토큰 수 (max_output_tokens로 전달)
                    Gemini 1.5 Pro: 최대 8,192
        streaming: 스트리밍 모드 활성화
        **kwargs: 추가 파라미터
                  - top_p: 누적 확률 샘플링
                  - top_k: 상위 k개 토큰만 샘플링

    Returns:
        ChatGoogleGenerativeAI: Google LLM 인스턴스
        ImportError 시 ChatOpenAI (OpenAI fallback)

    Note:
        - langchain_google_genai 패키지가 필요합니다.
        - API 키는 GOOGLE_API_KEY 환경변수로 설정하세요.
        - 패키지 미설치 시 OpenAI로 fallback합니다.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            streaming=streaming,
            google_api_key=GOOGLE_API_KEY,
            **kwargs
        )

        logger.debug(f"Created Google LLM: {model}, temp={temperature}")
        return llm

    except ImportError:
        logger.warning(
            "langchain_google_genai 패키지가 설치되지 않았습니다. "
            f"OpenAI ({DEFAULT_MODEL})로 fallback합니다. "
            "설치: pip install langchain-google-genai"
        )
        return _create_openai(DEFAULT_MODEL, temperature, max_tokens, streaming, **kwargs)


# =============================================================================
# LLM Service 클래스 - 통합 인터페이스
# =============================================================================
class LLMService:
    """
    통합 LLM 서비스 인터페이스

    다양한 LLM 공급자(OpenAI, Anthropic, Google, vLLM)를
    동일한 인터페이스로 제공하는 래퍼 클래스입니다.

    주요 기능:
        - Lazy 초기화: 실제 사용 시점에 LLM 인스턴스 생성 (메모리 효율)
        - 동적 설정 변경: 런타임에 모델/온도/토큰 수 변경 가능
        - 스트리밍 지원: 실시간 응답 스트리밍 (채팅 UI에 적합)
        - 도구 바인딩: Function calling / Tool use 지원

    Attributes:
        model: 현재 설정된 모델 이름
        temperature: 생성 다양성 설정값
        max_tokens: 최대 생성 토큰 수
        streaming: 스트리밍 모드 활성화 여부

    Examples:
        기본 사용:
            >>> service = LLMService(model="gpt-4o", temperature=0.7)
            >>> response = await service.invoke("Hello, how are you?")
            >>> print(response)

        스트리밍 사용:
            >>> service = LLMService(model="gpt-4o", streaming=True)
            >>> async for chunk in service.stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)

        설정 변경:
            >>> service = LLMService()
            >>> service.update_config(model="claude-3-5-sonnet-20241022", temperature=0.5)
            >>> response = await service.invoke("New model test")

        도구 바인딩:
            >>> tools = [{"name": "calculator", "description": "..."}]
            >>> llm_with_tools = service.with_tools(tools)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        streaming: bool = False
    ):
        """
        LLMService 초기화

        Args:
            model: 사용할 모델 이름 (기본값: config의 DEFAULT_MODEL)
                   예: "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-2.5-pro"
            temperature: 생성 다양성 (기본값: config의 DEFAULT_TEMPERATURE)
                        0.0=결정적, 1.0=창의적
            max_tokens: 최대 생성 토큰 수 (기본값: config의 MAX_TOKENS)
            streaming: 스트리밍 모드 (기본값: False)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        self._llm: Optional[BaseChatModel] = None


    def get_llm(self) -> BaseChatModel:
        """
        LLM 인스턴스 반환 (Lazy 초기화)

        처음 호출 시 LLM 인스턴스를 생성하고, 이후 호출에서는 캐시된 인스턴스를 반환합니다.
        update_config() 호출 후에는 새 인스턴스가 생성됩니다.

        Returns:
            BaseChatModel: LangChain LLM 인스턴스
                          invoke(), ainvoke(), stream(), astream() 등 사용 가능

        Note:
            내부적으로 create_llm_router()를 호출하여 적절한 공급자의 LLM을 생성합니다.
        """
        if self._llm is None:
            self._llm = create_llm_router(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                streaming=self.streaming
            )
        return self._llm


    async def invoke(self, prompt: str) -> str:
        """
        비동기 LLM 호출

        프롬프트를 LLM에 전달하고 전체 응답을 반환합니다.
        스트리밍이 필요 없는 일반적인 요청에 사용합니다.

        Args:
            prompt: LLM에 전달할 프롬프트 텍스트
                    단일 문자열 또는 시스템/사용자 메시지 형식 모두 가능

        Returns:
            str: LLM의 응답 텍스트

        Examples:
            >>> response = await service.invoke("What is Python?")
            >>> print(response)
            "Python is a high-level programming language..."
        """
        llm = self.get_llm()
        response = await llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)


    async def stream(self, prompt: str):
        """
        스트리밍 LLM 호출

        프롬프트를 LLM에 전달하고 응답을 토큰 단위로 실시간 반환합니다.
        채팅 UI에서 타이핑 효과를 구현할 때 유용합니다.

        Args:
            prompt: LLM에 전달할 프롬프트 텍스트

        Yields:
            str: 응답의 각 청크(토큰 또는 토큰 그룹)

        Examples:
            >>> async for chunk in service.stream("Tell me a joke"):
            ...     print(chunk, end="", flush=True)
            ...
            "Why did the programmer quit his job?..."
        """
        llm = self.get_llm()

        async for chunk in llm.astream(prompt):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                yield content


    def with_tools(self, tools: List[dict]) -> BaseChatModel:
        """
        도구 바인딩

        LLM에 도구(함수)를 바인딩하여 Function Calling / Tool Use를 활성화합니다.
        바인딩된 LLM은 필요에 따라 도구 호출을 생성합니다.

        Args:
            tools: 도구 정의 리스트
                   각 도구는 name, description, parameters를 포함하는 딕셔너리

        Returns:
            BaseChatModel: 도구가 바인딩된 LLM 인스턴스
                          원본 LLM이 도구를 지원하지 않으면 원본 반환

        Examples:
            >>> tools = [{
            ...     "name": "get_weather",
            ...     "description": "Get weather for a location",
            ...     "parameters": {"location": {"type": "string"}}
            ... }]
            >>> llm_with_tools = service.with_tools(tools)
            >>> response = await llm_with_tools.ainvoke("What's the weather in Seoul?")

        Note:
            모든 모델이 도구 바인딩을 지원하지는 않습니다.
            지원하지 않는 모델의 경우 경고 로그와 함께 원본 LLM을 반환합니다.
        """
        llm = self.get_llm()

        if hasattr(llm, 'bind_tools'):
            return llm.bind_tools(tools)
        else:
            logger.warning(
                f"Model '{self.model}' does not support tool binding. "
                f"Returning original LLM instance."
            )
            return llm


    def update_config(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> None:
        """
        설정 업데이트 (LLM 재생성)

        런타임에 모델, 온도, 토큰 수를 변경합니다.
        변경 후 다음 get_llm() 호출 시 새 인스턴스가 생성됩니다.

        Args:
            model: 새 모델 이름 (None이면 유지)
            temperature: 새 온도 값 (None이면 유지)
            max_tokens: 새 최대 토큰 수 (None이면 유지)

        Examples:
            >>> service = LLMService(model="gpt-4o")
            >>>
            >>> # 모델만 변경
            >>> service.update_config(model="gpt-4o-mini")
            >>>
            >>> # 여러 설정 동시 변경
            >>> service.update_config(
            ...     model="claude-3-5-sonnet-20241022",
            ...     temperature=0.8,
            ...     max_tokens=2048
            ... )

        Note:
            이 메서드 호출 후 캐시된 LLM 인스턴스가 삭제되고,
            다음 요청 시 새 설정으로 인스턴스가 재생성됩니다.
        """
        if model:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if max_tokens:
            self.max_tokens = max_tokens

        # 캐시된 LLM 인스턴스 삭제 → 다음 호출 시 재생성
        self._llm = None

        logger.info(
            f"LLM config updated - model: {self.model}, "
            f"temperature: {self.temperature}, max_tokens: {self.max_tokens}"
        )


# =============================================================================
# Embeddings 생성 함수
# =============================================================================

def create_embeddings(
    model: str = "BAAI/bge-m3",
    device: str = "cpu"
) -> Any:
    """
    임베딩 모델 생성

    텍스트를 벡터로 변환하는 임베딩 모델 인스턴스를 생성합니다.
    RAG(Retrieval-Augmented Generation) 파이프라인에서 사용됩니다.

    Args:
        model: HuggingFace 임베딩 모델 이름
               권장 모델:
               - BAAI/bge-m3: 다국어 지원, 고성능 (기본값)
               - sentence-transformers/all-MiniLM-L6-v2: 경량, 빠름
               - intfloat/multilingual-e5-large: 다국어, 대형
        device: 실행 디바이스
               - "cpu": CPU 사용 (기본값)
               - "cuda": GPU 사용 (NVIDIA)
               - "mps": Apple Silicon GPU

    Returns:
        HuggingFaceEmbeddings: 임베딩 모델 인스턴스
                              embed_documents(), embed_query() 메서드 사용 가능

    Examples:
        >>> embeddings = create_embeddings(model="BAAI/bge-m3", device="cuda")
        >>>
        >>> # 단일 텍스트 임베딩
        >>> vector = embeddings.embed_query("Hello, world!")
        >>>
        >>> # 여러 텍스트 임베딩
        >>> vectors = embeddings.embed_documents(["Text 1", "Text 2"])

    Note:
        - 처음 사용 시 모델 다운로드가 필요합니다.
        - GPU 사용 시 torch와 CUDA가 설치되어 있어야 합니다.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    logger.debug(f"Created embeddings: {model} on {device}")
    return embeddings
