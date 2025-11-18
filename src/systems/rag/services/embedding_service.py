"""
임베딩 서비스
다양한 임베딩 모델을 통합 관리
"""

from typing import List, Optional
import logging
import numpy as np

from ..constants import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    임베딩 생성 및 관리 서비스
    다양한 임베딩 모델 백엔드 지원
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cpu",
        normalize: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._embeddings = None
        self._dimension: Optional[int] = None

    async def initialize(self) -> None:
        """임베딩 모델 초기화"""
        if self._embeddings is not None:
            return

        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": self.normalize}
            )

            # 차원 확인
            test_embedding = self._embeddings.embed_query("test")
            self._dimension = len(test_embedding)

            logger.info(
                f"Initialized embeddings: {self.model_name} "
                f"(dim={self._dimension}, device={self.device})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    async def embed_query(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        if self._embeddings is None:
            await self.initialize()

        return self._embeddings.embed_query(text)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 임베딩"""
        if self._embeddings is None:
            await self.initialize()

        return self._embeddings.embed_documents(texts)

    async def similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간 코사인 유사도"""
        emb1 = await self.embed_query(text1)
        emb2 = await self.embed_query(text2)

        # 코사인 유사도 계산
        emb1_np = np.array(emb1)
        emb2_np = np.array(emb2)

        similarity = np.dot(emb1_np, emb2_np) / (
            np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np)
        )

        return float(similarity)

    @property
    def dimension(self) -> int:
        """임베딩 차원"""
        return self._dimension or EMBEDDING_DIMENSION

    def get_langchain_embeddings(self):
        """LangChain 호환 임베딩 객체 반환"""
        return self._embeddings


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI 임베딩 서비스"""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        super().__init__(model_name)
        self.api_key = api_key

    async def initialize(self) -> None:
        """OpenAI 임베딩 초기화"""
        if self._embeddings is not None:
            return

        try:
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.api_key
            )

            # 차원 확인
            test_embedding = self._embeddings.embed_query("test")
            self._dimension = len(test_embedding)

            logger.info(f"Initialized OpenAI embeddings: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise


def create_embedding_service(
    provider: str = "huggingface",
    model_name: Optional[str] = None,
    **kwargs
) -> EmbeddingService:
    """임베딩 서비스 팩토리 함수"""
    if provider == "openai":
        return OpenAIEmbeddingService(
            model_name=model_name or "text-embedding-3-small",
            **kwargs
        )
    else:
        return EmbeddingService(
            model_name=model_name or DEFAULT_EMBEDDING_MODEL,
            **kwargs
        )
