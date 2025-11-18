"""
리트리버 추상 기본 클래스
모든 검색 전략의 기반
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from src.models.base_models import Document, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    """리트리버 설정"""
    retrieval_k: int = 10        # 초기 검색 문서 수
    final_k: int = 5             # 최종 반환 문서 수
    score_threshold: float = 0.0  # 최소 유사도 점수


class BaseRetriever(ABC):
    """리트리버 추상 기본 클래스"""

    def __init__(self, config: Optional[RetrieverConfig] = None):
        self.config = config or RetrieverConfig()

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        쿼리에 대한 관련 문서 검색

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수 (None이면 config.final_k 사용)

        Returns:
            (DocumentChunk, score) 튜플 리스트, 점수 내림차순 정렬
        """
        pass

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """
        인덱스에 문서 추가

        Args:
            documents: 추가할 문서 리스트
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """인덱스 초기화"""
        pass

    def _filter_by_score(
        self,
        results: List[Tuple[DocumentChunk, float]]
    ) -> List[Tuple[DocumentChunk, float]]:
        """점수 임계값으로 필터링"""
        if self.config.score_threshold > 0:
            return [
                (doc, score) for doc, score in results
                if score >= self.config.score_threshold
            ]
        return results

    def get_documents_only(
        self,
        results: List[Tuple[DocumentChunk, float]]
    ) -> List[DocumentChunk]:
        """점수 제외하고 문서만 반환"""
        return [doc for doc, _ in results]
