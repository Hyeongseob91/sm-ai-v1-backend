"""
Hybrid Retriever 구현
벡터 검색 + BM25 스파스 검색 결합
"""

from typing import List, Tuple, Optional, Dict
import logging
import numpy as np

from .base_retriever import BaseRetriever, RetrieverConfig
from .naive_retriever import NaiveRetriever
from src.models.base_models import Document, DocumentChunk
from ..constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_WEIGHT,
    DEFAULT_BM25_WEIGHT
)

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    하이브리드 리트리버
    Dense (벡터) + Sparse (BM25) 검색 결합
    """

    def __init__(
        self,
        config: Optional[RetrieverConfig] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT
    ):
        super().__init__(config)
        self.embedding_model = embedding_model
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        # 내부 컴포넌트
        self._naive_retriever = NaiveRetriever(config, embedding_model)
        self._bm25 = None
        self._documents: List[Document] = []
        self._initialized = False

    async def add_documents(self, documents: List[Document]) -> None:
        """문서를 양쪽 인덱스에 추가"""
        self._documents = documents

        # Dense 인덱스 (FAISS)
        await self._naive_retriever.add_documents(documents)

        # Sparse 인덱스 (BM25)
        self._initialize_bm25(documents)

        self._initialized = True
        logger.info(f"Added {len(documents)} documents to hybrid retriever")

    def _initialize_bm25(self, documents: List[Document]) -> None:
        """BM25 인덱스 초기화"""
        from rank_bm25 import BM25Okapi

        # 토큰화
        tokenized_docs = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        self._bm25 = BM25Okapi(tokenized_docs)
        logger.info("Initialized BM25 index")

    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """하이브리드 검색 수행"""
        if not self._initialized:
            raise RuntimeError("Retriever not initialized. Call add_documents first.")

        k = k or self.config.final_k
        retrieval_k = self.config.retrieval_k

        # 1. Dense 검색 (FAISS)
        dense_results = await self._naive_retriever.retrieve(query, k=retrieval_k)

        # 2. Sparse 검색 (BM25)
        sparse_results = self._bm25_search(query, k=retrieval_k)

        # 3. 점수 융합 (Reciprocal Rank Fusion)
        fused_results = self._fuse_results(
            dense_results,
            sparse_results,
            k=k
        )

        # 점수 기준 필터링
        filtered = self._filter_by_score(fused_results)

        logger.debug(f"Hybrid search returned {len(filtered)} documents")
        return filtered

    def _bm25_search(
        self,
        query: str,
        k: int
    ) -> List[Tuple[DocumentChunk, float]]:
        """BM25 검색"""
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # 상위 k개 인덱스
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self._documents[idx]
                # 정규화된 점수 (0-1 범위)
                normalized_score = scores[idx] / (scores.max() + 1e-6)

                chunk = DocumentChunk(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=normalized_score
                )
                results.append((chunk, normalized_score))

        return results

    def _fuse_results(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
        k: int
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Reciprocal Rank Fusion (RRF) 적용
        가중치를 적용한 순위 기반 점수 융합
        """
        # 문서 내용을 키로 사용하여 점수 합산
        doc_scores: Dict[str, Tuple[DocumentChunk, float]] = {}
        rrf_k = 60  # RRF 상수

        # Dense 결과 처리
        for rank, (chunk, score) in enumerate(dense_results):
            key = chunk.content[:200]  # 내용 일부를 키로 사용
            rrf_score = self.vector_weight / (rrf_k + rank + 1)

            if key in doc_scores:
                existing_chunk, existing_score = doc_scores[key]
                doc_scores[key] = (existing_chunk, existing_score + rrf_score)
            else:
                doc_scores[key] = (chunk, rrf_score)

        # Sparse 결과 처리
        for rank, (chunk, score) in enumerate(sparse_results):
            key = chunk.content[:200]
            rrf_score = self.bm25_weight / (rrf_k + rank + 1)

            if key in doc_scores:
                existing_chunk, existing_score = doc_scores[key]
                doc_scores[key] = (existing_chunk, existing_score + rrf_score)
            else:
                doc_scores[key] = (chunk, rrf_score)

        # 점수 기준 정렬
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results[:k]

    async def clear(self) -> None:
        """인덱스 초기화"""
        await self._naive_retriever.clear()
        self._bm25 = None
        self._documents = []
        self._initialized = False
        logger.info("Cleared hybrid retriever")
