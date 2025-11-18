"""
Naive Retriever 구현
단순 벡터 유사도 검색
"""

from typing import List, Tuple, Optional
import logging
import numpy as np

from .base_retriever import BaseRetriever, RetrieverConfig
from src.models.base_models import Document, DocumentChunk
from ..constants import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class NaiveRetriever(BaseRetriever):
    """
    단순 벡터 기반 리트리버
    FAISS를 사용한 벡터 유사도 검색
    """

    def __init__(
        self,
        config: Optional[RetrieverConfig] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ):
        super().__init__(config)
        self.embedding_model = embedding_model
        self._embeddings = None
        self._vector_store = None
        self._initialized = False

    async def _initialize_embeddings(self):
        """임베딩 모델 초기화"""
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info(f"Initialized embeddings: {self.embedding_model}")

    async def add_documents(self, documents: List[Document]) -> None:
        """문서를 벡터 스토어에 추가"""
        await self._initialize_embeddings()

        from langchain_community.vectorstores import FAISS
        from langchain.schema import Document as LCDocument

        # LangChain Document로 변환
        lc_docs = [
            LCDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in documents
        ]

        # FAISS 벡터 스토어 생성
        self._vector_store = FAISS.from_documents(
            documents=lc_docs,
            embedding=self._embeddings
        )
        self._initialized = True

        logger.info(f"Added {len(documents)} documents to vector store")

    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """벡터 유사도 기반 검색"""
        if not self._initialized or self._vector_store is None:
            raise RuntimeError("Retriever not initialized. Call add_documents first.")

        k = k or self.config.final_k

        # 유사도 검색 수행
        results = self._vector_store.similarity_search_with_score(
            query=query,
            k=k
        )

        # DocumentChunk로 변환
        chunks = []
        for doc, score in results:
            # FAISS는 거리를 반환하므로 유사도로 변환 (낮을수록 좋음)
            # L2 거리를 유사도로 변환
            similarity = 1 / (1 + score)

            chunk = DocumentChunk(
                content=doc.page_content,
                metadata=doc.metadata,
                score=similarity
            )
            chunks.append((chunk, similarity))

        # 점수 기준 필터링
        filtered = self._filter_by_score(chunks)

        logger.debug(f"Retrieved {len(filtered)} documents for query: {query[:50]}...")
        return filtered

    async def clear(self) -> None:
        """벡터 스토어 초기화"""
        self._vector_store = None
        self._initialized = False
        logger.info("Cleared vector store")

    def get_vector_store(self):
        """내부 벡터 스토어 반환 (고급 사용)"""
        return self._vector_store

    async def save(self, path: str) -> None:
        """벡터 스토어 저장"""
        if self._vector_store:
            self._vector_store.save_local(path)
            logger.info(f"Saved vector store to {path}")

    async def load(self, path: str) -> None:
        """벡터 스토어 로드"""
        await self._initialize_embeddings()

        from langchain_community.vectorstores import FAISS
        self._vector_store = FAISS.load_local(
            path,
            self._embeddings,
            allow_dangerous_deserialization=True
        )
        self._initialized = True
        logger.info(f"Loaded vector store from {path}")
