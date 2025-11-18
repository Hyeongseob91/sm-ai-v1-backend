"""
벡터 스토어 서비스
다양한 벡터 DB 백엔드 지원
"""

from typing import List, Tuple, Optional, Any
from enum import Enum
import logging

from src.models.base_models import Document, DocumentChunk
from .embedding_service import EmbeddingService
from ..constants import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class VectorStoreType(str, Enum):
    """벡터 스토어 타입"""
    FAISS = "faiss"
    CHROMA = "chroma"
    WEAVIATE = "weaviate"


class VectorStoreService:
    """
    벡터 스토어 서비스
    다양한 벡터 DB 백엔드 추상화
    """

    def __init__(
        self,
        store_type: VectorStoreType = VectorStoreType.FAISS,
        embedding_service: Optional[EmbeddingService] = None,
        **kwargs
    ):
        self.store_type = store_type
        self.embedding_service = embedding_service or EmbeddingService()
        self._store = None
        self._kwargs = kwargs

    async def initialize(self) -> None:
        """서비스 초기화"""
        await self.embedding_service.initialize()

    async def create_from_documents(
        self,
        documents: List[Document]
    ) -> None:
        """문서들로부터 벡터 스토어 생성"""
        await self.initialize()

        from langchain.schema import Document as LCDocument

        # LangChain Document로 변환
        lc_docs = [
            LCDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in documents
        ]

        embeddings = self.embedding_service.get_langchain_embeddings()

        if self.store_type == VectorStoreType.FAISS:
            self._store = await self._create_faiss(lc_docs, embeddings)
        elif self.store_type == VectorStoreType.CHROMA:
            self._store = await self._create_chroma(lc_docs, embeddings)
        elif self.store_type == VectorStoreType.WEAVIATE:
            self._store = await self._create_weaviate(lc_docs, embeddings)

        logger.info(
            f"Created {self.store_type.value} vector store "
            f"with {len(documents)} documents"
        )

    async def _create_faiss(self, documents, embeddings):
        """FAISS 벡터 스토어 생성"""
        from langchain_community.vectorstores import FAISS
        return FAISS.from_documents(documents, embeddings)

    async def _create_chroma(self, documents, embeddings):
        """Chroma 벡터 스토어 생성"""
        from langchain_community.vectorstores import Chroma

        persist_directory = self._kwargs.get("persist_directory", "./.chroma")

        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )

    async def _create_weaviate(self, documents, embeddings):
        """Weaviate 벡터 스토어 생성"""
        from langchain_community.vectorstores import Weaviate
        import weaviate

        url = self._kwargs.get("weaviate_url", "http://localhost:8080")
        index_name = self._kwargs.get("index_name", "Document")

        client = weaviate.Client(url=url)

        return Weaviate.from_documents(
            documents=documents,
            embedding=embeddings,
            client=client,
            index_name=index_name
        )

    async def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """유사도 검색"""
        if self._store is None:
            raise RuntimeError("Vector store not initialized")

        results = self._store.similarity_search_with_score(query, k=k)

        chunks = []
        for doc, score in results:
            # 점수 정규화 (스토어마다 다를 수 있음)
            normalized_score = self._normalize_score(score)

            chunk = DocumentChunk(
                content=doc.page_content,
                metadata=doc.metadata,
                score=normalized_score
            )
            chunks.append((chunk, normalized_score))

        return chunks

    def _normalize_score(self, score: float) -> float:
        """점수 정규화 (0-1 범위)"""
        if self.store_type == VectorStoreType.FAISS:
            # FAISS는 거리를 반환 (낮을수록 좋음)
            return 1 / (1 + score)
        else:
            # Chroma, Weaviate 등은 유사도 반환
            return score

    async def add_documents(self, documents: List[Document]) -> None:
        """기존 스토어에 문서 추가"""
        if self._store is None:
            await self.create_from_documents(documents)
            return

        from langchain.schema import Document as LCDocument

        lc_docs = [
            LCDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in documents
        ]

        self._store.add_documents(lc_docs)
        logger.info(f"Added {len(documents)} documents to vector store")

    async def delete(self, ids: Optional[List[str]] = None) -> None:
        """문서 삭제"""
        if self._store is None:
            return

        if hasattr(self._store, 'delete'):
            self._store.delete(ids)
            logger.info(f"Deleted documents: {ids}")

    async def save(self, path: str) -> None:
        """벡터 스토어 저장"""
        if self._store is None:
            return

        if self.store_type == VectorStoreType.FAISS:
            self._store.save_local(path)
            logger.info(f"Saved FAISS index to {path}")
        elif self.store_type == VectorStoreType.CHROMA:
            # Chroma는 자동 저장
            pass

    async def load(self, path: str) -> None:
        """벡터 스토어 로드"""
        await self.initialize()
        embeddings = self.embedding_service.get_langchain_embeddings()

        if self.store_type == VectorStoreType.FAISS:
            from langchain_community.vectorstores import FAISS
            self._store = FAISS.load_local(
                path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded FAISS index from {path}")

    def get_store(self) -> Any:
        """내부 벡터 스토어 객체 반환"""
        return self._store
