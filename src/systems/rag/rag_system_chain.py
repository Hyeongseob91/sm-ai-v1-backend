"""
RAG System Chain
문서 기반 검색 및 생성 시스템
"""

from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
import logging
import yaml
from pathlib import Path

from src.models.base_models import Document, DocumentChunk, RAGContext
from src.core.llm_service import create_llm_router
from src.core.prompts_service import PromptsService
from src.config.config_model import DEFAULT_MODEL, DEFAULT_TEMPERATURE

from .constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_FINAL_K,
    DEFAULT_VECTOR_WEIGHT,
    DEFAULT_BM25_WEIGHT,
)
from .processors import DocumentLoaderFactory, create_chunker
from .retrievers import NaiveRetriever, HybridRetriever, RetrieverConfig
from .services import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """RAG 시스템 설정"""
    # LLM 설정
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE

    # 문서 처리
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    loader_type: str = "pdfplumber"
    chunking_strategy: str = "recursive"

    # 검색 설정
    use_hybrid_search: bool = True
    use_reranking: bool = False
    retrieval_k: int = DEFAULT_RETRIEVAL_K
    final_k: int = DEFAULT_FINAL_K
    vector_weight: float = DEFAULT_VECTOR_WEIGHT
    bm25_weight: float = DEFAULT_BM25_WEIGHT

    # 프롬프트
    prompt_name: str = "01-pdf-rag"


class RAGSystemChain:
    """
    RAG 시스템 메인

    문서 로딩, 청킹, 임베딩, 검색, 응답 생성을 통합 관리
    """

    def __init__(
        self,
        session_id: str,
        config: Optional[RAGConfig] = None,
        **kwargs
    ):
        self.session_id = session_id
        self.config = config or RAGConfig(**kwargs)

        # 컴포넌트들
        self._retriever = None
        self._llm = None
        self._embedding_service = None

        # 상태
        self.document_name: Optional[str] = None
        self.chunk_count: int = 0
        self._documents: List[Document] = []
        self._initialized = False

    async def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        문서 로드 및 인덱싱

        Args:
            file_path: 문서 파일 경로

        Returns:
            인덱싱 결과
        """
        logger.info(f"Ingesting document: {file_path}")

        try:
            # 1. 문서 로드
            documents = DocumentLoaderFactory.load_document(
                file_path,
                loader_type=self.config.loader_type
            )

            if not documents:
                raise ValueError("No content extracted from document")

            # 2. 청킹
            chunker = create_chunker(
                strategy=self.config.chunking_strategy,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            chunks = chunker.split(documents)

            # 3. 리트리버 초기화 및 문서 추가
            retriever_config = RetrieverConfig(
                retrieval_k=self.config.retrieval_k,
                final_k=self.config.final_k
            )

            if self.config.use_hybrid_search:
                self._retriever = HybridRetriever(
                    config=retriever_config,
                    vector_weight=self.config.vector_weight,
                    bm25_weight=self.config.bm25_weight
                )
            else:
                self._retriever = NaiveRetriever(config=retriever_config)

            await self._retriever.add_documents(chunks)

            # 상태 업데이트
            self.document_name = Path(file_path).name
            self.chunk_count = len(chunks)
            self._documents = chunks
            self._initialized = True

            logger.info(f"Ingested {len(chunks)} chunks from {self.document_name}")

            return {
                "success": True,
                "document_name": self.document_name,
                "chunk_count": self.chunk_count,
                "session_id": self.session_id
            }

        except Exception as e:
            logger.error(f"Ingest error: {e}")
            raise

    async def query(self, question: str) -> 'RAGResponse':
        """
        질의응답 (동기)

        Args:
            question: 질문

        Returns:
            RAGResponse 객체
        """
        if not self._initialized:
            raise RuntimeError("No document ingested. Call ingest_document first.")

        # 1. 검색
        results = await self._retriever.retrieve(question, k=self.config.final_k)

        # 2. 컨텍스트 생성
        context = self._build_context(results)

        # 3. 프롬프트 로드
        prompt_template = self._load_prompt()

        # 4. LLM 호출
        if self._llm is None:
            self._llm = create_llm_router(
                model=self.config.model,
                temperature=self.config.temperature
            )

        full_prompt = prompt_template.format(
            context=context.context,
            question=question
        )

        response = await self._llm.ainvoke(full_prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        return RAGResponse(
            answer=answer,
            sources=context.sources,
            question=question
        )

    async def query_stream(self, question: str) -> AsyncIterator[str]:
        """
        질의응답 (스트리밍)

        Args:
            question: 질문

        Yields:
            응답 청크
        """
        if not self._initialized:
            raise RuntimeError("No document ingested. Call ingest_document first.")

        # 1. 검색
        results = await self._retriever.retrieve(question, k=self.config.final_k)

        # 2. 컨텍스트 생성
        context = self._build_context(results)

        # 3. 프롬프트 로드
        prompt_template = self._load_prompt()

        # 4. LLM 스트리밍
        if self._llm is None:
            self._llm = create_llm_router(
                model=self.config.model,
                temperature=self.config.temperature,
                streaming=True
            )

        full_prompt = prompt_template.format(
            context=context.context,
            question=question
        )

        async for chunk in self._llm.astream(full_prompt):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                yield content

    def _build_context(
        self,
        results: List[tuple]
    ) -> RAGContext:
        """검색 결과로부터 컨텍스트 생성"""
        documents = []
        for doc, score in results:
            if isinstance(doc, DocumentChunk):
                documents.append(doc)
            else:
                documents.append(DocumentChunk(
                    content=doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    metadata=doc.metadata if hasattr(doc, 'metadata') else {},
                    score=score
                ))

        return RAGContext.from_documents(
            question="",  # placeholder
            documents=documents
        )

    def _load_prompt(self) -> str:
        """프롬프트 템플릿 로드"""
        try:
            content = PromptsService.load_prompt("rag", self.config.prompt_name)
            template = content.get("template", "")

            # 기본 템플릿 형식 확인
            if "{context}" not in template or "{question}" not in template:
                template = """다음 컨텍스트를 바탕으로 질문에 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:"""

            return template

        except Exception as e:
            logger.warning(f"Failed to load prompt, using default: {e}")
            return """다음 컨텍스트를 바탕으로 질문에 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:"""

    async def update_config(self, **kwargs) -> None:
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.config, key) and value is not None:
                setattr(self.config, key, value)

        # 리트리버 설정 업데이트
        if self._retriever:
            if 'retrieval_k' in kwargs or 'final_k' in kwargs:
                self._retriever.config.retrieval_k = self.config.retrieval_k
                self._retriever.config.final_k = self.config.final_k

        logger.info(f"Config updated: {kwargs}")

    async def clear(self) -> None:
        """세션 정리"""
        if self._retriever:
            await self._retriever.clear()

        self._documents = []
        self.document_name = None
        self.chunk_count = 0
        self._initialized = False

        logger.info(f"Cleared RAG session: {self.session_id}")

    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[tuple]:
        """직접 검색 (tool용)"""
        if not self._initialized:
            return []

        return await self._retriever.retrieve(query, k=k)


@dataclass
class RAGResponse:
    """RAG 응답"""
    answer: str
    sources: List[Dict[str, Any]]
    question: str = ""


def create_rag_system(session_id: str, **kwargs) -> RAGSystemChain:
    """
    RAG 시스템 인스턴스 생성 함수

    Args:
        session_id: 세션 ID
        **kwargs: RAGConfig 인자

    Returns:
        RAGSystemChain 인스턴스
    """
    return RAGSystemChain(session_id=session_id, **kwargs)
