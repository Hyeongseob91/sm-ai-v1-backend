"""
청킹 전략 구현
문서를 작은 청크로 분할하는 다양한 전략
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import logging

from src.models.base_models import Document
from ..constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class ChunkingStrategy(ABC):
    """청킹 전략 추상 기본 클래스"""

    @abstractmethod
    def split(self, documents: List[Document]) -> List[Document]:
        """문서를 청크로 분할"""
        pass


class RecursiveChunker(ChunkingStrategy):
    """재귀적 문자 기반 청킹"""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(self, documents: List[Document]) -> List[Document]:
        """문서를 재귀적으로 분할"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )

        chunks = []
        for doc in documents:
            # LangChain Document로 변환
            from langchain.schema import Document as LCDocument
            lc_doc = LCDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            )

            # 분할
            split_docs = splitter.split_documents([lc_doc])

            # 다시 우리 Document로 변환
            for i, split_doc in enumerate(split_docs):
                chunk = Document(
                    page_content=split_doc.page_content,
                    metadata={
                        **split_doc.metadata,
                        "chunk_index": i
                    }
                )
                chunks.append(chunk)

        logger.info(
            f"Split {len(documents)} documents into {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks


class SemanticChunker(ChunkingStrategy):
    """의미 기반 청킹 (임베딩 활용)"""

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        breakpoint_threshold: float = 0.5
    ):
        self.embedding_model = embedding_model
        self.breakpoint_threshold = breakpoint_threshold

    def split(self, documents: List[Document]) -> List[Document]:
        """의미적 유사성 기반 분할"""
        try:
            from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            logger.warning(
                "SemanticChunker requires langchain_experimental. "
                "Falling back to RecursiveChunker."
            )
            return RecursiveChunker().split(documents)

        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        splitter = LCSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=self.breakpoint_threshold * 100
        )

        chunks = []
        for doc in documents:
            from langchain.schema import Document as LCDocument
            lc_doc = LCDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            )

            split_docs = splitter.split_documents([lc_doc])

            for i, split_doc in enumerate(split_docs):
                chunk = Document(
                    page_content=split_doc.page_content,
                    metadata={
                        **split_doc.metadata,
                        "chunk_index": i,
                        "chunking_method": "semantic"
                    }
                )
                chunks.append(chunk)

        logger.info(f"Semantic chunking: {len(documents)} docs -> {len(chunks)} chunks")
        return chunks


class CharacterChunker(ChunkingStrategy):
    """단순 문자 기반 청킹"""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Document]) -> List[Document]:
        """단순 문자 수 기반 분할"""
        from langchain.text_splitter import CharacterTextSplitter

        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n"
        )

        chunks = []
        for doc in documents:
            from langchain.schema import Document as LCDocument
            lc_doc = LCDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            )

            split_docs = splitter.split_documents([lc_doc])

            for i, split_doc in enumerate(split_docs):
                chunk = Document(
                    page_content=split_doc.page_content,
                    metadata={
                        **split_doc.metadata,
                        "chunk_index": i
                    }
                )
                chunks.append(chunk)

        return chunks


def create_chunker(
    strategy: str = "recursive",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    **kwargs
) -> ChunkingStrategy:
    """청킹 전략 팩토리 함수"""
    strategies = {
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "character": CharacterChunker,
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            f"Available: {list(strategies.keys())}"
        )

    if strategy == "semantic":
        return strategies[strategy](**kwargs)
    else:
        return strategies[strategy](
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
