"""
문서 로더 구현
다양한 파일 형식을 지원하는 문서 로딩 모듈
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
import logging

from src.models.base_models import Document
from ..constants import SUPPORTED_FILE_TYPES

logger = logging.getLogger(__name__)


class DocumentLoader(ABC):
    """문서 로더 추상 기본 클래스"""

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """파일을 로드하여 Document 리스트 반환"""
        pass

    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """해당 파일 확장자 지원 여부"""
        pass


class PDFLoader(DocumentLoader):
    """PDF 문서 로더"""

    def __init__(self, loader_type: str = "pdfplumber"):
        """
        Args:
            loader_type: 'pdfplumber', 'pypdf', 'unstructured'
        """
        self.loader_type = loader_type

    def load(self, file_path: str) -> List[Document]:
        """PDF 파일 로드"""
        documents = []
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if self.loader_type == "pdfplumber":
                documents = self._load_with_pdfplumber(file_path)
            elif self.loader_type == "pypdf":
                documents = self._load_with_pypdf(file_path)
            else:
                documents = self._load_with_unstructured(file_path)

            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            return documents

        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise

    def _load_with_pdfplumber(self, file_path: str) -> List[Document]:
        """pdfplumber를 사용한 PDF 로딩"""
        import pdfplumber

        documents = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": i + 1,
                            "total_pages": len(pdf.pages)
                        }
                    ))
        return documents

    def _load_with_pypdf(self, file_path: str) -> List[Document]:
        """PyPDF를 사용한 PDF 로딩"""
        from pypdf import PdfReader

        documents = []
        reader = PdfReader(file_path)

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": i + 1,
                        "total_pages": len(reader.pages)
                    }
                ))
        return documents

    def _load_with_unstructured(self, file_path: str) -> List[Document]:
        """Unstructured를 사용한 PDF 로딩"""
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(file_path)
        documents = []

        for element in elements:
            if hasattr(element, 'text') and element.text.strip():
                documents.append(Document(
                    page_content=element.text,
                    metadata={
                        "source": file_path,
                        "element_type": type(element).__name__
                    }
                ))
        return documents

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == ".pdf"


class TextLoader(DocumentLoader):
    """텍스트 파일 로더"""

    def load(self, file_path: str) -> List[Document]:
        """텍스트 파일 로드"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return [Document(
            page_content=content,
            metadata={"source": file_path}
        )]

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in [".txt", ".md"]


class DocumentLoaderFactory:
    """문서 로더 팩토리"""

    _loaders = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
    }

    @classmethod
    def get_loader(
        cls,
        file_path: str,
        loader_type: Optional[str] = None
    ) -> DocumentLoader:
        """파일 확장자에 맞는 로더 반환"""
        ext = Path(file_path).suffix.lower()

        if ext not in cls._loaders:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported types: {SUPPORTED_FILE_TYPES}"
            )

        loader_class = cls._loaders[ext]

        if ext == ".pdf" and loader_type:
            return loader_class(loader_type=loader_type)

        return loader_class()

    @classmethod
    def load_document(
        cls,
        file_path: str,
        loader_type: Optional[str] = None
    ) -> List[Document]:
        """편의 메서드: 파일 로드"""
        loader = cls.get_loader(file_path, loader_type)
        return loader.load(file_path)
