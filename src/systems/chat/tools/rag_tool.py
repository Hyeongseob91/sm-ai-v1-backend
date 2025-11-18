"""
RAG Tool
RAG 시스템을 호출하여 문서 검색을 수행하는 도구
"""

from typing import Any, Dict, List, Optional
import logging

from .tool_registry import BaseTool
from ..models.function_call_model import ToolDefinition, ToolParameter, ParameterType

logger = logging.getLogger(__name__)


class RAGTool(BaseTool):
    """RAG 검색 도구"""

    name = "rag_tool"
    description = "문서 데이터베이스에서 질문과 관련된 정보를 검색합니다. 업로드된 PDF나 문서에서 맥락 정보를 찾을 때 사용합니다."

    def __init__(self, rag_system=None):
        """
        Args:
            rag_system: RAGSystemChain 인스턴스 (런타임에 주입)
        """
        self._rag_system = rag_system

    def set_rag_system(self, rag_system) -> None:
        """RAG 시스템 설정"""
        self._rag_system = rag_system

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="검색할 질문 또는 키워드",
                    required=True
                ),
                ToolParameter(
                    name="num_results",
                    type=ParameterType.INTEGER,
                    description="반환할 결과 수",
                    required=False
                ),
                ToolParameter(
                    name="include_scores",
                    type=ParameterType.BOOLEAN,
                    description="유사도 점수 포함 여부",
                    required=False
                )
            ]
        )

    async def execute(
        self,
        query: str,
        num_results: int = 5,
        include_scores: bool = True
    ) -> Dict[str, Any]:
        """RAG 검색 실행"""

        # RAG 시스템이 설정되지 않은 경우
        if self._rag_system is None:
            return {
                "error": "RAG system not initialized. Please upload a document first.",
                "results": []
            }

        try:
            # RAG 시스템에서 검색
            # 실제 구현은 rag_system_chain.py에서 수행
            results = await self._retrieve_documents(query, num_results)

            if not results:
                return {
                    "query": query,
                    "num_results": 0,
                    "results": [],
                    "message": "No relevant documents found."
                }

            # 결과 포맷팅
            formatted_results = []
            for i, (doc, score) in enumerate(results):
                result = {
                    "rank": i + 1,
                    "content": doc.content,
                    "source": doc.metadata.get('source', 'unknown'),
                    "page": doc.metadata.get('page', None)
                }

                if include_scores:
                    result["score"] = round(score, 4)

                formatted_results.append(result)

            return {
                "query": query,
                "num_results": len(formatted_results),
                "results": formatted_results
            }

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return {
                "error": str(e),
                "results": []
            }

    async def _retrieve_documents(
        self,
        query: str,
        num_results: int
    ) -> List:
        """문서 검색 (내부 메서드)"""

        # RAG 시스템의 리트리버 사용
        if hasattr(self._rag_system, 'retriever'):
            return await self._rag_system.retriever.retrieve(query, k=num_results)

        # 직접 검색 메서드가 있는 경우
        if hasattr(self._rag_system, 'retrieve'):
            return await self._rag_system.retrieve(query, k=num_results)

        raise RuntimeError("RAG system does not have a valid retriever")

    def get_context_for_llm(
        self,
        results: List[Dict[str, Any]]
    ) -> str:
        """LLM에 전달할 컨텍스트 문자열 생성"""
        if not results:
            return "관련 문서를 찾지 못했습니다."

        context_parts = []
        for result in results:
            part = f"[문서 {result['rank']}]\n{result['content']}"
            if result.get('source'):
                part += f"\n출처: {result['source']}"
            if result.get('page'):
                part += f" (페이지 {result['page']})"
            context_parts.append(part)

        return "\n\n---\n\n".join(context_parts)
