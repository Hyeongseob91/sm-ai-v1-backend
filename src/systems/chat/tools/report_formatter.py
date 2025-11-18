"""
Report Formatter Tool
분석 결과를 보고서 형식으로 포맷팅
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import logging
from datetime import datetime

from .tool_registry import BaseTool
from ..models.function_call_model import ToolDefinition, ToolParameter, ParameterType
from ..constants import REPORT_FORMATS, DEFAULT_REPORT_FORMAT

logger = logging.getLogger(__name__)


class ReportFormatterTool(BaseTool):
    """보고서 포맷팅 도구"""

    name = "report_formatter"
    description = "분석 결과를 구조화된 보고서 형식으로 포맷팅합니다. Markdown, HTML 형식을 지원합니다."

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="content",
                    type=ParameterType.OBJECT,
                    description="보고서 내용 (title, sections, summary 등)",
                    required=True
                ),
                ToolParameter(
                    name="format",
                    type=ParameterType.STRING,
                    description="출력 형식",
                    required=False,
                    enum=REPORT_FORMATS
                ),
                ToolParameter(
                    name="options",
                    type=ParameterType.OBJECT,
                    description="추가 포맷팅 옵션",
                    required=False
                )
            ]
        )

    async def execute(
        self,
        content: Dict[str, Any],
        format: str = DEFAULT_REPORT_FORMAT,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """보고서 포맷팅 실행"""
        options = options or {}

        try:
            if format == "markdown":
                formatted = self._format_markdown(content, options)
            elif format == "html":
                formatted = self._format_html(content, options)
            else:
                return {"error": f"Unsupported format: {format}"}

            return {
                "success": True,
                "format": format,
                "content": formatted,
                "length": len(formatted)
            }

        except Exception as e:
            logger.error(f"Report formatting error: {e}")
            return {"error": str(e)}

    def _format_markdown(
        self,
        content: Dict[str, Any],
        options: Dict[str, Any]
    ) -> str:
        """Markdown 형식으로 포맷팅"""
        lines = []

        # 제목
        title = content.get('title', 'Analysis Report')
        lines.append(f"# {title}")
        lines.append("")

        # 메타 정보
        if options.get('include_meta', True):
            lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            lines.append("")

        # 요약
        if 'summary' in content:
            lines.append("## Summary")
            lines.append("")
            lines.append(content['summary'])
            lines.append("")

        # 섹션들
        if 'sections' in content:
            for section in content['sections']:
                lines.append(f"## {section.get('title', 'Section')}")
                lines.append("")

                # 본문
                if 'content' in section:
                    lines.append(section['content'])
                    lines.append("")

                # 데이터 테이블
                if 'data' in section:
                    lines.extend(self._format_table_md(section['data']))
                    lines.append("")

                # 하위 항목
                if 'items' in section:
                    for item in section['items']:
                        lines.append(f"- {item}")
                    lines.append("")

        # 결론
        if 'conclusion' in content:
            lines.append("## Conclusion")
            lines.append("")
            lines.append(content['conclusion'])
            lines.append("")

        # 참조
        if 'references' in content:
            lines.append("## References")
            lines.append("")
            for i, ref in enumerate(content['references'], 1):
                lines.append(f"{i}. {ref}")

        return "\n".join(lines)

    def _format_table_md(self, data: List[Dict[str, Any]]) -> List[str]:
        """Markdown 테이블 생성"""
        if not data:
            return []

        lines = []

        # 헤더
        headers = list(data[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # 데이터
        for row in data:
            values = [str(row.get(h, '')) for h in headers]
            lines.append("| " + " | ".join(values) + " |")

        return lines

    def _format_html(
        self,
        content: Dict[str, Any],
        options: Dict[str, Any]
    ) -> str:
        """HTML 형식으로 포맷팅"""
        html_parts = []

        # 스타일
        if options.get('include_style', True):
            html_parts.append("""
<style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
    h1 { color: #333; border-bottom: 2px solid #4CAF50; }
    h2 { color: #555; margin-top: 30px; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #4CAF50; color: white; }
    tr:nth-child(even) { background-color: #f2f2f2; }
    .meta { color: #888; font-style: italic; margin-bottom: 20px; }
    .summary { background-color: #f9f9f9; padding: 15px; border-left: 4px solid #4CAF50; }
</style>
""")

        # 제목
        title = content.get('title', 'Analysis Report')
        html_parts.append(f"<h1>{title}</h1>")

        # 메타 정보
        if options.get('include_meta', True):
            html_parts.append(
                f'<p class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>'
            )

        # 요약
        if 'summary' in content:
            html_parts.append('<h2>Summary</h2>')
            html_parts.append(f'<div class="summary">{content["summary"]}</div>')

        # 섹션들
        if 'sections' in content:
            for section in content['sections']:
                html_parts.append(f"<h2>{section.get('title', 'Section')}</h2>")

                if 'content' in section:
                    html_parts.append(f"<p>{section['content']}</p>")

                if 'data' in section:
                    html_parts.append(self._format_table_html(section['data']))

                if 'items' in section:
                    html_parts.append("<ul>")
                    for item in section['items']:
                        html_parts.append(f"<li>{item}</li>")
                    html_parts.append("</ul>")

        # 결론
        if 'conclusion' in content:
            html_parts.append('<h2>Conclusion</h2>')
            html_parts.append(f"<p>{content['conclusion']}</p>")

        return "\n".join(html_parts)

    def _format_table_html(self, data: List[Dict[str, Any]]) -> str:
        """HTML 테이블 생성"""
        if not data:
            return ""

        headers = list(data[0].keys())

        html = ["<table>", "<thead><tr>"]
        for h in headers:
            html.append(f"<th>{h}</th>")
        html.append("</tr></thead>")

        html.append("<tbody>")
        for row in data:
            html.append("<tr>")
            for h in headers:
                html.append(f"<td>{row.get(h, '')}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")

        return "\n".join(html)
