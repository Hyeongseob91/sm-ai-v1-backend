"""
Chart Generator Tool
데이터 시각화 및 차트 생성 도구
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

from .tool_registry import BaseTool
from ..models.function_call_model import ToolDefinition, ToolParameter, ParameterType
from ..constants import CHART_OUTPUT_DIR

logger = logging.getLogger(__name__)


class ChartGeneratorTool(BaseTool):
    """차트 생성 도구"""

    name = "chart_generator"
    description = "데이터를 시각화하여 차트를 생성합니다. 라인, 바, 파이, 산점도 등 다양한 차트 유형을 지원합니다."

    def __init__(self, output_dir: str = CHART_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="data",
                    type=ParameterType.OBJECT,
                    description="차트 데이터 (x, y 값 포함)",
                    required=True
                ),
                ToolParameter(
                    name="chart_type",
                    type=ParameterType.STRING,
                    description="차트 유형",
                    required=True,
                    enum=["line", "bar", "pie", "scatter", "histogram", "box"]
                ),
                ToolParameter(
                    name="title",
                    type=ParameterType.STRING,
                    description="차트 제목",
                    required=False
                ),
                ToolParameter(
                    name="options",
                    type=ParameterType.OBJECT,
                    description="추가 차트 옵션 (색상, 레이블 등)",
                    required=False
                )
            ]
        )

    async def execute(
        self,
        data: Dict[str, Any],
        chart_type: str,
        title: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """차트 생성 실행"""
        import matplotlib.pyplot as plt
        import uuid

        options = options or {}
        title = title or f"{chart_type.capitalize()} Chart"

        try:
            # Figure 생성
            fig, ax = plt.subplots(figsize=options.get('figsize', (10, 6)))

            # 차트 유형별 생성
            if chart_type == "line":
                self._create_line_chart(ax, data, options)
            elif chart_type == "bar":
                self._create_bar_chart(ax, data, options)
            elif chart_type == "pie":
                self._create_pie_chart(ax, data, options)
            elif chart_type == "scatter":
                self._create_scatter_chart(ax, data, options)
            elif chart_type == "histogram":
                self._create_histogram(ax, data, options)
            elif chart_type == "box":
                self._create_box_chart(ax, data, options)
            else:
                return {"error": f"Unknown chart type: {chart_type}"}

            # 제목 설정
            ax.set_title(title, fontsize=options.get('title_fontsize', 14))

            # 그리드
            if options.get('grid', True):
                ax.grid(True, alpha=0.3)

            # 파일 저장
            filename = f"{chart_type}_{uuid.uuid4().hex[:8]}.png"
            filepath = self.output_dir / filename

            plt.tight_layout()
            plt.savefig(filepath, dpi=options.get('dpi', 150))
            plt.close()

            logger.info(f"Chart saved: {filepath}")

            return {
                "success": True,
                "filepath": str(filepath),
                "filename": filename,
                "chart_type": chart_type,
                "title": title
            }

        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            plt.close()
            return {"error": str(e)}

    def _create_line_chart(self, ax, data, options):
        """라인 차트 생성"""
        x = data.get('x', list(range(len(data.get('y', [])))))
        y = data.get('y', [])

        ax.plot(x, y,
                color=options.get('color', 'blue'),
                linewidth=options.get('linewidth', 2),
                marker=options.get('marker', 'o'))

        if 'xlabel' in options:
            ax.set_xlabel(options['xlabel'])
        if 'ylabel' in options:
            ax.set_ylabel(options['ylabel'])

    def _create_bar_chart(self, ax, data, options):
        """바 차트 생성"""
        x = data.get('x', data.get('labels', []))
        y = data.get('y', data.get('values', []))

        colors = options.get('colors', None)
        ax.bar(x, y, color=colors)

        if options.get('rotate_labels', False):
            plt.xticks(rotation=45, ha='right')

    def _create_pie_chart(self, ax, data, options):
        """파이 차트 생성"""
        labels = data.get('labels', [])
        values = data.get('values', [])

        ax.pie(values, labels=labels,
               autopct=options.get('autopct', '%1.1f%%'),
               colors=options.get('colors', None))

    def _create_scatter_chart(self, ax, data, options):
        """산점도 생성"""
        x = data.get('x', [])
        y = data.get('y', [])

        ax.scatter(x, y,
                   c=options.get('color', 'blue'),
                   alpha=options.get('alpha', 0.6),
                   s=options.get('size', 50))

    def _create_histogram(self, ax, data, options):
        """히스토그램 생성"""
        values = data.get('values', data.get('y', []))

        ax.hist(values,
                bins=options.get('bins', 'auto'),
                color=options.get('color', 'blue'),
                alpha=options.get('alpha', 0.7),
                edgecolor='black')

    def _create_box_chart(self, ax, data, options):
        """박스 플롯 생성"""
        if 'groups' in data:
            # 여러 그룹
            ax.boxplot(data['groups'], labels=data.get('labels', None))
        else:
            # 단일 데이터
            values = data.get('values', data.get('y', []))
            ax.boxplot(values)
