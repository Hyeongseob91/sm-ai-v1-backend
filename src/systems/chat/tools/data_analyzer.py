"""
Data Analyzer Tool
데이터 분석 및 통계 계산 도구
"""

from typing import Any, Dict, List, Optional
import logging

from .tool_registry import BaseTool
from ..models.function_call_model import ToolDefinition, ToolParameter, ParameterType

logger = logging.getLogger(__name__)


class DataAnalyzerTool(BaseTool):
    """데이터 분석 도구"""

    name = "data_analyzer"
    description = "데이터를 분석하고 통계를 계산합니다. 평균, 중앙값, 표준편차 등의 기술 통계와 상관분석을 수행할 수 있습니다."

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="data",
                    type=ParameterType.ARRAY,
                    description="분석할 데이터 배열",
                    required=True
                ),
                ToolParameter(
                    name="analysis_type",
                    type=ParameterType.STRING,
                    description="분석 유형",
                    required=True,
                    enum=["descriptive", "correlation", "distribution", "trend"]
                ),
                ToolParameter(
                    name="options",
                    type=ParameterType.OBJECT,
                    description="추가 분석 옵션",
                    required=False
                )
            ]
        )

    async def execute(
        self,
        data: List[Any],
        analysis_type: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """데이터 분석 실행"""
        import numpy as np

        options = options or {}

        try:
            # 숫자 데이터로 변환
            numeric_data = np.array([float(x) for x in data if x is not None])

            if len(numeric_data) == 0:
                return {"error": "No valid numeric data provided"}

            if analysis_type == "descriptive":
                return self._descriptive_stats(numeric_data)
            elif analysis_type == "correlation":
                return self._correlation_analysis(data, options)
            elif analysis_type == "distribution":
                return self._distribution_analysis(numeric_data)
            elif analysis_type == "trend":
                return self._trend_analysis(numeric_data)
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}

        except Exception as e:
            logger.error(f"Data analysis error: {e}")
            return {"error": str(e)}

    def _descriptive_stats(self, data) -> Dict[str, Any]:
        """기술 통계"""
        import numpy as np

        return {
            "count": len(data),
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "q1": float(np.percentile(data, 25)),
            "q3": float(np.percentile(data, 75)),
            "variance": float(np.var(data))
        }

    def _correlation_analysis(
        self,
        data: List[Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """상관관계 분석"""
        import numpy as np

        # 2차원 데이터 필요
        if not isinstance(data[0], (list, tuple)):
            return {"error": "Correlation requires 2D data"}

        try:
            arr = np.array(data, dtype=float)
            corr_matrix = np.corrcoef(arr.T)

            return {
                "correlation_matrix": corr_matrix.tolist(),
                "shape": arr.shape
            }
        except Exception as e:
            return {"error": f"Correlation analysis failed: {e}"}

    def _distribution_analysis(self, data) -> Dict[str, Any]:
        """분포 분석"""
        import numpy as np
        from scipy import stats

        try:
            # 정규성 검정
            stat, p_value = stats.shapiro(data) if len(data) <= 5000 else (None, None)

            # 히스토그램 데이터
            hist, bin_edges = np.histogram(data, bins='auto')

            return {
                "is_normal": p_value > 0.05 if p_value else None,
                "shapiro_stat": stat,
                "shapiro_pvalue": p_value,
                "skewness": float(stats.skew(data)),
                "kurtosis": float(stats.kurtosis(data)),
                "histogram": {
                    "counts": hist.tolist(),
                    "bins": bin_edges.tolist()
                }
            }
        except ImportError:
            # scipy가 없는 경우 기본 분석만
            return {
                "histogram": {
                    "counts": np.histogram(data, bins='auto')[0].tolist(),
                    "bins": np.histogram(data, bins='auto')[1].tolist()
                }
            }

    def _trend_analysis(self, data) -> Dict[str, Any]:
        """추세 분석"""
        import numpy as np

        x = np.arange(len(data))

        # 선형 회귀
        coeffs = np.polyfit(x, data, 1)
        trend_line = np.polyval(coeffs, x)

        # 이동 평균
        window = min(5, len(data) // 3) or 1
        moving_avg = np.convolve(data, np.ones(window)/window, mode='valid')

        return {
            "slope": float(coeffs[0]),
            "intercept": float(coeffs[1]),
            "trend_direction": "increasing" if coeffs[0] > 0 else "decreasing",
            "trend_line": trend_line.tolist(),
            "moving_average": moving_avg.tolist(),
            "moving_average_window": window
        }
