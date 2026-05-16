# src/algorithms/preprocess/__init__.py
"""
数据预处理模块
包含缺失值处理、异常值处理、正向化、标准化、降维等全流程预处理算法
所有类均继承 PreprocessBase，统一提供 get_output_data() 接口
"""

from __future__ import annotations

import abc
from typing import Any

import pandas as pd

from ..base import BaseMethod, MethodCategory, MethodResult
from ...utils.logging_config import get_logger

logger = get_logger("auto_eval.preprocess")


# ============================================================
#  预处理算法中间基类
# ============================================================

class PreprocessBase(BaseMethod, abc.ABC):
    """
    所有预处理算法的中间基类。

    在 BaseMethod 基础上额外提供：
    - get_output_data()：直接获取处理后的 DataFrame
    - 宽松的输入校验（允许缺失值、允许少量非数值列）
    - _extract_numeric()：自动提取数值列

    子类无需重复实现上述逻辑，只需专注于核心算法。
    """

    CATEGORY = MethodCategory.PREPROCESS

    # ── 覆盖基类输入校验（预处理阶段允许缺失值）──────────────

    def validate_input(self, data: pd.DataFrame) -> None:
        """
        预处理专用输入校验，不对缺失值报错。

        Raises
        ------
        TypeError  : data 不是 DataFrame
        ValueError : 数据为空 / 行数不足
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data 必须为 pd.DataFrame，实际类型: {type(data).__name__}"
            )
        if data.empty:
            raise ValueError("输入数据不能为空 DataFrame")
        if data.shape[0] < 1:
            raise ValueError(
                f"数据至少需要 1 行，当前: {data.shape[0]} 行"
            )

    # ── 便捷接口 ───────────────────────────────────────────

    def get_output_data(self) -> pd.DataFrame:
        """
        获取处理后的 DataFrame（compute() 后可调用）。

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        RuntimeError : 未执行 compute() 时抛出
        """
        result = self.get_result()
        output = result.metadata.get("output_data")
        if output is None:
            raise RuntimeError(
                f"[{self.METHOD_NAME_EN}] 结果中不含 'output_data'，"
                "请检查 compute() 实现是否正确写入 metadata['output_data']。"
            )
        if not isinstance(output, pd.DataFrame):
            raise RuntimeError(
                f"[{self.METHOD_NAME_EN}] metadata['output_data'] "
                f"类型错误: {type(output)}"
            )
        return output

    # ── 内部工具 ───────────────────────────────────────────

    @staticmethod
    def _extract_numeric(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        提取数值型列，返回 (数值子集DataFrame, 数值列名列表)。
        """
        num_cols = data.select_dtypes(include=["number"]).columns.tolist()
        return data[num_cols], num_cols


# ── 公开导出 ────────────────────────────────────────────────

from .cleaner import MissingValueHandler, OutlierHandler, DataCleaner
from .normalizer import (
    PositivityTransformer,
    MinMaxNormalizer,
    ZScoreNormalizer,
    VectorNormalizer,
    SumNormalizer,
    DataNormalizer,
)
from .reduction import CorrelationAnalyzer, PCAReducer

__all__ = [
    # 基类
    "PreprocessBase",
    # 清洗
    "MissingValueHandler",
    "OutlierHandler",
    "DataCleaner",
    # 标准化
    "PositivityTransformer",
    "MinMaxNormalizer",
    "ZScoreNormalizer",
    "VectorNormalizer",
    "SumNormalizer",
    "DataNormalizer",
    # 降维/相关性
    "CorrelationAnalyzer",
    "PCAReducer",
]