# src/parser/data_profiler.py
"""
数据特征画像分析器
==================
自动分析数据集的统计特征（缺失率、相关性、变异系数等），
为算法推荐器提供决策依据。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    数据特征画像分析器

    Parameters
    ----------
    matrix : np.ndarray
        纯数值指标矩阵，shape = (n_objects, n_indicators)。
    column_names : list[str], optional
        指标列名列表，默认自动编号。

    Examples
    --------
    >>> profiler = DataProfiler(X_norm, ["指标A", "指标B", "指标C"])
    >>> info = profiler.analyze()
    >>> print(info["max_corr"])
    0.87
    """

    def __init__(
        self,
        matrix: np.ndarray,
        column_names: Optional[List[str]] = None,
    ) -> None:
        if isinstance(matrix, pd.DataFrame):
            if column_names is None:
                column_names = matrix.columns.tolist()
            matrix = matrix.values

        self.matrix = np.asarray(matrix, dtype=float)
        self.n_objects, self.n_indicators = self.matrix.shape
        self.column_names = (
            column_names
            if column_names is not None
            else [f"X{i+1}" for i in range(self.n_indicators)]
        )

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def analyze(self) -> Dict[str, Any]:
        """
        运行全套数据特征分析，返回结构化结果字典。

        Returns
        -------
        dict
            键包括:
            - n_objects, n_indicators : 样本数、指标数
            - missing_rate            : 总体缺失率
            - max_corr, mean_corr     : 最大/平均两两相关系数（绝对值）
            - corr_matrix             : 相关系数矩阵 (n, n)
            - mean_cv                 : 变异系数均值
            - ratio_n_p               : 样本数/指标数
            - descriptive             : 各指标的 min/max/mean/std
        """
        result: Dict[str, Any] = {
            "n_objects": self.n_objects,
            "n_indicators": self.n_indicators,
            "ratio_n_p": self.n_objects / max(self.n_indicators, 1),
        }

        # 缺失率
        result["missing_rate"] = self._calc_missing_rate()

        # 相关性
        corr_info = self._calc_correlation()
        result.update(corr_info)

        # 变异系数
        result["mean_cv"] = self._calc_cv()

        # 描述性统计
        result["descriptive"] = self._descriptive_stats()

        logger.info(
            "数据画像: %d 对象 × %d 指标, "
            "缺失率=%.2f%%, 最大相关=%.3f, 平均CV=%.3f",
            result["n_objects"],
            result["n_indicators"],
            result["missing_rate"] * 100,
            result.get("max_corr", 0),
            result.get("mean_cv", 0),
        )
        return result

    def get_high_corr_pairs(
        self, threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        获取相关系数（绝对值）超过阈值的指标对。

        Parameters
        ----------
        threshold : float
            相关系数阈值（绝对值），默认 0.85。

        Returns
        -------
        list[dict]
            每个元素包含 indicator_a, indicator_b, corr_value。
        """
        corr = self._corr_matrix()
        pairs = []
        n = corr.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) >= threshold:
                    pairs.append({
                        "indicator_a": self.column_names[i],
                        "indicator_b": self.column_names[j],
                        "corr_value": float(corr[i, j]),
                    })
        pairs.sort(key=lambda p: abs(p["corr_value"]), reverse=True)
        return pairs

    def summary_text(self) -> str:
        """返回可打印的文字摘要。"""
        info = self.analyze()
        lines = [
            f"数据规模    : {info['n_objects']} 个对象 × {info['n_indicators']} 个指标",
            f"样本/指标比 : {info['ratio_n_p']:.2f}",
            f"总体缺失率  : {info['missing_rate']*100:.2f}%",
            f"最大相关系数: {info.get('max_corr', 0):.4f}",
            f"平均相关系数: {info.get('mean_corr', 0):.4f}",
            f"变异系数均值: {info.get('mean_cv', 0):.4f}",
        ]
        high_corr = self.get_high_corr_pairs(0.85)
        if high_corr:
            lines.append("高相关指标对:")
            for p in high_corr[:5]:
                lines.append(
                    f"  {p['indicator_a']} ↔ {p['indicator_b']} "
                    f"(r={p['corr_value']:.3f})"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _calc_missing_rate(self) -> float:
        if np.isnan(self.matrix).any():
            return float(np.isnan(self.matrix).mean())
        return 0.0

    def _corr_matrix(self) -> np.ndarray:
        try:
            corr = np.corrcoef(self.matrix.T)
            return np.nan_to_num(corr, nan=0.0)
        except Exception:
            return np.zeros((self.n_indicators, self.n_indicators))

    def _calc_correlation(self) -> Dict[str, Any]:
        corr = self._corr_matrix()
        abs_corr = np.abs(corr.copy())
        np.fill_diagonal(abs_corr, 0.0)
        return {
            "max_corr": float(abs_corr.max()) if abs_corr.size > 0 else 0.0,
            "mean_corr": float(abs_corr.mean()) if abs_corr.size > 0 else 0.0,
            "corr_matrix": corr,
        }

    def _calc_cv(self) -> float:
        """变异系数（标准差/均值绝对值）的均值。"""
        try:
            std = self.matrix.std(axis=0, ddof=1)
            mean_abs = np.abs(self.matrix.mean(axis=0)) + 1e-10
            cv = std / mean_abs
            return float(cv.mean())
        except Exception:
            return 0.0

    def _descriptive_stats(self) -> Dict[str, Dict[str, float]]:
        """各指标的描述性统计。"""
        stats = {}
        for j, name in enumerate(self.column_names):
            col = self.matrix[:, j]
            stats[name] = {
                "min": float(np.nanmin(col)),
                "max": float(np.nanmax(col)),
                "mean": float(np.nanmean(col)),
                "std": float(np.nanstd(col, ddof=1)),
            }
        return stats

    def __repr__(self) -> str:
        return (
            f"DataProfiler(n_objects={self.n_objects}, "
            f"n_indicators={self.n_indicators})"
        )