# src/algorithms/preprocess/normalizer.py
"""
指标正向化与标准化模块
─────────────────────────────────────────────────────────────
处理顺序（数学建模标准流程）：
  原始数据
    │
    ▼  Step 1: PositivityTransformer（正向化）
  所有指标统一为"越大越好"
    │
    ▼  Step 2: 选用一种标准化方法
  ┌──────────────────────────────────────┐
  │  MinMaxNormalizer   → [0, 1]         │
  │  ZScoreNormalizer   → μ=0, σ=1       │
  │  VectorNormalizer   → 列L2范数=1     │
  │  SumNormalizer      → 列和=1（熵权用）│
  └──────────────────────────────────────┘
    │
    ▼
  DataNormalizer（一次性完成以上两步）
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from . import PreprocessBase
from ..base import IndicatorDirection, MethodCategory, MethodResult
from ...utils.logging_config import get_logger

logger = get_logger("auto_eval.normalizer")

# 标准化方法合法值
_NORM_METHODS = frozenset(["minmax", "zscore", "vector", "sum", "none"])


# ============================================================
#  Step 1 — 正向化处理
# ============================================================

class PositivityTransformer(PreprocessBase):
    """
    将所有指标转换为正向指标（越大越好）。

    支持的指标类型
    --------------
    positive  (极大型) : 不作变换，原样保留
    negative  (极小型) : x'_i = max(X) - x_i         （极差变换）
    moderate  (中间型) : x'_i = 1 - |x_i - x*| / M   （偏离最优值变换）
                         其中 x* 为最优值，M = max|x_i - x*|
    interval  (区间型) : 支持 [a, b] 最优区间变换（可选）

    Parameters（fit 传入）
    ----------------------
    directions       : dict[列名, IndicatorDirection | str]
                       指标方向，未指定列默认为 positive
    moderate_optimal : dict[列名, float]
                       适度型指标的最优值（direction='moderate' 时必填）
    interval_bounds  : dict[列名, tuple[float, float]]
                       区间型指标的最优区间 [a, b]（direction='interval' 时必填）
    """

    METHOD_NAME_ZH = "指标正向化"
    METHOD_NAME_EN = "PositivityTransformer"
    METHOD_ABBR    = "PT"

    def __init__(self, language: str = "zh"):
        super().__init__(language)
        self._directions:       dict[str, str]            = {}
        self._moderate_optimal: dict[str, float]          = {}
        self._interval_bounds:  dict[str, tuple[float, float]] = {}
        self._transform_log:    list[dict[str, Any]]      = []

    # ── fit ────────────────────────────────────────────────

    def fit(
        self,
        data: pd.DataFrame,
        directions: Optional[dict[str, Union[IndicatorDirection, str]]] = None,
        moderate_optimal: Optional[dict[str, float]] = None,
        interval_bounds: Optional[dict[str, tuple[float, float]]] = None,
        **kwargs: Any,
    ) -> "PositivityTransformer":
        """
        Parameters
        ----------
        data             : 数值 DataFrame，行为对象，列为指标
        directions       : 各指标方向，未指定则默认 positive
        moderate_optimal : 适度型最优值字典 {列名: 最优值}
        interval_bounds  : 区间型最优区间字典 {列名: (a, b)}
        """
        self.validate_input(data)

        self._raw_data        = data.copy()
        self._indicator_names = list(data.columns)
        self._object_names    = list(data.index)

        # 规范化 directions，key 为列名，value 为字符串
        raw_dirs = directions or {}
        self._directions = {
            col: (
                d.value if isinstance(d, IndicatorDirection) else str(d).lower()
            )
            for col, d in raw_dirs.items()
        }
        # 未指定的列默认 positive
        for col in self._indicator_names:
            self._directions.setdefault(col, "positive")

        self._moderate_optimal = moderate_optimal or {}
        self._interval_bounds  = interval_bounds  or {}

        # 校验：中间型指标必须提供最优值
        for col, d in self._directions.items():
            if d == "moderate" and col not in self._moderate_optimal:
                raise ValueError(
                    f"指标 '{col}' 设定为适度型(moderate)，"
                    f"但未在 moderate_optimal 中提供最优值。"
                )
            if d == "interval" and col not in self._interval_bounds:
                raise ValueError(
                    f"指标 '{col}' 设定为区间型(interval)，"
                    f"但未在 interval_bounds 中提供区间 (a, b)。"
                )

        self._is_fitted = True
        n_neg = sum(1 for d in self._directions.values() if d == "negative")
        n_mod = sum(1 for d in self._directions.values() if d == "moderate")
        self.logger.info(
            f"PositivityTransformer.fit | 共 {len(self._indicator_names)} 个指标 | "
            f"负向: {n_neg} | 适度: {n_mod}"
        )
        return self

    # ── compute ────────────────────────────────────────────

    def compute(self) -> MethodResult:
        """执行正向化变换。"""
        self._check_fitted("compute")
        t0 = self._start_timer()

        df = self._raw_data.copy().astype(float)
        self._transform_log = []

        for col in self._indicator_names:
            if col not in df.columns:
                continue
            direction = self._directions.get(col, "positive")
            df, log_entry = self._transform_col(df, col, direction)
            self._transform_log.append(log_entry)

        # 汇总报告
        report_df = pd.DataFrame(self._transform_log)

        result = self._build_result(
            tables={
                "output":        df,
                "transform_log": report_df,
            },
            scalars={
                "n_positive": sum(1 for d in self._directions.values() if d == "positive"),
                "n_negative": sum(1 for d in self._directions.values() if d == "negative"),
                "n_moderate": sum(1 for d in self._directions.values() if d == "moderate"),
                "n_interval": sum(1 for d in self._directions.values() if d == "interval"),
            },
            metadata={"output_data": df},
            elapsed=self._stop_timer(t0),
        )
        self._result = result
        self.logger.info(
            f"正向化完成 | 负向变换 {result.scalars['n_negative']} 个 | "
            f"适度变换 {result.scalars['n_moderate']} 个 | "
            f"耗时 {result.elapsed_time:.3f}s"
        )
        return result

    # ── 内部变换逻辑 ───────────────────────────────────────

    def _transform_col(
        self,
        df: pd.DataFrame,
        col: str,
        direction: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        对单列执行正向化变换，返回 (更新后df, 日志条目)。
        """
        series = df[col]
        log = {"指标": col, "方向": direction, "变换公式": "—", "备注": ""}

        if direction == "positive":
            log["变换公式"] = "x（不变）"

        elif direction == "negative":
            max_val = series.max()
            df[col]           = max_val - series
            log["变换公式"]   = f"max({max_val:.4f}) - x"

        elif direction == "moderate":
            optimal = self._moderate_optimal[col]
            deviations = (series - optimal).abs()
            M = deviations.max()
            if M == 0:
                df[col]         = 1.0
                log["变换公式"] = "全部最优（偏差=0），赋值 1"
            else:
                df[col]         = 1 - deviations / M
                log["变换公式"] = f"1 - |x - {optimal}| / {M:.4f}"

        elif direction == "interval":
            a, b = self._interval_bounds[col]
            lower_excess = max(a - series.min(), 0)
            upper_excess = max(series.max() - b, 0)
            M = max(lower_excess, upper_excess)

            if M == 0:
                df[col]         = 1.0
                log["变换公式"] = "全部在最优区间内，赋值 1"
            else:
                within    = (series >= a) & (series <= b)
                below_a   = series < a
                above_b   = series > b
                result_col = series.copy()
                result_col[within]  = 1.0
                result_col[below_a] = 1 - (a - series[below_a]) / M
                result_col[above_b] = 1 - (series[above_b] - b) / M
                df[col]         = result_col
                log["变换公式"] = f"区间 [{a}, {b}]，M={M:.4f}"

        else:
            log["备注"] = f"未知方向 '{direction}'，跳过"

        return df, log

    # ── summary / tex ──────────────────────────────────────

    def summary(self) -> str:
        r = self.get_result()
        lines = [
            f"\n{'='*55}",
            f"  指标正向化结果 (PositivityTransformer)",
            f"{'='*55}",
            f"  正向指标: {r.scalars['n_positive']} 个  "
            f"负向指标: {r.scalars['n_negative']} 个  "
            f"适度指标: {r.scalars['n_moderate']} 个",
        ]
        if "transform_log" in r.tables and not r.tables["transform_log"].empty:
            lines.append("\n  变换明细：")
            lines.append(r.tables["transform_log"].to_string(index=False))
        lines.append("="*55)
        text = "\n".join(lines)
        print(text)
        return text

    def tex_description(self) -> str:
        has_neg = any(d == "negative" for d in self._directions.values())
        has_mod = any(d == "moderate" for d in self._directions.values())

        neg_section = r"""
\paragraph{负向指标正向化}
对于极小型（负向）指标，采用极差变换将其转换为极大型指标：
\begin{equation}
    x'_{ij} = \max_{i}(x_{ij}) - x_{ij}
\end{equation}""" if has_neg else ""

        mod_section = r"""
\paragraph{适度型指标正向化}
对于中间型（适度型）指标，设最优值为 $x^*$，则变换公式为：
\begin{equation}
    x'_{ij} = 1 - \frac{|x_{ij} - x^*|}{M_j}, \quad
    M_j = \max_{i}|x_{ij} - x^*|
\end{equation}""" if has_mod else ""

        return rf"""
\subsubsection{{指标正向化处理}}
为消除指标方向差异，确保所有指标满足"越大越好"的一致性，
对原始数据矩阵进行正向化处理。
{neg_section}
{mod_section}
经正向化处理后，所有指标均满足极大型特征，
可直接用于后续标准化与综合评价计算。
"""


# ============================================================
#  Step 2a — Min-Max 归一化
# ============================================================

class MinMaxNormalizer(PreprocessBase):
    """
    Min-Max 归一化（极差标准化）。

    公式：
        x*_ij = (x_ij - min_j) / (max_j - min_j)

    输出范围：[0, 1]
    适用场景：TOPSIS（手动版）、灰色关联分析（GRA）等

    Parameters（fit 传入）
    ----------------------
    feature_range : tuple[float, float]，默认 (0, 1)
                    输出值域，设为 (0.001, 1) 可避免零值
    """

    METHOD_NAME_ZH = "Min-Max 归一化"
    METHOD_NAME_EN = "MinMaxNormalizer"
    METHOD_ABBR    = "MMN"

    def __init__(self, language: str = "zh"):
        super().__init__(language)
        self._feature_range: tuple[float, float] = (0.0, 1.0)
        self._col_min: Optional[pd.Series] = None
        self._col_max: Optional[pd.Series] = None

    def fit(
        self,
        data: pd.DataFrame,
        feature_range: tuple[float, float] = (0.0, 1.0),
        **kwargs: Any,
    ) -> "MinMaxNormalizer":
        self.validate_input(data)

        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                f"feature_range 的下界必须小于上界，"
                f"当前: {feature_range}"
            )

        self._raw_data        = data.copy()
        self._indicator_names = list(data.columns)
        self._object_names    = list(data.index)
        self._feature_range   = feature_range

        # 记录训练集统计量（支持 transform 独立样本）
        _, num_cols = self._extract_numeric(data)
        self._col_min = data[num_cols].min()
        self._col_max = data[num_cols].max()
        self._is_fitted = True
        return self

    def compute(self) -> MethodResult:
        self._check_fitted("compute")
        t0 = self._start_timer()

        df = self._raw_data.copy()
        _, num_cols = self._extract_numeric(df)

        out_min, out_max = self._feature_range
        stats_rows = []

        for col in num_cols:
            mn, mx = self._col_min[col], self._col_max[col]
            rng = mx - mn

            if rng < 1e-12:
                # 常数列：所有值相同，归一化为中点
                df[col] = (out_min + out_max) / 2.0
                self.logger.warning(
                    f"列 '{col}' 的极差≈0（常数列），已设为 {df[col].iloc[0]:.4f}"
                )
                stats_rows.append({
                    "指标": col,
                    "最小值": mn,
                    "最大值": mx,
                    "极差": rng,
                    "备注": "常数列，归一化为均值",
                })
            else:
                # 标准 MinMax，然后缩放到 feature_range
                df[col] = (df[col] - mn) / rng
                if out_min != 0.0 or out_max != 1.0:
                    df[col] = df[col] * (out_max - out_min) + out_min
                stats_rows.append({
                    "指标": col,
                    "最小值": round(mn, 4),
                    "最大值": round(mx, 4),
                    "极差":   round(rng, 4),
                    "备注":   "正常归一化",
                })

        stats_df = pd.DataFrame(stats_rows)
        result = self._build_result(
            tables={"output": df, "stats": stats_df},
            scalars={
                "feature_range_min": out_min,
                "feature_range_max": out_max,
            },
            metadata={"output_data": df},
            elapsed=self._stop_timer(t0),
        )
        self._result = result
        return result

    def summary(self) -> str:
        r = self.get_result()
        lines = [
            f"\n{'='*55}",
            f"  Min-Max 归一化结果摘要",
            f"{'='*55}",
            f"  输出范围: [{r.scalars['feature_range_min']}, "
            f"{r.scalars['feature_range_max']}]",
        ]
        if "stats" in r.tables:
            lines.append("\n  各指标统计：")
            lines.append(r.tables["stats"].to_string(index=False))
        lines.append("="*55)
        text = "\n".join(lines)
        print(text)
        return text

    def tex_description(self) -> str:
        lo, hi = self._feature_range
        range_note = (
            f"输出范围为 $[{lo}, {hi}]$"
            if (lo, hi) != (0.0, 1.0)
            else "输出范围为 $[0, 1]$"
        )
        return rf"""
\subsubsection{{Min-Max 归一化}}
采用极差标准化方法对正向化后的指标数据进行量纲统一，
{range_note}，计算公式为：
\begin{{equation}}
    x^*_{{ij}} = \frac{{x_{{ij}} - \min_i(x_{{ij}})}}{{\max_i(x_{{ij}}) - \min_i(x_{{ij}})}}
\end{{equation}}
其中 $x_{{ij}}$ 为第 $i$ 个评价对象在第 $j$ 项指标上的正向化数值。
"""


# ============================================================
#  Step 2b — Z-score 标准化
# ============================================================

class ZScoreNormalizer(PreprocessBase):
    """
    Z-score 标准化（均值-方差标准化）。

    公式：
        x*_ij = (x_ij - μ_j) / σ_j

    输出：均值 0，标准差 1（各列）
    适用场景：PCA降维前的数据准备、因子分析
    """

    METHOD_NAME_ZH = "Z-score 标准化"
    METHOD_NAME_EN = "ZScoreNormalizer"
    METHOD_ABBR    = "ZSN"

    def __init__(self, language: str = "zh"):
        super().__init__(language)
        self._col_mean: Optional[pd.Series] = None
        self._col_std:  Optional[pd.Series] = None

    def fit(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> "ZScoreNormalizer":
        self.validate_input(data)
        self._raw_data        = data.copy()
        self._indicator_names = list(data.columns)
        self._object_names    = list(data.index)

        _, num_cols = self._extract_numeric(data)
        self._col_mean = data[num_cols].mean()
        self._col_std  = data[num_cols].std(ddof=1)   # 样本标准差
        self._is_fitted = True
        return self

    def compute(self) -> MethodResult:
        self._check_fitted("compute")
        t0 = self._start_timer()

        df = self._raw_data.copy()
        _, num_cols = self._extract_numeric(df)
        stats_rows = []

        for col in num_cols:
            mu  = self._col_mean[col]
            std = self._col_std[col]

            if std < 1e-12:
                df[col] = 0.0
                self.logger.warning(f"列 '{col}' 标准差≈0（常数列），Z-score 置为 0")
                stats_rows.append({
                    "指标": col, "均值": round(mu, 4), "标准差": round(std, 6), "备注": "常数列"
                })
            else:
                df[col] = (df[col] - mu) / std
                stats_rows.append({
                    "指标": col, "均值": round(mu, 4), "标准差": round(std, 4), "备注": "正常"
                })

        result = self._build_result(
            tables={"output": df, "stats": pd.DataFrame(stats_rows)},
            metadata={"output_data": df},
            elapsed=self._stop_timer(t0),
        )
        self._result = result
        return result

    def summary(self) -> str:
        r = self.get_result()
        lines = [f"\n{'='*55}", "  Z-score 标准化摘要", f"{'='*55}"]
        if "stats" in r.tables:
            lines.append(r.tables["stats"].to_string(index=False))
        lines.append("="*55)
        text = "\n".join(lines)
        print(text)
        return text

    def tex_description(self) -> str:
        return r"""
\subsubsection{Z-score 标准化}
采用均值-方差标准化处理数据，使各指标均值为 0、标准差为 1，
消除量纲影响，公式为：
\begin{equation}
    x^*_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}
\end{equation}
其中 $\bar{x}_j$ 和 $s_j$ 分别为第 $j$ 项指标的样本均值与标准差。
"""


# ============================================================
#  Step 2c — 向量归一化（列L2范数）
# ============================================================

class VectorNormalizer(PreprocessBase):
    """
    向量归一化（列L2范数归一化）。

    公式：
        x*_ij = x_ij / sqrt( Σ_i x_ij² )

    适用场景：TOPSIS 标准建模步骤（构建规范化决策矩阵）
    """

    METHOD_NAME_ZH = "向量归一化"
    METHOD_NAME_EN = "VectorNormalizer"
    METHOD_ABBR    = "VN"

    def fit(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> "VectorNormalizer":
        self.validate_input(data)
        self._raw_data        = data.copy()
        self._indicator_names = list(data.columns)
        self._object_names    = list(data.index)
        self._is_fitted       = True
        return self

    def compute(self) -> MethodResult:
        self._check_fitted("compute")
        t0 = self._start_timer()

        df = self._raw_data.copy()
        _, num_cols = self._extract_numeric(df)
        stats_rows = []

        for col in num_cols:
            col_norm = np.sqrt((df[col] ** 2).sum())
            if col_norm < 1e-12:
                self.logger.warning(f"列 '{col}' 的 L2 范数≈0，向量归一化置为 0")
                df[col] = 0.0
                stats_rows.append({"指标": col, "L2范数": 0.0, "备注": "零向量"})
            else:
                df[col] = df[col] / col_norm
                stats_rows.append({
                    "指标": col,
                    "L2范数": round(col_norm, 4),
                    "备注": "正常",
                })

        result = self._build_result(
            tables={"output": df, "stats": pd.DataFrame(stats_rows)},
            metadata={"output_data": df},
            elapsed=self._stop_timer(t0),
        )
        self._result = result
        return result

    def summary(self) -> str:
        r = self.get_result()
        lines = [f"\n{'='*55}", "  向量归一化摘要", f"{'='*55}"]
        if "stats" in r.tables:
            lines.append(r.tables["stats"].to_string(index=False))
        lines.append("="*55)
        text = "\n".join(lines)
        print(text)
        return text

    def tex_description(self) -> str:
        return r"""
\subsubsection{向量归一化}
采用向量归一化方法构建规范化决策矩阵，公式为：
\begin{equation}
    x^*_{ij} = \frac{x_{ij}}{\sqrt{\displaystyle\sum_{i=1}^{m} x_{ij}^2}}
\end{equation}
该方法常用于 TOPSIS 模型的数据预处理阶段。
"""


# ============================================================
#  Step 2d — 总和归一化（熵权法专用）
# ============================================================

class SumNormalizer(PreprocessBase):
    """
    总和（比例）归一化。

    公式：
        p_ij = x_ij / Σ_i x_ij

    说明：
    - 要求数据经过正向化处理，所有值 ≥ 0
    - 该方法专为熵权法（Entropy Weight Method）设计
    - 每列之和为 1，可解释为概率分布
    """

    METHOD_NAME_ZH = "总和归一化"
    METHOD_NAME_EN = "SumNormalizer"
    METHOD_ABBR    = "SN"

    def fit(
        self,
        data: pd.DataFrame,
        epsilon: float = 1e-9,
        **kwargs: Any,
    ) -> "SumNormalizer":
        """
        Parameters
        ----------
        epsilon : float
            防零分母的极小量
        """
        self.validate_input(data)
        self._raw_data        = data.copy()
        self._indicator_names = list(data.columns)
        self._object_names    = list(data.index)
        self._epsilon         = epsilon
        self._is_fitted       = True
        return self

    def compute(self) -> MethodResult:
        self._check_fitted("compute")
        t0 = self._start_timer()

        df = self._raw_data.copy()
        _, num_cols = self._extract_numeric(df)

        # 处理可能存在的负值（轻微平移至非负）
        for col in num_cols:
            min_val = df[col].min()
            if min_val < 0:
                df[col] = df[col] - min_val
                self.logger.warning(
                    f"列 '{col}' 含负值（min={min_val:.4f}），"
                    f"已整体平移至非负"
                )

        for col in num_cols:
            col_sum = df[col].sum()
            df[col] = df[col] / (col_sum + self._epsilon)

        result = self._build_result(
            tables={"output": df},
            metadata={"output_data": df},
            elapsed=self._stop_timer(t0),
        )
        self._result = result
        return result

    def summary(self) -> str:
        r = self.get_result()
        text = f"\nSumNormalizer 完成 | 耗时 {r.elapsed_time:.3f}s\n"
        print(text)
        return text

    def tex_description(self) -> str:
        return r"""
\subsubsection{总和（比例）归一化}
为计算信息熵，将正向化数据按列进行比例归一化：
\begin{equation}
    p_{ij} = \frac{x_{ij}}{\displaystyle\sum_{i=1}^{m} x_{ij}}
\end{equation}
其中 $p_{ij}$ 可视为第 $i$ 个评价对象在第 $j$ 项指标上的贡献比例。
"""


# ============================================================
#  完整标准化流水线（推荐使用）
# ============================================================

class DataNormalizer(PreprocessBase):
    """
    数据标准化完整流水线（正向化 + 归一化一步到位）。

    内部执行顺序：
        PositivityTransformer → 选定的归一化方法

    Parameters（fit 传入）
    ----------------------
    directions       : 各指标方向字典
    moderate_optimal : 适度型指标最优值字典
    interval_bounds  : 区间型指标最优区间字典
    method           : 归一化方法 ('minmax'|'zscore'|'vector'|'sum'|'none')
    feature_range    : MinMax 输出范围（仅 method='minmax' 时生效）
    skip_positivity  : bool，True 则跳过正向化步骤
    """

    METHOD_NAME_ZH = "数据标准化处理器"
    METHOD_NAME_EN = "DataNormalizer"
    METHOD_ABBR    = "DN"

    def __init__(self, language: str = "zh"):
        super().__init__(language)
        self._positivity:   Optional[PositivityTransformer] = None
        self._normalizer:   Optional[PreprocessBase]        = None
        self._method:       str                             = "minmax"
        self._skip_positivity: bool                         = False

    def fit(
        self,
        data: pd.DataFrame,
        directions: Optional[dict[str, Union[IndicatorDirection, str]]] = None,
        moderate_optimal: Optional[dict[str, float]] = None,
        interval_bounds:  Optional[dict[str, tuple[float, float]]] = None,
        method: str = "minmax",
        feature_range: tuple[float, float] = (0.0, 1.0),
        skip_positivity: bool = False,
        **kwargs: Any,
    ) -> "DataNormalizer":
        """
        Parameters
        ----------
        data             : 原始数值 DataFrame
        directions       : 指标方向字典，同 PositivityTransformer
        moderate_optimal : 适度型最优值
        interval_bounds  : 区间型最优区间
        method           : 归一化方法
        feature_range    : MinMax 输出范围
        skip_positivity  : True 则跳过正向化（数据已是正向）
        """
        self.validate_input(data)

        if method not in _NORM_METHODS:
            raise ValueError(
                f"method='{method}' 无效，合法值: {sorted(_NORM_METHODS)}"
            )

        self._raw_data        = data.copy()
        self._indicator_names = list(data.columns)
        self._object_names    = list(data.index)
        self._method          = method
        self._skip_positivity = skip_positivity

        # 初始化正向化处理器
        if not skip_positivity:
            self._positivity = PositivityTransformer(self._language)
            self._positivity.fit(
                data,
                directions=directions,
                moderate_optimal=moderate_optimal,
                interval_bounds=interval_bounds,
            )

        # 初始化归一化处理器（数据在 compute 阶段才传入）
        _norm_cls_map: dict[str, type[PreprocessBase]] = {
            "minmax":  MinMaxNormalizer,
            "zscore":  ZScoreNormalizer,
            "vector":  VectorNormalizer,
            "sum":     SumNormalizer,
        }
        if method != "none":
            self._normalizer = _norm_cls_map[method](self._language)
            self._norm_kwargs = (
                {"feature_range": feature_range} if method == "minmax" else {}
            )
        else:
            self._normalizer  = None
            self._norm_kwargs = {}

        self._is_fitted = True
        self.logger.info(
            f"DataNormalizer.fit | method='{method}' | "
            f"skip_positivity={skip_positivity}"
        )
        return self

    def compute(self) -> MethodResult:
        self._check_fitted("compute")
        t0 = self._start_timer()

        current_df = self._raw_data.copy()
        positivity_result = None
        norm_result       = None

        # ── Step 1: 正向化 ─────────────────────────────────
        if not self._skip_positivity and self._positivity:
            self._positivity._raw_data = current_df
            positivity_result = self._positivity.compute()
            current_df = positivity_result.metadata["output_data"]
            self.logger.debug("正向化步骤完成")

        # ── Step 2: 归一化 ─────────────────────────────────
        if self._normalizer:
            self._normalizer.fit(current_df, **self._norm_kwargs)
            norm_result = self._normalizer.compute()
            current_df  = norm_result.metadata["output_data"]
            self.logger.debug(f"归一化（{self._method}）步骤完成")

        # ── 汇总 ───────────────────────────────────────────
        tables: dict[str, pd.DataFrame] = {"output": current_df}
        if positivity_result and "transform_log" in positivity_result.tables:
            tables["positivity_log"] = positivity_result.tables["transform_log"]
        if norm_result and "stats" in norm_result.tables:
            tables["norm_stats"] = norm_result.tables["stats"]

        result = self._build_result(
            tables=tables,
            scalars={
                "method":            self._method,
                "skip_positivity":   self._skip_positivity,
                "output_shape":      current_df.shape,
            },
            metadata={
                "output_data": current_df,
                "positivity_result": positivity_result,
                "norm_result":       norm_result,
            },
            elapsed=self._stop_timer(t0),
        )
        self._result = result
        self.logger.info(
            f"DataNormalizer 完成 | shape={current_df.shape} | "
            f"耗时 {result.elapsed_time:.3f}s"
        )
        return result

    def summary(self) -> str:
        r = self.get_result()
        lines = [
            f"\n{'='*60}",
            f"  数据标准化流水线摘要 (DataNormalizer)",
            f"{'='*60}",
            f"  归一化方法     : {r.scalars['method']}",
            f"  跳过正向化     : {r.scalars['skip_positivity']}",
            f"  输出形状       : {r.scalars['output_shape']}",
            f"  耗时           : {r.elapsed_time:.3f}s",
        ]
        lines.append("="*60)
        text = "\n".join(lines)
        print(text)
        return text

    def tex_description(self) -> str:
        pos_tex  = self._positivity.tex_description() if self._positivity else ""
        norm_tex = self._normalizer.tex_description() if self._normalizer else ""
        return pos_tex + "\n" + norm_tex