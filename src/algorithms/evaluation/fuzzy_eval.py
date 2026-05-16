"""
fuzzy_eval.py
模糊综合评价法（Fuzzy Comprehensive Evaluation）

Reference:
    Zadeh, L.A. (1965). Fuzzy sets. Information and Control, 8(3), 338-353.
    汪培庄 (1983). 模糊集合论及其应用. 上海科学技术出版社.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..base import BaseMethod

logger = logging.getLogger(__name__)

_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# ──────────────────────────────────────────────────────────────────────────────
# 隶属函数工具
# ──────────────────────────────────────────────────────────────────────────────

def triangular_membership(x: float, a: float, b: float, c: float) -> float:
    """
    三角形隶属函数 μ(x)。

    Parameters
    ----------
    a, b, c : float
        三角形顶点坐标，需满足 a ≤ b ≤ c。
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a + 1e-12)
    else:
        return (c - x) / (c - b + 1e-12)


def trapezoidal_membership(x: float, a: float, b: float, c: float, d: float) -> float:
    """梯形隶属函数 μ(x)，a ≤ b ≤ c ≤ d。"""
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a + 1e-12)
    elif b <= x <= c:
        return 1.0
    else:
        return (d - x) / (d - c + 1e-12)


def semi_left_membership(x: float, a: float, b: float) -> float:
    """左半梯形（最低等级）：x ≤ a → 1，a < x ≤ b → 线性衰减，x > b → 0。"""
    if x <= a:
        return 1.0
    elif a < x <= b:
        return (b - x) / (b - a + 1e-12)
    return 0.0


def semi_right_membership(x: float, a: float, b: float) -> float:
    """右半梯形（最高等级）：x < a → 0，a ≤ x < b → 线性上升，x ≥ b → 1。"""
    if x >= b:
        return 1.0
    elif a <= x < b:
        return (x - a) / (b - a + 1e-12)
    return 0.0


def build_membership_matrix(
    values: np.ndarray,
    level_boundaries: List[Tuple[float, ...]],
    level_type: str = "triangle",
) -> np.ndarray:
    """
    为一列指标值批量计算对各评价等级的隶属度矩阵。

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        某指标的 n 个样本值。
    level_boundaries : list of tuple
        每个等级对应的隶属函数参数元组，从最低到最高等级排列。
        - 三角形: (a, b, c)
        - 梯形:  (a, b, c, d)
        - 左半梯形（最低级）: (a, b)  → 自动识别
        - 右半梯形（最高级）: (a, b)  → 自动识别
    level_type : str
        'triangle' 或 'trapezoid'（边缘等级自动使用半梯形）。

    Returns
    -------
    np.ndarray, shape (n, p)
        n 个样本对 p 个等级的隶属度矩阵，行之和归一化为 1。
    """
    n = len(values)
    p = len(level_boundaries)
    R = np.zeros((n, p))

    for k, params in enumerate(level_boundaries):
        for i, x in enumerate(values):
            if k == 0 and len(params) == 2:
                R[i, k] = semi_left_membership(x, params[0], params[1])
            elif k == p - 1 and len(params) == 2:
                R[i, k] = semi_right_membership(x, params[0], params[1])
            elif len(params) == 3:
                R[i, k] = triangular_membership(x, *params)
            else:
                R[i, k] = trapezoidal_membership(x, *params)

    # 行归一化（确保各行之和为 1）
    row_sums = R.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    return R / row_sums


# ──────────────────────────────────────────────────────────────────────────────
# 主类
# ──────────────────────────────────────────────────────────────────────────────

class FuzzyComprehensiveEvaluation(BaseMethod):
    """
    模糊综合评价法。

    支持两种输入模式：
    1. **直接模式**：直接传入预计算的隶属度矩阵（三维数组）。
    2. **自动模式**：传入原始决策矩阵 + 各指标的等级边界参数，
       系统自动计算隶属度矩阵。

    Parameters
    ----------
    level_names : list of str
        评价等级名称，如 ``['差', '一般', '良', '优']``。
    level_scores : array-like
        各等级对应的量化分值，长度须与 ``level_names`` 相同。
    operator : str, default 'weighted_average'
        模糊合成算子：
        - ``'weighted_average'``：加权平均算子 M(·,+)
        - ``'min_max'``：最小—最大算子 M(∧,∨)
    weights : array-like, optional
        指标权重向量，默认等权。
    """

    def __init__(
        self,
        level_names: Optional[List[str]] = None,
        level_scores: Optional[Union[List[float], np.ndarray]] = None,
        operator: str = "weighted_average",
        weights: Optional[Union[List[float], np.ndarray]] = None,
    ) -> None:
        super().__init__(name="FuzzyComprehensiveEvaluation")
        self.level_names = level_names or ["差", "一般", "良", "优秀"]
        self.level_scores = (
            np.asarray(level_scores, dtype=float)
            if level_scores is not None
            else np.array([1.0, 2.0, 3.0, 4.0])
        )
        if len(self.level_names) != len(self.level_scores):
            raise ValueError("level_names 与 level_scores 长度须一致。")
        if operator not in ("weighted_average", "min_max"):
            raise ValueError("operator 须为 'weighted_average' 或 'min_max'。")
        self.operator = operator
        self.weights: Optional[np.ndarray] = (
            np.asarray(weights, dtype=float) if weights is not None else None
        )

        self._data: Optional[pd.DataFrame] = None
        self._alternatives: Optional[List] = None
        self._criteria: Optional[List] = None
        # membership_tensor: shape (n_alt, n_criteria, n_levels)
        self._membership_tensor: Optional[np.ndarray] = None
        # B matrix: shape (n_alt, n_levels)  ── 综合隶属向量
        self._B: Optional[np.ndarray] = None
        self._scores: Optional[np.ndarray] = None
        self._dominant_levels: Optional[np.ndarray] = None

    def fit(
        self,
        data: pd.DataFrame,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        membership_tensor: Optional[np.ndarray] = None,
        level_boundaries: Optional[List[List[Tuple]]] = None,
    ) -> "FuzzyComprehensiveEvaluation":
        """
        绑定数据与参数。

        Parameters
        ----------
        data : pd.DataFrame
            原始决策矩阵（自动模式）或任意形状标识用 DataFrame（直接模式）。
        membership_tensor : np.ndarray, shape (n_alt, n_criteria, n_levels), optional
            预计算的隶属度张量（直接模式）。若提供则忽略 ``level_boundaries``。
        level_boundaries : list of list of tuple, optional
            自动计算模式下，每个指标对应的等级边界列表。
            外层列表长度 = 指标数，内层列表长度 = 等级数。
        """
        if data.isnull().any().any():
            data = data.fillna(data.mean(numeric_only=True))
            logger.warning("缺失值已用列均值填充。")

        self._data = data.copy()
        self._alternatives = list(data.index)
        self._criteria = list(data.columns)
        n, m = data.shape
        p = len(self.level_names)

        if weights is not None:
            self.weights = np.asarray(weights, dtype=float)
        if self.weights is None:
            self.weights = np.ones(m, dtype=float)
        self.weights = self.weights / self.weights.sum()

        # ── 构建隶属度张量 ────────────────────────────────────────────────
        if membership_tensor is not None:
            tensor = np.asarray(membership_tensor, dtype=float)
            if tensor.shape != (n, m, p):
                raise ValueError(
                    f"membership_tensor 形状须为 ({n}, {m}, {p})，"
                    f"当前: {tensor.shape}"
                )
            self._membership_tensor = tensor
            logger.info("使用预计算隶属度张量（直接模式）。")

        elif level_boundaries is not None:
            if len(level_boundaries) != m:
                raise ValueError("level_boundaries 长度须等于指标数。")
            tensor = np.zeros((n, m, p))
            for j, boundaries in enumerate(level_boundaries):
                if len(boundaries) != p:
                    raise ValueError(
                        f"第 {j} 个指标的等级边界数 {len(boundaries)} ≠ 等级数 {p}。"
                    )
                col_vals = data.iloc[:, j].values
                tensor[:, j, :] = build_membership_matrix(col_vals, boundaries)
            self._membership_tensor = tensor
            logger.info("自动计算隶属度张量（级别边界模式）。")

        else:
            # 默认：以百分位数为等级边界，自动构建等分梯形隶属函数
            logger.warning(
                "未提供 membership_tensor 或 level_boundaries，"
                "将使用等分百分位数自动构建隶属函数。"
            )
            tensor = np.zeros((n, m, p))
            for j in range(m):
                col_vals = data.iloc[:, j].values
                percentiles = np.percentile(col_vals, np.linspace(0, 100, p + 1))
                boundaries = []
                for k in range(p):
                    if k == 0:
                        boundaries.append((percentiles[0], percentiles[1]))
                    elif k == p - 1:
                        boundaries.append((percentiles[-2], percentiles[-1]))
                    else:
                        a, b = percentiles[k], percentiles[k + 1]
                        mid = (a + b) / 2
                        boundaries.append((a, mid, b))
                tensor[:, j, :] = build_membership_matrix(col_vals, boundaries)
            self._membership_tensor = tensor

        if len(self.weights) != m:
            raise ValueError(f"权重维度 {len(self.weights)} ≠ 指标数 {m}。")

        self.metadata.update(
            {
                "n_alternatives": n,
                "n_criteria": m,
                "n_levels": p,
                "level_names": self.level_names,
                "level_scores": self.level_scores.tolist(),
                "operator": self.operator,
            }
        )
        return self

    def compute(self) -> pd.DataFrame:
        """
        执行模糊综合评价。

        Returns
        -------
        pd.DataFrame
            列：各等级隶属度 + ``Score`` + ``Dominant Level`` + ``Rank``，
            按得分降序排列。
        """
        if self._membership_tensor is None:
            raise RuntimeError("请先调用 fit()。")

        n, m, p = self._membership_tensor.shape

        # ── 模糊合成：B = W ∘ R ───────────────────────────────────────────
        self._B = np.zeros((n, p))

        if self.operator == "weighted_average":
            # b_ik = Σ_j w_j * r_ijk
            for i in range(n):
                for k in range(p):
                    self._B[i, k] = (
                        self.weights * self._membership_tensor[i, :, k]
                    ).sum()
        else:  # min_max
            for i in range(n):
                for k in range(p):
                    self._B[i, k] = np.max(
                        np.minimum(self.weights, self._membership_tensor[i, :, k])
                    )

        # 行归一化
        row_sums = self._B.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        self._B = self._B / row_sums

        # ── 去模糊化（加权平均法）────────────────────────────────────────
        self._scores = self._B @ self.level_scores  # (n,)

        # 最大隶属度等级
        self._dominant_levels = np.argmax(self._B, axis=1)

        ranks = (
            pd.Series(self._scores, index=self._alternatives)
            .rank(ascending=False, method="min")
            .astype(int)
        )

        level_cols = {
            f"μ({name})": np.round(self._B[:, k], 4)
            for k, name in enumerate(self.level_names)
        }
        self.result = pd.DataFrame(
            {
                **level_cols,
                "Score": np.round(self._scores, 4),
                "Dominant Level": [self.level_names[i] for i in self._dominant_levels],
                "Rank": ranks,
            },
            index=self._alternatives,
        ).sort_values("Rank")

        self.metadata.update(
            {
                "scores": dict(zip(self._alternatives, self._scores.round(4).tolist())),
                "best": self.result.index[0],
                "worst": self.result.index[-1],
            }
        )
        logger.info(
            "模糊综合评价完成 → 最优: %s (score=%.4f)",
            self.result.index[0], self._scores.max(),
        )
        return self.result

    def get_membership_dataframe(self, alternative: str) -> pd.DataFrame:
        """获取指定评价对象的隶属度矩阵（指标×等级）。"""
        if self._membership_tensor is None:
            raise RuntimeError("请先调用 fit()。")
        idx = self._alternatives.index(alternative)
        return pd.DataFrame(
            self._membership_tensor[idx],
            index=self._criteria,
            columns=self.level_names,
        ).round(4)

    def summary(self) -> Dict[str, Any]:
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")
        return {
            "method": "FuzzyComprehensiveEvaluation",
            "description": "模糊综合评价法",
            "level_names": self.level_names,
            "level_scores": self.level_scores.tolist(),
            "operator": self.operator,
            "n_alternatives": len(self._alternatives),
            "n_criteria": len(self._criteria),
            "weights": dict(zip(self._criteria, self.weights.tolist())),
            "B_matrix": pd.DataFrame(
                self._B,
                index=self._alternatives,
                columns=self.level_names,
            ).round(4).to_dict(),
            "scores": dict(zip(self._alternatives, self._scores.round(4).tolist())),
            "dominant_levels": {
                a: self.level_names[d]
                for a, d in zip(self._alternatives, self._dominant_levels)
            },
            "ranking": self.result["Rank"].to_dict(),
            "best": self.result.index[0],
        }

    def plot_membership_heatmap(
        self,
        figsize: Tuple = (10, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制综合隶属度矩阵 B 的热力图。"""
        if self._B is None:
            raise RuntimeError("请先调用 compute()。")

        import seaborn as sns
        sorted_alts = self.result.index.tolist()
        B_df = pd.DataFrame(
            self._B, index=self._alternatives, columns=self.level_names
        ).loc[sorted_alts]

        with plt.rc_context({"figure.dpi": 120}):
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                B_df, annot=True, fmt=".3f", cmap="Blues",
                linewidths=0.5, vmin=0, vmax=1,
                cbar_kws={"label": "隶属度"},
                ax=ax,
            )
            ax.set_title("模糊综合评价 — 综合隶属度矩阵",
                         fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("评价等级", fontsize=11)
            ax.set_ylabel("评价对象（按综合排名）", fontsize=11)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_stacked_bar(
        self,
        figsize: Tuple = (12, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制各对象等级隶属度的堆叠条形图。"""
        if self._B is None:
            raise RuntimeError("请先调用 compute()。")

        sorted_alts = self.result.index.tolist()
        B_arr = np.array([self._B[self._alternatives.index(a)] for a in sorted_alts])

        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(self.level_names)))
        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=figsize)
            bottom = np.zeros(len(sorted_alts))
            for k, (level, color) in enumerate(zip(self.level_names, colors)):
                ax.bar(
                    range(len(sorted_alts)), B_arr[:, k],
                    bottom=bottom, label=level, color=color,
                    edgecolor="white", linewidth=0.4,
                )
                bottom += B_arr[:, k]
            ax.set_xticks(range(len(sorted_alts)))
            ax.set_xticklabels(sorted_alts, rotation=40, ha="right", fontsize=10)
            ax.set_ylabel("综合隶属度", fontsize=12)
            ax.set_title("模糊综合评价 — 等级隶属度分布",
                         fontsize=13, fontweight="bold")
            ax.legend(title="等级", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def tex_description(self) -> str:
        levels_str = "、".join(self.level_names)
        scores_str = "、".join([str(s) for s in self.level_scores])
        return rf"""
\subsection{{模糊综合评价模型}}

\subsubsection{{模型原理}}

模糊综合评价法（Fuzzy Comprehensive Evaluation）基于模糊数学理论，
将定性评价定量化，适用于指标含义模糊、难以精确量化的评价问题。

\subsubsection{{评价集合}}

设因素集 $U = \{{u_1, u_2, \ldots, u_m\}}$，
评价集 $V = \{{{levels_str}\}}$，对应量化分值为 $\{{{scores_str}\}}$。

\subsubsection{{计算步骤}}

\noindent\textbf{{步骤1\quad 建立隶属度矩阵}}

对每个评价对象 $i$ 的每个指标 $j$，通过隶属函数确定其对各等级的隶属度，
得到模糊关系矩阵 $\boldsymbol{{R}}_i = (r_{{jk}})_{{m \times p}}$，
其中 $r_{{jk}}$ 表示指标 $j$ 对等级 $v_k$ 的隶属度。

\noindent\textbf{{步骤2\quad 模糊合成}}

采用加权平均算子（$M(\cdot, +)$）进行合成：
\begin{{equation}}
    b_{{ik}} = \sum_{{j=1}}^{{m}} w_j \cdot r_{{jk}},
    \quad \boldsymbol{{B}}_i = \boldsymbol{{W}} \cdot \boldsymbol{{R}}_i
    \label{{eq:fuzzy-compose}}
\end{{equation}}
其中权重向量 $\boldsymbol{{W}} = (w_1, w_2, \ldots, w_m)$，$\sum_j w_j = 1$。

\noindent\textbf{{步骤3\quad 去模糊化}}

采用加权平均法计算综合得分：
\begin{{equation}}
    S_i = \sum_{{k=1}}^{{p}} b_{{ik}} \cdot s_k
    \label{{eq:fuzzy-score}}
\end{{equation}}
其中 $s_k$ 为第 $k$ 等级的量化分值。
按 $S_i$ 降序排列得到最终综合排名。
"""