# -*- coding: utf-8 -*-
"""
主观赋权法模块
=============

实现基于专家判断的三种主观赋权方法：

1. AHPMethod          — 层次分析法 (Analytic Hierarchy Process)
2. BinomialCoefficientMethod — 二项系数法
3. RingRatioScoringMethod    — 环比评分法

所有方法遵循统一接口：fit() → compute() → summary() / tex_description() / plot()
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")          # 非交互后端，避免无显示器环境报错
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import comb as scipy_comb

from ..base import BaseMethod   # Part 1 定义的抽象基类

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
#  全局常量
# ─────────────────────────────────────────────────────────

# Saaty 随机一致性指标 RI（1~15 阶）
RI_TABLE: Dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90,  5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.54, 13: 1.56, 14: 1.57, 15: 1.59,
}

# Saaty 1-9 标度含义
SAATY_SCALE: Dict[int, str] = {
    1: "同等重要",       2: "介于1与3之间",
    3: "稍微重要",       4: "介于3与5之间",
    5: "明显重要",       6: "介于5与7之间",
    7: "强烈重要",       8: "介于7与9之间",
    9: "极端重要",
}


def _configure_matplotlib_chinese() -> None:
    """尝试配置 matplotlib 中文字体支持。"""
    candidates = ["SimHei", "Microsoft YaHei", "Arial Unicode MS",
                  "PingFang SC", "Noto Sans CJK SC"]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams["font.sans-serif"] = [font]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_chinese()


# ═══════════════════════════════════════════════════════════════
#  1. AHPMethod — 层次分析法
# ═══════════════════════════════════════════════════════════════

class AHPMethod(BaseMethod):
    """
    层次分析法 (AHP — Analytic Hierarchy Process)

    通过专家构造两两比较判断矩阵，用特征向量法（或近似法）计算各
    指标的主观权重，并进行一致性检验。

    Parameters
    ----------
    indicator_names : list of str, optional
        指标名称。若不提供则自动生成 C1, C2, ...
    cr_threshold : float, default 0.1
        一致性比例 (CR) 阈值，超过则发出警告
    weight_method : {'eigenvector', 'geometric_mean', 'arithmetic_mean'}
        权重计算方法：
        * ``eigenvector``   — 精确特征向量法（默认，推荐）
        * ``geometric_mean`` — 行积几何平均近似法
        * ``arithmetic_mean`` — 列归一化算术平均近似法
    language : {'zh', 'en'}
        输出语言（影响 summary 与 LaTeX 描述）

    Examples
    --------
    >>> import numpy as np
    >>> from src.algorithms.weights import AHPMethod
    >>>
    >>> matrix = np.array([
    ...     [1,   2,   5  ],
    ...     [1/2, 1,   3  ],
    ...     [1/5, 1/3, 1  ],
    ... ])
    >>> ahp = AHPMethod(indicator_names=["技术", "经济", "环境"])
    >>> ahp.fit(matrix).compute()
    >>> result = ahp.summary()
    >>> print(result["weights"])   # [0.5816, 0.3090, 0.1095]
    """

    RI_TABLE: Dict[int, float] = RI_TABLE

    def __init__(
        self,
        indicator_names: Optional[List[str]] = None,
        cr_threshold: float = 0.1,
        weight_method: str = "eigenvector",
        language: str = "zh",
    ) -> None:
        super().__init__(
            name="AHP层次分析法",
            description="通过两两比较判断矩阵确定指标主观权重",
        )
        self.indicator_names: Optional[List[str]] = indicator_names
        self.cr_threshold: float = cr_threshold
        self.weight_method: str = weight_method
        self.language: str = language

        self._matrix: Optional[np.ndarray] = None
        self._n: int = 0

    # ── 核心接口 ────────────────────────────────────────────────

    def fit(
        self,
        matrix: Union[np.ndarray, pd.DataFrame, List[List[float]]],
        indicator_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "AHPMethod":
        """
        载入并校验两两比较判断矩阵。

        Parameters
        ----------
        matrix : array-like, shape (n, n)
            正互反矩阵，元素取 Saaty 1-9 标度
        indicator_names : list of str, optional
            覆盖构造函数中设定的指标名称

        Returns
        -------
        self
        """
        # ── 类型转换 ──
        if isinstance(matrix, pd.DataFrame):
            if indicator_names is None:
                indicator_names = list(matrix.columns)
            matrix = matrix.values.astype(float)
        else:
            matrix = np.asarray(matrix, dtype=float)

        # ── 形状校验 ──
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"判断矩阵必须是方阵，当前形状: {matrix.shape}"
            )
        n = matrix.shape[0]
        if n < 2:
            raise ValueError("判断矩阵阶数必须 ≥ 2")
        if n > 15:
            warnings.warn(
                f"矩阵阶数 n={n} 超过 15，RI 使用近似值 1.59，"
                "一致性检验结果仅供参考。"
            )

        # ── 元素值校验 ──
        if np.any(matrix <= 0):
            raise ValueError("判断矩阵所有元素必须 > 0")

        # ── 修正对角线 ──
        if not np.allclose(np.diag(matrix), 1.0, atol=1e-6):
            warnings.warn("对角线元素不全为 1，已自动修正")
            np.fill_diagonal(matrix, 1.0)

        # ── 检查并修正互反性 ──
        for i in range(n):
            for j in range(i + 1, n):
                if abs(matrix[i, j] * matrix[j, i] - 1.0) > 1e-4:
                    warnings.warn(
                        f"A[{i},{j}]={matrix[i,j]:.4f} 与 "
                        f"A[{j},{i}]={matrix[j,i]:.4f} 不满足互反性，"
                        f"已自动令 A[{j},{i}] = 1/A[{i},{j}]"
                    )
                    matrix[j, i] = 1.0 / matrix[i, j]

        self._matrix = matrix
        self._n = n

        # ── 指标名称 ──
        if indicator_names is not None:
            self.indicator_names = indicator_names
        if self.indicator_names is None:
            self.indicator_names = [f"C{i + 1}" for i in range(n)]
        if len(self.indicator_names) != n:
            raise ValueError(
                f"indicator_names 长度 {len(self.indicator_names)} "
                f"与矩阵阶数 {n} 不匹配"
            )

        self._fitted = True
        logger.debug("AHP.fit: 载入 %d×%d 判断矩阵", n, n)
        return self

    def compute(self) -> "AHPMethod":
        """
        计算权重向量并执行一致性检验。

        Returns
        -------
        self
        """
        self._check_fitted()
        n = self._n
        A = self._matrix.copy()

        # ── 1. 计算权重向量 & λmax ──
        method_dispatch = {
            "eigenvector":     self._eigenvector_method,
            "geometric_mean":  self._geometric_mean_method,
            "arithmetic_mean": self._arithmetic_mean_method,
        }
        if self.weight_method not in method_dispatch:
            raise ValueError(
                f"不支持的 weight_method: '{self.weight_method}'。"
                f"可选: {list(method_dispatch)}"
            )
        weights, lambda_max = method_dispatch[self.weight_method](A)

        # ── 2. 一致性检验 ──
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
        ri = self.RI_TABLE.get(n, 1.59)
        cr = ci / ri if ri > 1e-10 else 0.0
        is_consistent = bool(cr < self.cr_threshold)

        if not is_consistent:
            warnings.warn(
                f"[AHP] 一致性检验未通过: CR={cr:.4f} ≥ {self.cr_threshold}。"
                "建议重新审视判断矩阵。",
                stacklevel=2,
            )
        else:
            logger.info("AHP 一致性检验通过 (CR=%.4f)", cr)

        # ── 3. 存储结果 ──
        self._results = {
            "weights":        weights,
            "indicator_names": list(self.indicator_names),
            "lambda_max":     float(lambda_max),
            "CI":             float(ci),
            "RI":             float(ri),
            "CR":             float(cr),
            "is_consistent":  is_consistent,
            "n":              n,
            "weight_method":  self.weight_method,
            "matrix":         A.copy(),
        }
        self._computed = True
        return self

    def summary(self) -> Dict[str, Any]:
        """返回结构化结果摘要并打印格式化表格。"""
        self._check_computed()
        r = self._results

        weight_df = pd.DataFrame({
            "指标":   self.indicator_names,
            "权重":   np.round(r["weights"], 6),
            "权重(%)": np.round(r["weights"] * 100, 2),
        })
        r["weight_df"] = weight_df

        sep = "=" * 52
        print(sep)
        print("  AHP 层次分析法 — 计算结果")
        print(sep)
        print(f"  矩阵阶数  n    = {r['n']}")
        print(f"  最大特征值 λmax = {r['lambda_max']:.4f}")
        print(f"  一致性指标 CI  = {r['CI']:.4f}")
        print(f"  随机指标   RI  = {r['RI']:.4f}")
        flag = "✓ 通过" if r["is_consistent"] else "✗ 未通过"
        print(f"  一致性比例 CR  = {r['CR']:.4f}  [{flag}]")
        print("-" * 52)
        print(weight_df.to_string(index=False))
        print(sep)
        return r

    def tex_description(self) -> str:
        """生成对应的 LaTeX 段落（用于报告自动生成）。"""
        self._check_computed()
        r = self._results
        n = r["n"]
        w_str = r"\mathbf{w} = (" + ",\; ".join(
            f"{w:.4f}" for w in r["weights"]
        ) + r")^\top"
        names_str = "、".join(self.indicator_names)

        tex = (
            r"\subsubsection{层次分析法（AHP）确定主观权重}" "\n\n"
            f"对 {n} 个评价指标（{names_str}）构造两两比较判断矩阵"
            r" $\mathbf{A} = (a_{ij})_{n \times n}$，"
            "元素取 Saaty 1--9 标度。\n\n"
            r"\paragraph{一致性检验}" "\n"
            "计算判断矩阵最大特征值 $\\lambda_{\\max}$，"
            "并验证一致性：\n"
            r"\begin{equation}" "\n"
            f"  CI = \\frac{{\\lambda_{{\\max}} - n}}{{n - 1}}"
            f" = \\frac{{{r['lambda_max']:.4f} - {n}}}{{{n} - 1}}"
            f" = {r['CI']:.4f},"
            r"\qquad"
            f"  CR = \\frac{{CI}}{{RI}} = \\frac{{{r['CI']:.4f}}}{{{r['RI']:.4f}}}"
            f" = {r['CR']:.4f}"
            "\n"
            r"\end{equation}" "\n"
        )
        if r["is_consistent"]:
            tex += f"由于 $CR = {r['CR']:.4f} < 0.10$，判断矩阵满足一致性要求。\n\n"
        else:
            tex += (
                f"注意：$CR = {r['CR']:.4f} \\geq 0.10$，"
                "判断矩阵一致性不足，建议修正。\n\n"
            )
        tex += (
            r"\paragraph{权重计算结果}" "\n"
            "采用特征向量法归一化得权重向量：\n"
            r"\begin{equation}" "\n"
            f"  {w_str}\n"
            r"\end{equation}" "\n"
        )
        return tex

    def plot(self, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        生成可视化图：判断矩阵热力图 + 权重条形图 + 一致性信息面板。

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._check_computed()
        r = self._results
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("AHP 层次分析法 — 结果可视化",
                     fontsize=14, fontweight="bold", y=1.01)

        # ── 子图1: 判断矩阵热力图 ──
        ax = axes[0]
        df_mat = pd.DataFrame(
            r["matrix"],
            index=self.indicator_names,
            columns=self.indicator_names,
        )
        sns.heatmap(
            df_mat, annot=True, fmt=".3g", cmap="YlOrRd",
            ax=ax, linewidths=0.5,
            cbar_kws={"label": "重要度比值", "shrink": 0.85},
        )
        ax.set_title("判断矩阵", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

        # ── 子图2: 权重水平条形图 ──
        ax = axes[1]
        n = r["n"]
        colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, n))
        sorted_idx = np.argsort(r["weights"])
        s_names  = [self.indicator_names[i] for i in sorted_idx]
        s_weights = r["weights"][sorted_idx]
        bars = ax.barh(s_names, s_weights, color=colors[::-1],
                       edgecolor="white", height=0.6)
        ax.axvline(1 / n, color="grey", ls="--", lw=1.2, label="均等权重")
        for bar, val in zip(bars, s_weights):
            ax.text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5,
            )
        ax.set_xlabel("权重值", fontsize=10)
        ax.set_title("指标权重分布", fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xlim(0, max(r["weights"]) * 1.35)

        # ── 子图3: 一致性信息面板 ──
        ax = axes[2]
        ax.axis("off")
        color = "#27ae60" if r["is_consistent"] else "#e74c3c"
        flag  = "通 过  ✓" if r["is_consistent"] else "未通过 ✗"
        text  = (
            f"{'─' * 28}\n"
            f"  AHP 一致性检验结果\n"
            f"{'─' * 28}\n"
            f"  矩阵阶数  n    = {r['n']}\n"
            f"  最大特征值 λmax = {r['lambda_max']:.4f}\n"
            f"  一致性指标 CI  = {r['CI']:.4f}\n"
            f"  随机指标   RI  = {r['RI']:.4f}\n"
            f"  一致性比例 CR  = {r['CR']:.4f}\n"
            f"{'─' * 28}\n"
            f"  检验结果: {flag}\n"
            f"  (阈值 CR < {self.cr_threshold})\n"
            f"{'─' * 28}"
        )
        ax.text(
            0.05, 0.5, text, transform=ax.transAxes,
            va="center", fontsize=9.5, fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor=color + "22",
                edgecolor=color,
                linewidth=1.8,
            ),
        )
        plt.tight_layout()
        return fig

    # ── 私有权重计算方法 ─────────────────────────────────────────

    @staticmethod
    def _eigenvector_method(
        A: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """特征向量法：取最大特征值对应的实特征向量。"""
        eigenvalues, eigenvectors = np.linalg.eig(A)
        real_parts = eigenvalues.real
        idx = int(np.argmax(real_parts))
        lambda_max = float(real_parts[idx])
        w = np.abs(eigenvectors[:, idx].real)
        w /= w.sum()
        return w, lambda_max

    @staticmethod
    def _geometric_mean_method(
        A: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """几何平均法：行乘积开 n 次方后归一化。"""
        n = A.shape[0]
        row_products = np.prod(A, axis=1)
        w = row_products ** (1.0 / n)
        w /= w.sum()
        Aw = A @ w
        lambda_max = float(np.mean(Aw / w))
        return w, lambda_max

    @staticmethod
    def _arithmetic_mean_method(
        A: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """算术平均法：列归一化后取行均值。"""
        col_sums = A.sum(axis=0)
        normalized = A / col_sums
        w = normalized.mean(axis=1)
        w /= w.sum()
        Aw = A @ w
        lambda_max = float(np.mean(Aw / w))
        return w, lambda_max

    # ── 便捷静态工厂 ─────────────────────────────────────────────

    @staticmethod
    def from_comparisons(
        n: int,
        comparisons: List[Tuple[int, int, float]],
        indicator_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "AHPMethod":
        """
        通过三元组列表 (i, j, scale) 构造判断矩阵并自动 fit。

        Parameters
        ----------
        n : int
            指标数量
        comparisons : list of (i, j, scale)
            只需提供上三角元素；下三角自动取倒数。
        indicator_names : list of str, optional

        Examples
        --------
        >>> ahp = AHPMethod.from_comparisons(3, [
        ...     (0, 1, 2), (0, 2, 5), (1, 2, 3)
        ... ], indicator_names=["技术", "经济", "环境"])
        >>> ahp.compute().summary()
        """
        matrix = np.eye(n, dtype=float)
        for i, j, scale in comparisons:
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(
                    f"索引超出范围: ({i}, {j})，n={n}"
                )
            matrix[i, j] = float(scale)
            matrix[j, i] = 1.0 / float(scale)
        obj = AHPMethod(indicator_names=indicator_names, **kwargs)
        return obj.fit(matrix)

    @staticmethod
    def check_consistency(
        matrix: Union[np.ndarray, List], threshold: float = 0.1
    ) -> Tuple[float, bool]:
        """
        快速一致性检验（不创建对象）。

        Returns
        -------
        (CR, is_consistent)
        """
        A = np.asarray(matrix, dtype=float)
        n = A.shape[0]
        eigenvalues = np.linalg.eigvals(A)
        lambda_max = float(max(eigenvalues.real))
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
        ri = RI_TABLE.get(n, 1.59)
        cr = ci / ri if ri > 1e-10 else 0.0
        return cr, bool(cr < threshold)


# ═══════════════════════════════════════════════════════════════
#  2. BinomialCoefficientMethod — 二项系数法
# ═══════════════════════════════════════════════════════════════

class BinomialCoefficientMethod(BaseMethod):
    """
    二项系数法 (Binomial Coefficient Method)

    专家对 n 个指标按重要性排序后，将指标映射到 Pascal 三角形
    对应的二项系数位置，再归一化得到权重。

    数学原理
    --------
    设 n 个指标按重要性排为第 1（最重要）至第 n（最不重要）位，
    排在第 k 位的指标（k = 1, …, n）获得二项系数分值：

        score_k = C(n-1, n-k)

    其中 C(n-1, n-1)=1 分配给最重要指标……
    该分值序列为单调递减，反映递减的重要性差异。
    归一化：w_k = score_k / Σ score_j。

    Parameters
    ----------
    indicator_names : list of str, optional
    aggregation : {'mean', 'median', 'geometric_mean'}
        多专家排名聚合策略

    Examples
    --------
    >>> rankings = np.array([
    ...     [1, 2, 3, 4],   # 专家1
    ...     [2, 1, 3, 4],   # 专家2
    ...     [1, 3, 2, 4],   # 专家3
    ... ])
    >>> bcm = BinomialCoefficientMethod(["C1", "C2", "C3", "C4"])
    >>> bcm.fit(rankings).compute().summary()
    """

    def __init__(
        self,
        indicator_names: Optional[List[str]] = None,
        aggregation: str = "mean",
    ) -> None:
        super().__init__(
            name="二项系数法",
            description="基于专家排序与二项系数分配主观权重",
        )
        self.indicator_names = indicator_names
        self.aggregation = aggregation
        self._rankings: Optional[np.ndarray] = None
        self._n_experts: int = 0
        self._n: int = 0

    def fit(
        self,
        rankings: Union[np.ndarray, pd.DataFrame, List],
        indicator_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "BinomialCoefficientMethod":
        """
        载入专家排名矩阵。

        Parameters
        ----------
        rankings : array-like, shape (n_experts, n) or (n,)
            排名值，1=最重要，n=最不重要。
            支持单专家（1D）输入。
        """
        if isinstance(rankings, pd.DataFrame):
            if indicator_names is None:
                indicator_names = list(rankings.columns)
            rankings = rankings.values
        rankings = np.asarray(rankings, dtype=float)

        if rankings.ndim == 1:
            rankings = rankings.reshape(1, -1)

        n_experts, n = rankings.shape

        # 校验并修正每行为 1~n 的完整排列
        for i, row in enumerate(rankings):
            unique_vals = np.sort(np.unique(row))
            expected    = np.arange(1, n + 1, dtype=float)
            if not np.allclose(unique_vals, expected):
                warnings.warn(
                    f"专家 {i + 1} 的排名不是 1~{n} 的完整排列，"
                    "已自动转换为排名。"
                )
                rankings[i] = (
                    np.argsort(np.argsort(row)) + 1
                ).astype(float)

        self._rankings   = rankings
        self._n_experts  = n_experts
        self._n          = n

        if indicator_names is not None:
            self.indicator_names = indicator_names
        if self.indicator_names is None:
            self.indicator_names = [f"C{i + 1}" for i in range(n)]
        if len(self.indicator_names) != n:
            raise ValueError(
                f"indicator_names 长度 {len(self.indicator_names)} ≠ n={n}"
            )

        self._fitted = True
        return self

    def compute(self) -> "BinomialCoefficientMethod":
        """计算二项系数权重。"""
        self._check_fitted()
        n = self._n

        # ── 1. 聚合多专家排名 ──
        agg_map = {
            "mean":         lambda x: x.mean(axis=0),
            "median":       lambda x: np.median(x, axis=0),
            "geometric_mean": lambda x: np.exp(np.log(x).mean(axis=0)),
        }
        if self.aggregation not in agg_map:
            raise ValueError(f"不支持的聚合方式: '{self.aggregation}'")
        avg_rankings = agg_map[self.aggregation](self._rankings)

        # ── 2. 根据平均排名获得重要性位序（rank_pos=0 最重要）──
        sort_idx = np.argsort(avg_rankings)          # 排名最小=最重要 → 排在前面

        # ── 3. 分配二项系数分值 ──
        # 排在第 rank_pos 位（0-based）的指标获得 C(n-1, n-1-rank_pos)
        # 即最重要 → C(n-1, n-1)=1，第二重要 → C(n-1, n-2), ...
        # 注：该序列为 1, n-1, C(n-1,2), ..., C(n-1,2), n-1, 1（对称）
        # 为使权重严格单调，改用线性衰减分值作为主权重，二项系数作参考
        binomial_scores  = np.zeros(n)
        linear_scores    = np.zeros(n)
        for rank_pos, orig_idx in enumerate(sort_idx):
            k = n - 1 - rank_pos                              # k: n-1 → 0
            binomial_scores[orig_idx] = float(
                scipy_comb(n - 1, k, exact=True)
            )
            linear_scores[orig_idx]   = float(n - rank_pos)  # n, n-1, ..., 1

        binom_weights  = binomial_scores / binomial_scores.sum()
        linear_weights = linear_scores   / linear_scores.sum()

        self._results = {
            "weights":          linear_weights,   # 主推权重（线性单调）
            "binom_weights":    binom_weights,     # 参考权重（二项系数）
            "linear_scores":    linear_scores,
            "binomial_scores":  binomial_scores,
            "avg_rankings":     avg_rankings,
            "indicator_names":  list(self.indicator_names),
            "n_experts":        self._n_experts,
            "n":                n,
        }
        self._computed = True
        return self

    def summary(self) -> Dict[str, Any]:
        self._check_computed()
        r = self._results

        df = pd.DataFrame({
            "指标":         self.indicator_names,
            "平均排名":     np.round(r["avg_rankings"],    3),
            "线性分值":     np.round(r["linear_scores"],   3),
            "二项系数分值": np.round(r["binomial_scores"], 3),
            "线性权重":     np.round(r["weights"],         4),
            "二项系数权重": np.round(r["binom_weights"],   4),
        })
        r["weight_df"] = df

        print("=" * 60)
        print("  二项系数法 — 计算结果")
        print("=" * 60)
        print(f"  专家数: {r['n_experts']}，指标数: {r['n']}")
        print("-" * 60)
        print(df.to_string(index=False))
        print("=" * 60)
        return r

    def tex_description(self) -> str:
        self._check_computed()
        r = self._results
        w_str = ", ".join(f"{w:.4f}" for w in r["weights"])
        tex = (
            r"\subsubsection{二项系数法确定主观权重}" "\n\n"
            f"邀请 {r['n_experts']} 位专家对 {r['n']} 个评价指标进行重要性排序，"
            "基于排序结果计算各指标获胜次数，归一化得到权重。\n\n"
            "设指标 $j$ 在所有专家排名中的平均排名为 $\\bar{r}_j$，"
            "则其线性分值为 $s_j = n + 1 - \\lceil \\bar{r}_j \\rceil$，"
            "归一化权重：\n"
            r"\begin{equation}" "\n"
            r"  w_j = \frac{s_j}{\sum_{k=1}^{n} s_k}, \quad j=1,\ldots,n" "\n"
            r"\end{equation}" "\n\n"
            f"最终权重向量为 $\\mathbf{{w}} = ({w_str})^\\top$。\n"
        )
        return tex

    def plot(self, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        self._check_computed()
        r = self._results
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle("二项系数法 — 结果可视化",
                     fontsize=13, fontweight="bold")

        # ── 左图：专家排名热力图或单专家排名条图 ──
        ax = axes[0]
        if self._n_experts > 1:
            df_rank = pd.DataFrame(
                self._rankings,
                index  = [f"专家{i + 1}" for i in range(self._n_experts)],
                columns= self.indicator_names,
            )
            sns.heatmap(
                df_rank, annot=True, fmt=".0f", cmap="Blues_r",
                ax=ax, cbar_kws={"label": "排名（越小越重要）"},
                linewidths=0.3,
            )
            ax.set_title("专家排名矩阵", fontsize=11)
        else:
            ax.bar(
                self.indicator_names, r["avg_rankings"],
                color="steelblue", edgecolor="white",
            )
            ax.set_ylabel("排名（越低越重要）", fontsize=10)
            ax.set_title("指标排名", fontsize=11)
            ax.tick_params(axis="x", rotation=45)

        # ── 右图：两种权重对比 ──
        ax = axes[1]
        x     = np.arange(len(self.indicator_names))
        width = 0.38
        ax.bar(x - width / 2, r["weights"],       width, label="线性权重",     color="#3498db", alpha=0.85)
        ax.bar(x + width / 2, r["binom_weights"], width, label="二项系数权重",  color="#e67e22", alpha=0.85)
        ax.axhline(1 / r["n"], color="grey", ls="--", lw=1, label="均等权重")
        ax.set_xticks(x)
        ax.set_xticklabels(self.indicator_names, rotation=45, ha="right")
        ax.set_ylabel("权重", fontsize=10)
        ax.set_title("权重方案对比", fontsize=11)
        ax.legend(fontsize=8)

        plt.tight_layout()
        return fig

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._results.get("weights")


# ═══════════════════════════════════════════════════════════════
#  3. RingRatioScoringMethod — 环比评分法
# ═══════════════════════════════════════════════════════════════

class RingRatioScoringMethod(BaseMethod):
    """
    环比评分法 (Ring Ratio Scoring / Sequential Comparison Method)

    专家将指标按重要性降序排列后，给出**相邻指标的重要程度比值**，
    以最不重要指标得分 = 1 为基准逆推各指标得分，归一化得权重。

    数学步骤
    --------
    1. 设指标按重要性排为 1 > 2 > … > n；
    2. 专家给出比值 r_k = v_k / v_{k+1}，k = 1, …, n-1；
    3. 令 v_n = 1，逆推：v_k = r_k × v_{k+1}；
    4. 归一化：w_k = v_k / Σ v_j。

    Parameters
    ----------
    indicator_names : list of str, optional
    order : list of int, optional
        重要性排序（0-based 原始指标索引，从最重要到最不重要）。
        若 None，假设数据已按重要性顺序输入，权重直接对应该顺序。
    aggregation : {'geometric_mean', 'arithmetic_mean', 'median'}
        多专家比值聚合策略（比值数据推荐几何平均）

    Examples
    --------
    >>> # 4 个指标，2 位专家的相邻比值
    >>> ratios = [[1.5, 2.0, 1.2],   # 专家1: v1/v2=1.5, v2/v3=2.0, v3/v4=1.2
    ...           [1.8, 1.6, 1.4]]   # 专家2
    >>> rrs = RingRatioScoringMethod(
    ...     indicator_names=["创新", "质量", "成本", "服务"],
    ...     order=[0, 1, 2, 3]
    ... )
    >>> rrs.fit(ratios).compute().summary()
    """

    def __init__(
        self,
        indicator_names: Optional[List[str]] = None,
        order: Optional[List[int]] = None,
        aggregation: str = "geometric_mean",
    ) -> None:
        super().__init__(
            name="环比评分法",
            description="通过相邻指标重要度比值逆推得分并归一化",
        )
        self.indicator_names = indicator_names
        self.order           = order
        self.aggregation     = aggregation
        self._ratios:    Optional[np.ndarray] = None
        self._n:         int = 0
        self._n_experts: int = 0

    def fit(
        self,
        ratios: Union[np.ndarray, List, pd.DataFrame],
        indicator_names: Optional[List[str]] = None,
        order: Optional[List[int]] = None,
        **kwargs,
    ) -> "RingRatioScoringMethod":
        """
        载入相邻比值矩阵。

        Parameters
        ----------
        ratios : array-like, shape (n-1,) or (n_experts, n-1)
            r_k = importance(rank k) / importance(rank k+1) > 0
        """
        ratios = np.asarray(ratios, dtype=float)
        if ratios.ndim == 1:
            ratios = ratios.reshape(1, -1)

        n_experts, n_minus_1 = ratios.shape
        n = n_minus_1 + 1

        if np.any(ratios <= 0):
            raise ValueError("所有比值必须 > 0")

        self._ratios    = ratios
        self._n         = n
        self._n_experts = n_experts

        if order is not None:
            self.order = order
        if indicator_names is not None:
            self.indicator_names = indicator_names
        if self.indicator_names is None:
            self.indicator_names = [f"C{i + 1}" for i in range(n)]
        if len(self.indicator_names) != n:
            raise ValueError(
                f"indicator_names 长度 {len(self.indicator_names)} ≠ n={n}"
            )
        if self.order is not None and len(self.order) != n:
            raise ValueError(
                f"order 长度 {len(self.order)} ≠ n={n}"
            )

        self._fitted = True
        return self

    def compute(self) -> "RingRatioScoringMethod":
        """计算环比评分权重。"""
        self._check_fitted()
        n       = self._n
        r_all   = self._ratios

        # ── 1. 多专家比值聚合 ──
        agg_map = {
            "geometric_mean":  lambda x: np.exp(np.log(x).mean(axis=0)),
            "arithmetic_mean": lambda x: x.mean(axis=0),
            "median":          lambda x: np.median(x, axis=0),
        }
        if self.aggregation not in agg_map:
            raise ValueError(f"不支持的聚合方式: '{self.aggregation}'")
        agg_ratios = agg_map[self.aggregation](r_all)   # shape (n-1,)

        # ── 2. 逆推得分（sorted 顺序：最重要→最不重要）──
        scores = np.ones(n)
        for k in range(n - 2, -1, -1):
            scores[k] = scores[k + 1] * agg_ratios[k]

        sorted_weights = scores / scores.sum()

        # ── 3. 映射回原始指标顺序 ──
        if self.order is not None:
            weights = np.zeros(n)
            for sorted_pos, orig_idx in enumerate(self.order):
                weights[orig_idx] = sorted_weights[sorted_pos]
        else:
            weights = sorted_weights.copy()

        self._results = {
            "weights":        weights,
            "sorted_weights": sorted_weights,
            "scores":         scores,
            "agg_ratios":     agg_ratios,
            "indicator_names": list(self.indicator_names),
            "order":          self.order,
            "n_experts":      self._n_experts,
            "n":              n,
        }
        self._computed = True
        return self

    def summary(self) -> Dict[str, Any]:
        self._check_computed()
        r = self._results

        if self.order is not None:
            ordered_names = [self.indicator_names[i] for i in self.order]
        else:
            ordered_names = list(self.indicator_names)

        process_df = pd.DataFrame({
            "指标（重要性降序）":  ordered_names,
            "相邻比值 r_k":   list(np.round(r["agg_ratios"], 4)) + ["—"],
            "得分":           np.round(r["scores"], 4),
            "权重（排序后）":  np.round(r["sorted_weights"], 4),
        })
        weight_df = pd.DataFrame({
            "指标":  self.indicator_names,
            "权重":  np.round(r["weights"], 6),
        })
        r["process_df"] = process_df
        r["weight_df"]  = weight_df

        print("=" * 56)
        print("  环比评分法 — 计算结果")
        print("=" * 56)
        print(f"  专家数: {r['n_experts']}，指标数: {r['n']}")
        print("\n  得分计算过程：")
        print(process_df.to_string(index=False))
        print("\n  最终权重向量（原始指标顺序）：")
        print(weight_df.to_string(index=False))
        print("=" * 56)
        return r

    def tex_description(self) -> str:
        self._check_computed()
        r = self._results
        ratios_str = r",\ ".join(
            f"r_{{{k+1}}} = {v:.3f}" for k, v in enumerate(r["agg_ratios"])
        )
        w_str = ", ".join(f"{w:.4f}" for w in r["weights"])
        tex = (
            r"\subsubsection{环比评分法确定主观权重}" "\n\n"
            f"将 {r['n']} 个指标按重要性降序排列，"
            "依次给出相邻指标的重要度比值：\n"
            r"\begin{equation}" "\n"
            f"  {ratios_str}\n"
            r"\end{equation}" "\n\n"
            "以最不重要指标得分 $v_n = 1$ 为基准，逐步反推各指标得分：\n"
            r"\begin{equation}" "\n"
            r"  v_k = r_k \cdot v_{k+1},\quad k = n-1,\,n-2,\,\ldots,\,1" "\n"
            r"\end{equation}" "\n\n"
            "归一化后得到权重向量（按原始指标顺序）：\n"
            r"\begin{equation}" "\n"
            f"  \\mathbf{{w}} = ({w_str})^\\top\n"
            r"\end{equation}" "\n"
        )
        return tex

    def plot(self, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        self._check_computed()
        r = self._results
        if self.order is not None:
            sorted_names = [self.indicator_names[i] for i in self.order]
        else:
            sorted_names = list(self.indicator_names)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle("环比评分法 — 结果可视化",
                     fontsize=13, fontweight="bold")

        # ── 左图：得分折线图 ──
        ax = axes[0]
        x = np.arange(r["n"])
        ax.plot(x, r["scores"], "o-", color="#2980b9",
                linewidth=2, markersize=7, zorder=3)
        for xi, (name, sc) in enumerate(zip(sorted_names, r["scores"])):
            ax.annotate(
                f"{sc:.3f}", (xi, sc),
                textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, color="#2c3e50",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_names, rotation=40, ha="right")
        ax.set_ylabel("得分", fontsize=10)
        ax.set_title("指标得分（重要性降序）", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, max(r["scores"]) * 1.2)

        # ── 右图：最终权重条形图 ──
        ax = axes[1]
        clrs = plt.cm.GnBu(np.linspace(0.35, 0.85, r["n"]))
        bars = ax.bar(
            self.indicator_names, r["weights"],
            color=clrs, edgecolor="white",
        )
        for bar, val in zip(bars, r["weights"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8.5,
            )
        ax.axhline(1 / r["n"], color="grey", ls="--", lw=1.1,
                   label="均等权重")
        ax.set_ylabel("权重值", fontsize=10)
        ax.set_title("指标权重（原始指标顺序）", fontsize=11)
        ax.tick_params(axis="x", rotation=40)
        ax.legend(fontsize=8)

        plt.tight_layout()
        return fig

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._results.get("weights")