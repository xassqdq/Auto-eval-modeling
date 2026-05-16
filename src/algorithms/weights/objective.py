# -*- coding: utf-8 -*-
"""
客观赋权法模块
=============

实现四种数据驱动的客观赋权方法：

1. EntropyWeightMethod  — 熵权法
2. CRITICMethod         — CRITIC 法（对比强度 + 冲突性）
3. StdDeviationMethod   — 标准离差法
4. PCAWeightMethod      — 主成分分析法确定权重

所有方法接受已完成**正向化**处理的决策矩阵作为输入
（各指标值 ≥ 0，数值越大越优）。
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..base import BaseMethod

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
#  内部归一化工具
# ─────────────────────────────────────────────────────────

def _minmax_normalize(X: np.ndarray) -> np.ndarray:
    """按列 Min-Max 归一化到 [0, 1]。列极差为 0 时保留原列（全 0.5）。"""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = X_max - X_min
    denom_safe = np.where(denom < 1e-12, 1.0, denom)
    X_norm = (X - X_min) / denom_safe
    X_norm[:, denom < 1e-12] = 0.5  # 无差异列置 0.5
    return X_norm


def _proportion_normalize(X: np.ndarray) -> np.ndarray:
    """按列比例归一化：p_ij = x_ij / Σ_i x_ij（列和归一）。
    要求所有元素 > 0（经过正向化处理后满足）。
    极小列处理：将负列整体平移后再归一。
    """
    X_shifted = X.copy()
    col_mins = X_shifted.min(axis=0)
    neg_cols = col_mins <= 0
    if neg_cols.any():
        warnings.warn(
            "部分列含 ≤ 0 的值，已自动平移至正数区间再做比例归一。"
            "建议在预处理阶段完成正向化。",
            stacklevel=4,
        )
        X_shifted[:, neg_cols] -= col_mins[neg_cols] - 1e-8
    col_sums = X_shifted.sum(axis=0)
    col_sums_safe = np.where(col_sums < 1e-12, 1.0, col_sums)
    return X_shifted / col_sums_safe


def _configure_zh_font() -> None:
    candidates = ["SimHei", "Microsoft YaHei", "Arial Unicode MS",
                  "PingFang SC", "Noto Sans CJK SC"]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams["font.sans-serif"] = [font]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_zh_font()


# ═══════════════════════════════════════════════════════════════
#  1. EntropyWeightMethod — 熵权法
# ═══════════════════════════════════════════════════════════════

class EntropyWeightMethod(BaseMethod):
    """
    熵权法 (Entropy Weight Method)

    利用信息熵度量各指标数据的**离散程度**，
    信息熵越小（越有序）→ 指标差异越大 → 权重越高。

    计算步骤
    --------
    设决策矩阵 :math:`X = (x_{ij})_{m \\times n}`，

    1. 比例归一化：

       .. math:: p_{ij} = x_{ij} / \\sum_{i=1}^{m} x_{ij}

    2. 计算信息熵：

       .. math:: e_j = -\\frac{1}{\\ln m} \\sum_{i=1}^{m} p_{ij} \\ln p_{ij}

       约定 :math:`0 \\cdot \\ln 0 = 0`。

    3. 差异系数与权重：

       .. math:: d_j = 1 - e_j, \\quad w_j = d_j / \\sum_{k=1}^{n} d_k

    Parameters
    ----------
    indicator_names : list of str, optional
    epsilon : float, default 1e-12
        数值稳定性小量（防止 log(0)）

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from src.algorithms.weights import EntropyWeightMethod
    >>>
    >>> data = pd.DataFrame({
    ...     "R&D投入": [2.1, 3.5, 1.8, 4.2, 2.9],
    ...     "专利数":  [120, 250, 90, 310, 180],
    ...     "GDP增长": [6.2, 7.1, 5.8, 8.3, 6.9],
    ... })
    >>> ew = EntropyWeightMethod()
    >>> ew.fit(data).compute().summary()
    """

    def __init__(
            self,
            indicator_names: Optional[List[str]] = None,
            epsilon: float = 1e-12,
    ) -> None:
        super().__init__(
            name="熵权法",
            description="基于信息熵度量指标离散性，客观确定权重",
        )
        self.indicator_names = indicator_names
        self.epsilon = epsilon
        self._X: Optional[np.ndarray] = None
        self._m: int = 0
        self._n: int = 0

    def fit(
            self,
            data: Union[np.ndarray, pd.DataFrame],
            indicator_names: Optional[List[str]] = None,
            **kwargs,
    ) -> "EntropyWeightMethod":
        """
        载入决策矩阵。

        Parameters
        ----------
        data : array-like, shape (m, n)
            m 个评价对象 × n 个指标，值已正向化（越大越优，≥ 0）
        """
        if isinstance(data, pd.DataFrame):
            if indicator_names is None:
                indicator_names = list(data.columns)
            data = data.values.astype(float)
        else:
            data = np.asarray(data, dtype=float)

        if data.ndim != 2:
            raise ValueError("data 必须是 2D 矩阵，shape=(m, n)")
        if data.shape[0] < 2:
            raise ValueError("评价对象数 m 必须 ≥ 2")

        self._X = data
        self._m, self._n = data.shape

        if indicator_names is not None:
            self.indicator_names = indicator_names
        if self.indicator_names is None:
            self.indicator_names = [f"C{j + 1}" for j in range(self._n)]
        if len(self.indicator_names) != self._n:
            raise ValueError(
                f"indicator_names 长度 {len(self.indicator_names)} ≠ n={self._n}"
            )

        self._fitted = True
        return self

    def compute(self) -> "EntropyWeightMethod":
        """执行熵权法计算。"""
        self._check_fitted()
        m, n = self._m, self._n
        eps = self.epsilon

        # ── 1. 比例归一化 ──
        P = _proportion_normalize(self._X)  # shape (m, n)

        # ── 2. 信息熵 e_j ──
        # safe_log: 0 * ln(0) = 0
        log_P = np.where(P > eps, np.log(P + eps), 0.0)
        entropy = -(1.0 / np.log(m)) * (P * log_P).sum(axis=0)
        # 截断到 [0, 1]
        entropy = np.clip(entropy, 0.0, 1.0)

        # ── 3. 差异系数 & 权重 ──
        d = 1.0 - entropy
        d_sum = d.sum()
        if d_sum < 1e-10:
            warnings.warn(
                "所有指标的差异系数之和接近 0，数据可能无区分度，"
                "将返回均等权重。"
            )
            weights = np.ones(n) / n
        else:
            weights = d / d_sum

        self._results = {
            "weights": weights,
            "entropy": entropy,
            "diff_coeff": d,
            "P_matrix": P,
            "indicator_names": list(self.indicator_names),
            "m": m,
            "n": n,
        }
        self._computed = True
        return self

    def summary(self) -> Dict[str, Any]:
        self._check_computed()
        r = self._results
        df = pd.DataFrame({
            "指标": self.indicator_names,
            "信息熵 e_j": np.round(r["entropy"], 6),
            "差异系数 d_j": np.round(r["diff_coeff"], 6),
            "权重 w_j": np.round(r["weights"], 6),
            "权重(%)": np.round(r["weights"] * 100, 2),
        })
        r["weight_df"] = df

        print("=" * 56)
        print("  熵权法 — 计算结果")
        print("=" * 56)
        print(f"  评价对象数 m={r['m']}，指标数 n={r['n']}")
        print("-" * 56)
        print(df.to_string(index=False))
        print("=" * 56)
        return r

    def tex_description(self) -> str:
        self._check_computed()
        r = self._results
        rows = []
        for name, ej, dj, wj in zip(
                self.indicator_names,
                r["entropy"], r["diff_coeff"], r["weights"]
        ):
            rows.append(
                f"  {name} & {ej:.4f} & {dj:.4f} & {wj:.4f} \\\\"
            )
        table_body = "\n".join(rows)
        w_str = ", ".join(f"{w:.4f}" for w in r["weights"])

        tex = (
            r"\subsubsection{熵权法确定客观权重}" "\n\n"
            f"对 {r['m']} 个评价对象、{r['n']} 个指标的决策矩阵"
            "采用信息熵方法确定客观权重。\n\n"
            "对决策矩阵按列作比例归一化，计算各指标信息熵：\n"
            r"\begin{equation}"
            r"  e_j = -\frac{1}{\ln m} \sum_{i=1}^{m} p_{ij} \ln p_{ij},"
            r"\quad p_{ij} = \frac{x_{ij}}{\sum_{i=1}^{m} x_{ij}}"
            r"\end{equation}" "\n\n"
            "差异系数 $d_j = 1 - e_j$，权重 $w_j = d_j / \\sum_k d_k$。"
            "计算结果如下表所示：\n\n"
            r"\begin{table}[htbp]" "\n"
            r"\centering" "\n"
            r"\caption{熵权法权重计算结果}" "\n"
            r"\begin{tabular}{lccc}" "\n"
            r"\toprule" "\n"
            r"指标 & 信息熵 $e_j$ & 差异系数 $d_j$ & 权重 $w_j$ \\" "\n"
            r"\midrule" "\n"
            f"{table_body}\n"
            r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table}" "\n\n"
            f"最终权重向量 $\\mathbf{{w}} = ({w_str})^\\top$。\n"
        )
        return tex

    def plot(self, figsize: Tuple[int, int] = (13, 4)) -> plt.Figure:
        self._check_computed()
        r = self._results
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("熵权法 — 结果可视化", fontsize=13, fontweight="bold")

        names = self.indicator_names
        x = np.arange(len(names))
        palette = plt.cm.tab10.colors

        # ── 信息熵 ──
        ax = axes[0]
        ax.bar(x, r["entropy"], color=palette[0], alpha=0.8, edgecolor="white")
        ax.axhline(1.0, color="red", ls="--", lw=1, label="最大熵=1")
        ax.set_xticks(x);
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("信息熵 $e_j$");
        ax.set_title("信息熵")
        ax.legend(fontsize=8)

        # ── 差异系数 ──
        ax = axes[1]
        ax.bar(x, r["diff_coeff"], color=palette[1], alpha=0.8, edgecolor="white")
        ax.set_xticks(x);
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("差异系数 $d_j = 1 - e_j$");
        ax.set_title("差异系数")

        # ── 权重 ──
        ax = axes[2]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.85, len(names)))
        bars = ax.bar(x, r["weights"], color=colors, edgecolor="white")
        ax.axhline(1 / r["n"], color="grey", ls="--", lw=1, label="均等权重")
        for bar, val in zip(bars, r["weights"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x);
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("权重 $w_j$");
        ax.set_title("指标权重")
        ax.legend(fontsize=8)

        plt.tight_layout()
        return fig

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._results.get("weights")



# ═══════════════════════════════════════════════════════════════
#  2. CRITICMethod — CRITIC 法
# ═══════════════════════════════════════════════════════════════

class CRITICMethod(BaseMethod):
    """
    CRITIC 法 (Criteria Importance Through Intercriteria Correlation)

    同时考虑指标的**对比强度**（标准差）和**冲突性**（与其他指标
    的相关程度），综合度量每个指标所含信息量，作为客观权重依据。

    计算步骤
    --------
    设归一化决策矩阵 :math:`R = (r_{ij})_{m \\times n}`：

    1. 对比强度（标准差）：

       .. math:: \\sigma_j = \\sqrt{\\frac{1}{m-1}\\sum_{i=1}^{m}(r_{ij}-\\bar{r}_j)^2}

    2. 冲突性（1 - 线性相关系数之和）：

       .. math:: f_j = \\sum_{k=1}^{n}(1 - r_{jk}),
                 \\quad r_{jk} = \\text{Pearson}(j, k)

    3. 信息量与权重：

       .. math:: C_j = \\sigma_j \\cdot f_j, \\quad
                 w_j = C_j / \\sum_{l=1}^{n} C_l

    Parameters
    ----------
    indicator_names : list of str, optional
    correlation_method : {'pearson', 'spearman', 'kendall'}
        相关系数计算方法

    Examples
    --------
    >>> from src.algorithms.weights import CRITICMethod
    >>> import pandas as pd
    >>>
    >>> data = pd.DataFrame({
    ...     "经济": [0.6, 0.8, 0.4, 0.9, 0.7],
    ...     "社会": [0.5, 0.7, 0.6, 0.8, 0.6],
    ...     "环境": [0.9, 0.3, 0.8, 0.4, 0.6],
    ... })
    >>> critic = CRITICMethod()
    >>> critic.fit(data).compute().summary()
    """

    def __init__(
        self,
        indicator_names: Optional[List[str]] = None,
        correlation_method: str = "pearson",
    ) -> None:
        super().__init__(
            name="CRITIC法",
            description="综合对比强度与冲突性客观确定指标权重",
        )
        self.indicator_names    = indicator_names
        self.correlation_method = correlation_method
        self._X:  Optional[np.ndarray] = None
        self._m:  int = 0
        self._n:  int = 0

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        indicator_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "CRITICMethod":
        """
        载入决策矩阵（须已正向化处理）。

        Parameters
        ----------
        data : array-like, shape (m, n)
        """
        if isinstance(data, pd.DataFrame):
            if indicator_names is None:
                indicator_names = list(data.columns)
            data = data.values.astype(float)
        else:
            data = np.asarray(data, dtype=float)

        if data.ndim != 2:
            raise ValueError("data 必须是 2D 矩阵")
        if data.shape[0] < 3:
            warnings.warn(
                "评价对象数 m < 3，相关系数估计可靠性低，权重仅供参考。"
            )

        self._X  = data
        self._m, self._n = data.shape

        if indicator_names is not None:
            self.indicator_names = indicator_names
        if self.indicator_names is None:
            self.indicator_names = [f"C{j + 1}" for j in range(self._n)]
        if len(self.indicator_names) != self._n:
            raise ValueError(
                f"indicator_names 长度 ≠ n={self._n}"
            )

        self._fitted = True
        return self

    def compute(self) -> "CRITICMethod":
        """执行 CRITIC 法计算。"""
        self._check_fitted()
        m, n = self._m, self._n

        # ── 1. Min-Max 归一化 ──
        R = _minmax_normalize(self._X)          # shape (m, n)

        # ── 2. 对比强度（标准差）──
        sigma = R.std(axis=0, ddof=1)           # shape (n,)

        # ── 3. 相关矩阵 ──
        df_R = pd.DataFrame(R, columns=self.indicator_names)
        corr_matrix = df_R.corr(method=self.correlation_method).values  # (n, n)

        # 处理 NaN（常数列相关系数为 NaN）
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        # ── 4. 冲突性 f_j = Σ_k (1 - r_jk) ──
        conflict = np.sum(1.0 - corr_matrix, axis=1)   # shape (n,)

        # ── 5. 信息量 C_j = σ_j × f_j ──
        C = sigma * conflict
        C_sum = C.sum()
        if C_sum < 1e-12:
            warnings.warn("所有指标信息量接近 0，返回均等权重。")
            weights = np.ones(n) / n
        else:
            weights = C / C_sum

        self._results = {
            "weights":          weights,
            "sigma":            sigma,
            "conflict":         conflict,
            "information":      C,
            "corr_matrix":      corr_matrix,
            "R_normalized":     R,
            "indicator_names":  list(self.indicator_names),
            "m":                m,
            "n":                n,
            "correlation_method": self.correlation_method,
        }
        self._computed = True
        return self

    def summary(self) -> Dict[str, Any]:
        self._check_computed()
        r = self._results
        df = pd.DataFrame({
            "指标":          self.indicator_names,
            "标准差 σ_j":    np.round(r["sigma"],       4),
            "冲突性 f_j":    np.round(r["conflict"],    4),
            "信息量 C_j":    np.round(r["information"], 4),
            "权重 w_j":      np.round(r["weights"],     6),
            "权重(%)":       np.round(r["weights"] * 100, 2),
        })
        r["weight_df"] = df

        print("=" * 62)
        print("  CRITIC 法 — 计算结果")
        print("=" * 62)
        print(f"  评价对象数 m={r['m']}，指标数 n={r['n']}")
        print(f"  相关系数方法: {r['correlation_method']}")
        print("-" * 62)
        print(df.to_string(index=False))
        print("=" * 62)
        return r

    def tex_description(self) -> str:
        self._check_computed()
        r = self._results
        rows = []
        for name, sig, fj, cj, wj in zip(
            self.indicator_names,
            r["sigma"], r["conflict"], r["information"], r["weights"]
        ):
            rows.append(
                f"  {name} & {sig:.4f} & {fj:.4f} & {cj:.4f} & {wj:.4f} \\\\"
            )
        table_body = "\n".join(rows)
        w_str = ", ".join(f"{w:.4f}" for w in r["weights"])

        tex = (
            r"\subsubsection{CRITIC法确定客观权重}" "\n\n"
            "CRITIC法综合考虑指标的对比强度（标准差）与冲突性"
            "（指标间相关程度的负向度量），"
            f"使用{r['correlation_method'].capitalize()}相关系数。\n\n"
            r"\begin{equation}" "\n"
            r"  C_j = \sigma_j \cdot f_j, \quad"
            r"  f_j = \sum_{k=1}^{n}(1 - r_{jk}), \quad"
            r"  w_j = \frac{C_j}{\sum_{l=1}^{n} C_l}"
            "\n"
            r"\end{equation}" "\n\n"
            "计算结果如下表所示：\n\n"
            r"\begin{table}[htbp]" "\n"
            r"\centering" "\n"
            r"\caption{CRITIC法权重计算结果}" "\n"
            r"\begin{tabular}{lcccc}" "\n"
            r"\toprule" "\n"
            r"指标 & 标准差 $\sigma_j$ & 冲突性 $f_j$"
            r" & 信息量 $C_j$ & 权重 $w_j$ \\" "\n"
            r"\midrule" "\n"
            f"{table_body}\n"
            r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table}" "\n\n"
            f"最终权重向量 $\\mathbf{{w}} = ({w_str})^\\top$。\n"
        )
        return tex

    def plot(self, figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
        self._check_computed()
        r = self._results
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("CRITIC 法 — 结果可视化", fontsize=13, fontweight="bold")
        names = self.indicator_names
        x = np.arange(len(names))
        pal = plt.cm.Set2.colors

        # ── 子图1: 相关系数矩阵热力图 ──
        ax = axes[0]
        mask = np.eye(r["n"], dtype=bool)
        sns.heatmap(
            pd.DataFrame(r["corr_matrix"],
                         index=names, columns=names),
            annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, mask=mask,
            linewidths=0.4,
            cbar_kws={"label": "相关系数", "shrink": 0.8},
        )
        ax.set_title(f"{r['correlation_method'].capitalize()} 相关矩阵",
                     fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

        # ── 子图2: 对比强度 & 冲突性对比柱图 ──
        ax = axes[1]
        width = 0.38
        ax2 = ax.twinx()
        b1 = ax.bar(x - width / 2, r["sigma"],    width,
                    color=pal[0], alpha=0.85, label="标准差 σ_j")
        b2 = ax2.bar(x + width / 2, r["conflict"], width,
                     color=pal[1], alpha=0.85, label="冲突性 f_j")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("标准差", color=pal[0])
        ax2.set_ylabel("冲突性", color=pal[1])
        ax.set_title("对比强度 vs 冲突性", fontsize=11)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

        # ── 子图3: 信息量 & 最终权重 ──
        ax = axes[2]
        ax3 = ax.twinx()
        b3 = ax.bar(x - width / 2, r["information"], width,
                    color=pal[2], alpha=0.85, label="信息量 C_j")
        b4 = ax3.bar(x + width / 2, r["weights"],     width,
                     color=pal[3], alpha=0.85, label="权重 w_j")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("信息量", color=pal[2])
        ax3.set_ylabel("权重", color=pal[3])
        ax.set_title("信息量 vs 最终权重", fontsize=11)
        lines3, labels3 = ax.get_legend_handles_labels()
        lines4, labels4 = ax3.get_legend_handles_labels()
        ax.legend(lines3 + lines4, labels3 + labels4, fontsize=8, loc="upper right")

        plt.tight_layout()
        return fig

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._results.get("weights")


# ═══════════════════════════════════════════════════════════════
#  3. StdDeviationMethod — 标准离差法
# ═══════════════════════════════════════════════════════════════

class StdDeviationMethod(BaseMethod):
    """
    标准离差法 (Standard Deviation Method)

    以各指标的标准差作为权重依据：标准差越大，说明各评价对象
    在该指标上差异越大，该指标对评价结果的影响越重要。

    计算步骤
    --------
    设归一化决策矩阵列标准差为 :math:`\\sigma_j`：

    .. math::
        w_j = \\sigma_j / \\sum_{k=1}^{n} \\sigma_k

    支持多种归一化策略：

    * ``minmax``  — Min-Max 归一到 [0, 1]（推荐）
    * ``zscore``  — Z-score 标准化后取绝对标准差
    * ``raw``     — 直接使用原始数据标准差

    Parameters
    ----------
    indicator_names : list of str, optional
    normalization : {'minmax', 'zscore', 'raw'}
    ddof : int, default 1
        标准差自由度（1=样本标准差，0=总体标准差）

    Examples
    --------
    >>> from src.algorithms.weights import StdDeviationMethod
    >>> import numpy as np
    >>>
    >>> X = np.array([[3,1,2],[2,3,1],[1,2,3],[3,2,1],[2,1,3]])
    >>> sd = StdDeviationMethod(["A","B","C"])
    >>> sd.fit(X).compute().summary()
    """

    def __init__(
        self,
        indicator_names: Optional[List[str]] = None,
        normalization: str = "minmax",
        ddof: int = 1,
    ) -> None:
        super().__init__(
            name="标准离差法",
            description="以指标标准差度量区分能力，客观确定权重",
        )
        self.indicator_names = indicator_names
        self.normalization   = normalization
        self.ddof            = ddof
        self._X:  Optional[np.ndarray] = None
        self._m:  int = 0
        self._n:  int = 0

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        indicator_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "StdDeviationMethod":
        if isinstance(data, pd.DataFrame):
            if indicator_names is None:
                indicator_names = list(data.columns)
            data = data.values.astype(float)
        else:
            data = np.asarray(data, dtype=float)

        if data.ndim != 2:
            raise ValueError("data 必须是 2D 矩阵")

        self._X  = data
        self._m, self._n = data.shape

        if indicator_names is not None:
            self.indicator_names = indicator_names
        if self.indicator_names is None:
            self.indicator_names = [f"C{j + 1}" for j in range(self._n)]

        self._fitted = True
        return self

    def compute(self) -> "StdDeviationMethod":
        """计算标准离差法权重。"""
        self._check_fitted()
        n = self._n

        # ── 归一化 ──
        norm_map = {
            "minmax": lambda X: _minmax_normalize(X),
            "zscore": lambda X: (
                X - X.mean(axis=0)
            ) / (X.std(axis=0, ddof=1) + 1e-12),
            "raw":    lambda X: X,
        }
        if self.normalization not in norm_map:
            raise ValueError(
                f"不支持的归一化方式: '{self.normalization}'。"
                f"可选: {list(norm_map)}"
            )
        X_norm = norm_map[self.normalization](self._X)

        # ── 各列标准差 ──
        sigma = X_norm.std(axis=0, ddof=self.ddof)

        sigma_sum = sigma.sum()
        if sigma_sum < 1e-12:
            warnings.warn("所有标准差接近 0，返回均等权重。")
            weights = np.ones(n) / n
        else:
            weights = sigma / sigma_sum

        self._results = {
            "weights":         weights,
            "sigma":           sigma,
            "X_normalized":    X_norm,
            "indicator_names": list(self.indicator_names),
            "m":               self._m,
            "n":               n,
            "normalization":   self.normalization,
        }
        self._computed = True
        return self

    def summary(self) -> Dict[str, Any]:
        self._check_computed()
        r = self._results
        df = pd.DataFrame({
            "指标":      self.indicator_names,
            "标准差 σ_j": np.round(r["sigma"],   4),
            "权重 w_j":   np.round(r["weights"], 6),
            "权重(%)":    np.round(r["weights"] * 100, 2),
        })
        r["weight_df"] = df

        print("=" * 50)
        print("  标准离差法 — 计算结果")
        print("=" * 50)
        print(f"  归一化方式: {r['normalization']}")
        print("-" * 50)
        print(df.to_string(index=False))
        print("=" * 50)
        return r

    def tex_description(self) -> str:
        self._check_computed()
        r = self._results
        w_str = ", ".join(f"{w:.4f}" for w in r["weights"])
        tex = (
            r"\subsubsection{标准离差法确定客观权重}" "\n\n"
            "标准离差法以各指标的标准差衡量指标对区分评价对象的贡献，"
            f"采用 {r['normalization'].upper()} 归一化后计算各列标准差：\n\n"
            r"\begin{equation}" "\n"
            r"  w_j = \frac{\sigma_j}{\sum_{k=1}^{n} \sigma_k}"
            "\n"
            r"\end{equation}" "\n\n"
            f"最终权重向量 $\\mathbf{{w}} = ({w_str})^\\top$。\n"
        )
        return tex

    def plot(self, figsize: Tuple[int, int] = (10, 4)) -> plt.Figure:
        self._check_computed()
        r = self._results
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle("标准离差法 — 结果可视化",
                     fontsize=12, fontweight="bold")
        names = self.indicator_names
        x = np.arange(len(names))

        # ── 左图: 标准差 ──
        ax = axes[0]
        ax.bar(x, r["sigma"], color="#3498db", alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("标准差 σ_j")
        ax.set_title("各指标标准差（归一化后）")

        # ── 右图: 权重 ──
        ax = axes[1]
        colors = plt.cm.YlOrRd(np.linspace(0.35, 0.85, len(names)))
        bars = ax.bar(x, r["weights"], color=colors, edgecolor="white")
        ax.axhline(1 / r["n"], color="grey", ls="--", lw=1, label="均等权重")
        for bar, val in zip(bars, r["weights"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("权重 w_j")
        ax.set_title("指标权重")
        ax.legend(fontsize=8)

        plt.tight_layout()
        return fig

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._results.get("weights")


# ═══════════════════════════════════════════════════════════════
#  4. PCAWeightMethod — 主成分分析法确定权重
# ═══════════════════════════════════════════════════════════════

class PCAWeightMethod(BaseMethod):
    """
    主成分分析法确定权重 (PCA-based Weight Method)

    利用 PCA 提取主成分，将各主成分**方差贡献率**作为该主成分
    的权重，再结合各指标在各主成分上的**绝对载荷系数**，
    合成每个指标的综合权重。

    计算步骤
    --------
    1. 对数据 Z-score 标准化；
    2. 计算协方差矩阵的特征值 :math:`\\lambda_k` 和特征向量；
    3. 方差贡献率 :math:`\\theta_k = \\lambda_k / \\sum_j \\lambda_j`；
    4. 各指标综合得分系数：

       .. math::
           v_j = \\sum_{k=1}^{K} \\theta_k \\cdot |a_{jk}|

       其中 :math:`a_{jk}` 为指标 :math:`j` 在第 :math:`k` 个
       主成分上的载荷系数，:math:`K` 为保留主成分数；

    5. 归一化 :math:`w_j = v_j / \\sum_l v_l`。

    Parameters
    ----------
    indicator_names : list of str, optional
    n_components : int or float or None
        保留主成分数。
        * int  → 精确数量；
        * float (0, 1) → 累计方差解释率阈值（默认 0.85）；
        * None → 全部保留；
    use_correlation : bool, default True
        True: 使用相关系数矩阵（等价于先 Z-score 标准化）；
        False: 使用协方差矩阵。

    Examples
    --------
    >>> from src.algorithms.weights import PCAWeightMethod
    >>> import numpy as np
    >>>
    >>> np.random.seed(42)
    >>> data = np.random.randn(20, 5)
    >>> pca_w = PCAWeightMethod(n_components=0.85)
    >>> pca_w.fit(data).compute().summary()
    """

    def __init__(
        self,
        indicator_names: Optional[List[str]] = None,
        n_components: Union[int, float, None] = 0.85,
        use_correlation: bool = True,
    ) -> None:
        super().__init__(
            name="PCA权重法",
            description="基于主成分分析方差贡献率确定指标权重",
        )
        self.indicator_names = indicator_names
        self.n_components    = n_components
        self.use_correlation = use_correlation
        self._X:  Optional[np.ndarray] = None
        self._m:  int = 0
        self._n:  int = 0

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        indicator_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "PCAWeightMethod":
        if isinstance(data, pd.DataFrame):
            if indicator_names is None:
                indicator_names = list(data.columns)
            data = data.values.astype(float)
        else:
            data = np.asarray(data, dtype=float)

        if data.ndim != 2:
            raise ValueError("data 必须是 2D 矩阵")
        if data.shape[0] < data.shape[1]:
            warnings.warn(
                f"样本数 m={data.shape[0]} < 指标数 n={data.shape[1]}，"
                "PCA 结果可能不稳定。"
            )

        self._X  = data
        self._m, self._n = data.shape

        if indicator_names is not None:
            self.indicator_names = indicator_names
        if self.indicator_names is None:
            self.indicator_names = [f"C{j + 1}" for j in range(self._n)]

        self._fitted = True
        return self

    def compute(self) -> "PCAWeightMethod":
        """执行 PCA 权重计算。"""
        self._check_fitted()
        m, n = self._m, self._n

        # ── 1. 标准化 ──
        if self.use_correlation:
            scaler = StandardScaler()
            X_std  = scaler.fit_transform(self._X)
        else:
            X_std  = self._X.copy()

        # ── 2. PCA 拟合（全量保留，后续再截断）──
        pca_full = PCA(n_components=None, svd_solver="full")
        pca_full.fit(X_std)
        explained_var_ratio  = pca_full.explained_variance_ratio_   # (n,)
        components           = pca_full.components_                  # (n, n)
        explained_var        = pca_full.explained_variance_          # (n,)

        # ── 3. 确定保留主成分数 K ──
        if self.n_components is None:
            K = n
        elif isinstance(self.n_components, float):
            cumsum = np.cumsum(explained_var_ratio)
            K = int(np.searchsorted(cumsum, self.n_components) + 1)
            K = min(K, n)
        else:
            K = min(int(self.n_components), n)
        K = max(K, 1)

        # ── 4. 综合权重合成 ──
        # 保留 K 个主成分的方差贡献率
        theta = explained_var_ratio[:K]
        theta = theta / theta.sum()           # 重新归一至保留部分
        # 载荷矩阵 (K, n) → 转置为 (n, K)
        loadings = components[:K, :].T        # shape (n, K)
        # 各指标综合分值
        v = np.abs(loadings) @ theta          # shape (n,)
        v_sum = v.sum()
        weights = v / v_sum if v_sum > 1e-12 else np.ones(n) / n

        # ── 5. 完整 PCA 变换结果（供可视化）──
        pca_K = PCA(n_components=K)
        X_transformed = pca_K.fit_transform(X_std)

        self._results = {
            "weights":            weights,
            "v_scores":           v,
            "theta":              theta,
            "loadings":           loadings,         # (n, K)
            "components_full":    components,        # (n_full, n)
            "explained_var_ratio": explained_var_ratio,
            "explained_var":      explained_var,
            "K":                  K,
            "X_transformed":      X_transformed,
            "indicator_names":    list(self.indicator_names),
            "pc_names":           [f"PC{k+1}" for k in range(K)],
            "m":                  m,
            "n":                  n,
        }
        self._computed = True
        return self

    def summary(self) -> Dict[str, Any]:
        self._check_computed()
        r = self._results
        K = r["K"]

        # ── 方差贡献率表 ──
        var_df = pd.DataFrame({
            "主成分":     [f"PC{k+1}" for k in range(len(r["explained_var_ratio"]))],
            "方差贡献率":  np.round(r["explained_var_ratio"] * 100, 2),
            "累计贡献率":  np.round(np.cumsum(r["explained_var_ratio"]) * 100, 2),
        })
        var_df["保留"] = var_df.index.map(lambda i: "★" if i < K else "")

        # ── 载荷矩阵表 ──
        load_df = pd.DataFrame(
            np.round(r["loadings"], 4),
            index   = self.indicator_names,
            columns = [f"PC{k+1}" for k in range(K)],
        )
        load_df["综合分值 v_j"] = np.round(r["v_scores"], 4)
        load_df["权重 w_j"]    = np.round(r["weights"],   6)

        r["var_df"]  = var_df
        r["load_df"] = load_df

        print("=" * 60)
        print("  PCA 权重法 — 计算结果")
        print("=" * 60)
        print(f"  保留主成分数 K={K}，累计方差贡献率 = "
              f"{r['explained_var_ratio'][:K].sum()*100:.2f}%")
        print("\n  方差贡献率：")
        print(var_df.to_string(index=False))
        print("\n  载荷矩阵与综合权重：")
        print(load_df.to_string())
        print("=" * 60)
        return r

    def tex_description(self) -> str:
        self._check_computed()
        r = self._results
        cum_var = r["explained_var_ratio"][:r["K"]].sum() * 100
        w_str   = ", ".join(f"{w:.4f}" for w in r["weights"])
        rows = []
        for j, (name, vj, wj) in enumerate(zip(
            self.indicator_names, r["v_scores"], r["weights"]
        )):
            rows.append(f"  {name} & {vj:.4f} & {wj:.4f} \\\\")
        table_body = "\n".join(rows)

        tex = (
            r"\subsubsection{主成分分析法（PCA）确定客观权重}" "\n\n"
            "对标准化后的决策矩阵进行主成分分析，"
            f"保留前 {r['K']} 个主成分（累计方差贡献率"
            f" ${cum_var:.2f}\\%$）。\n\n"
            "以各主成分方差贡献率 $\\theta_k$ 为权重，"
            "结合指标在各主成分上的绝对载荷系数合成综合分值：\n"
            r"\begin{equation}" "\n"
            r"  v_j = \sum_{k=1}^{K} \theta_k \cdot |a_{jk}|,"
            r"\quad w_j = \frac{v_j}{\sum_{l=1}^{n} v_l}"
            "\n"
            r"\end{equation}" "\n\n"
            r"\begin{table}[htbp]" "\n"
            r"\centering\caption{PCA综合权重计算结果}" "\n"
            r"\begin{tabular}{lcc}" "\n"
            r"\toprule" "\n"
            r"指标 & 综合分值 $v_j$ & 权重 $w_j$ \\" "\n"
            r"\midrule" "\n"
            f"{table_body}\n"
            r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table}" "\n\n"
            f"最终权重向量 $\\mathbf{{w}} = ({w_str})^\\top$。\n"
        )
        return tex

    def plot(self, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        self._check_computed()
        r = self._results
        K    = r["K"]
        n    = r["n"]
        names = self.indicator_names

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("PCA 权重法 — 结果可视化",
                     fontsize=13, fontweight="bold")

        # ── 子图1: 碎石图（Scree Plot）──
        ax = axes[0]
        all_K = len(r["explained_var_ratio"])
        k_range = np.arange(1, all_K + 1)
        ax.bar(k_range, r["explained_var_ratio"] * 100,
               color="#3498db", alpha=0.7, label="单个方差贡献率")
        ax.plot(k_range, np.cumsum(r["explained_var_ratio"]) * 100,
                "r-o", markersize=5, lw=1.8, label="累计方差贡献率")
        ax.axvline(K + 0.5, color="orange", ls="--", lw=1.5,
                   label=f"保留前 {K} 个")
        ax.axhline(85, color="green", ls=":", lw=1.2, alpha=0.7,
                   label="85% 阈值")
        ax.set_xlabel("主成分序号")
        ax.set_ylabel("方差贡献率 (%)")
        ax.set_title("碎石图（Scree Plot）")
        ax.legend(fontsize=8)
        ax.set_xticks(k_range)
        ax.set_ylim(0, 115)

        # ── 子图2: 载荷热力图 ──
        ax = axes[1]
        load_matrix = r["loadings"]    # (n, K)
        sns.heatmap(
            pd.DataFrame(load_matrix,
                         index=names,
                         columns=[f"PC{k+1}" for k in range(K)]),
            annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, linewidths=0.3,
            cbar_kws={"label": "载荷系数", "shrink": 0.85},
        )
        ax.set_title(f"因子载荷矩阵（前{K}个主成分）")
        ax.tick_params(axis="y", rotation=0)

        # ── 子图3: 综合权重 ──
        ax = axes[2]
        x = np.arange(n)
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, n))
        bars = ax.bar(x, r["weights"], color=colors, edgecolor="white")
        ax.axhline(1 / n, color="grey", ls="--", lw=1, label="均等权重")
        for bar, val in zip(bars, r["weights"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("综合权重")
        ax.set_title("PCA 合成权重")
        ax.legend(fontsize=8)

        plt.tight_layout()
        return fig

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._results.get("weights")