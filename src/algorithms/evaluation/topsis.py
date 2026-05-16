"""
topsis.py
TOPSIS — Technique for Order Preference by Similarity to Ideal Solution
逼近理想解排序法

Reference:
    Hwang, C.L. & Yoon, K. (1981). Multiple Attribute Decision Making.
    Lecture Notes in Economics and Mathematical Systems, 186. Springer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..base import BaseMethod

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 全局绘图风格
# ──────────────────────────────────────────────────────────────────────────────
_STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
}


class TOPSIS(BaseMethod):
    """
    TOPSIS 逼近理想解排序法。

    Parameters
    ----------
    weights : array-like, optional
        指标权重向量（未归一化亦可，内部自动归一）。默认等权。
    indicator_types : list of str, optional
        每个指标的方向：``'positive'``（效益型）或 ``'negative'``（成本型）。
        默认全部为 ``'positive'``。
    normalize : bool, default True
        是否对输入矩阵做向量归一化。若上游已归一化可设 ``False``。

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame(
    ...     {"GDP增长": [8.1, 7.5, 9.2, 6.8],
    ...      "失业率":   [4.2, 3.8, 5.1, 4.0],
    ...      "创新指数": [72, 68, 85, 61]},
    ...     index=["城市A", "城市B", "城市C", "城市D"]
    ... )
    >>> model = TOPSIS(indicator_types=["positive", "negative", "positive"])
    >>> model.fit(data, weights=[0.3, 0.3, 0.4])
    >>> result = model.compute()
    >>> print(result)
    """

    def __init__(
        self,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        indicator_types: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__(name="TOPSIS")
        self.weights: Optional[np.ndarray] = (
            np.asarray(weights, dtype=float) if weights is not None else None
        )
        self.indicator_types = indicator_types
        self.normalize = normalize

        # 内部状态（fit/compute 后填充）
        self._data: Optional[pd.DataFrame] = None
        self._alternatives: Optional[List] = None
        self._criteria: Optional[List] = None
        self._norm_matrix: Optional[np.ndarray] = None
        self._weighted_matrix: Optional[np.ndarray] = None
        self._a_pos: Optional[np.ndarray] = None  # 正理想解
        self._a_neg: Optional[np.ndarray] = None  # 负理想解
        self._d_pos: Optional[np.ndarray] = None
        self._d_neg: Optional[np.ndarray] = None
        self._scores: Optional[np.ndarray] = None

    # ── 公共 API ──────────────────────────────────────────────────────────────

    def fit(
        self,
        data: pd.DataFrame,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        indicator_types: Optional[List[str]] = None,
    ) -> "TOPSIS":
        """绑定决策矩阵与参数。"""
        if data.isnull().any().any():
            logger.warning("决策矩阵含缺失值，已用列均值填充。")
            data = data.fillna(data.mean(numeric_only=True))

        self._data = data.copy()
        self._alternatives = list(data.index)
        self._criteria = list(data.columns)
        n, m = data.shape

        # 权重
        if weights is not None:
            self.weights = np.asarray(weights, dtype=float)
        if self.weights is None:
            self.weights = np.ones(m, dtype=float)
        w_sum = self.weights.sum()
        if w_sum <= 0:
            raise ValueError("权重之和必须为正数。")
        self.weights = self.weights / w_sum
        if len(self.weights) != m:
            raise ValueError(f"权重维度 {len(self.weights)} ≠ 指标数 {m}。")

        # 指标方向
        if indicator_types is not None:
            self.indicator_types = indicator_types
        if self.indicator_types is None:
            self.indicator_types = ["positive"] * m
        if len(self.indicator_types) != m:
            raise ValueError(f"指标类型维度 {len(self.indicator_types)} ≠ 指标数 {m}。")
        invalid = set(self.indicator_types) - {"positive", "negative"}
        if invalid:
            raise ValueError(f"无效指标类型 {invalid}，只接受 'positive'/'negative'。")

        self.metadata.update(
            {
                "n_alternatives": n,
                "n_criteria": m,
                "alternatives": self._alternatives,
                "criteria": self._criteria,
            }
        )
        logger.info("TOPSIS.fit: %d 个对象, %d 个指标。", n, m)
        return self

    def compute(self) -> pd.DataFrame:
        """
        执行 TOPSIS 全流程计算。

        Returns
        -------
        pd.DataFrame
            列：``Score (Ci)`` / ``D+`` / ``D-`` / ``Rank``，按排名升序排列。
        """
        if self._data is None:
            raise RuntimeError("请先调用 fit()。")

        X = self._data.values.astype(float)
        n, m = X.shape

        # ① 向量归一化
        if self.normalize:
            norms = np.linalg.norm(X, axis=0)
            norms[norms < 1e-12] = 1.0
            self._norm_matrix = X / norms
        else:
            self._norm_matrix = X.copy()

        # ② 加权归一化矩阵
        self._weighted_matrix = self._norm_matrix * self.weights

        # ③ 正负理想解
        self._a_pos = np.where(
            [t == "positive" for t in self.indicator_types],
            self._weighted_matrix.max(axis=0),
            self._weighted_matrix.min(axis=0),
        )
        self._a_neg = np.where(
            [t == "positive" for t in self.indicator_types],
            self._weighted_matrix.min(axis=0),
            self._weighted_matrix.max(axis=0),
        )

        # ④ 欧氏距离
        self._d_pos = np.sqrt(((self._weighted_matrix - self._a_pos) ** 2).sum(axis=1))
        self._d_neg = np.sqrt(((self._weighted_matrix - self._a_neg) ** 2).sum(axis=1))

        # ⑤ 相对贴近度
        denom = self._d_pos + self._d_neg
        denom[denom < 1e-12] = 1e-12
        self._scores = self._d_neg / denom

        # ⑥ 排名
        ranks = (
            pd.Series(self._scores, index=self._alternatives)
            .rank(ascending=False, method="min")
            .astype(int)
        )

        self.result = pd.DataFrame(
            {
                "Score (Ci)": np.round(self._scores, 6),
                "D+": np.round(self._d_pos, 6),
                "D-": np.round(self._d_neg, 6),
                "Rank": ranks,
            },
            index=self._alternatives,
        ).sort_values("Rank")

        self.metadata.update(
            {
                "scores": self._scores.tolist(),
                "best": self.result.index[0],
                "worst": self.result.index[-1],
                "ideal_positive": dict(zip(self._criteria, self._a_pos.round(6).tolist())),
                "ideal_negative": dict(zip(self._criteria, self._a_neg.round(6).tolist())),
            }
        )
        logger.info(
            "TOPSIS 计算完成 → 最优: %s (Ci=%.4f)",
            self.result.index[0],
            self._scores.max(),
        )
        return self.result

    def summary(self) -> Dict[str, Any]:
        """返回结果摘要字典（便于 LaTeX 生成器使用）。"""
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")
        return {
            "method": "TOPSIS",
            "description": "逼近理想解排序法",
            "n_alternatives": len(self._alternatives),
            "n_criteria": len(self._criteria),
            "criteria": self._criteria,
            "weights": {c: float(w) for c, w in zip(self._criteria, self.weights)},
            "indicator_types": dict(zip(self._criteria, self.indicator_types)),
            "ideal_positive": self.metadata["ideal_positive"],
            "ideal_negative": self.metadata["ideal_negative"],
            "scores": self.result["Score (Ci)"].to_dict(),
            "distances_pos": self.result["D+"].to_dict(),
            "distances_neg": self.result["D-"].to_dict(),
            "ranking": self.result["Rank"].to_dict(),
            "best": self.result.index[0],
            "worst": self.result.index[-1],
        }

    # ── 可视化 ────────────────────────────────────────────────────────────────

    def plot_scores(
        self,
        title: str = "TOPSIS 综合评价得分",
        figsize: Tuple = (10, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制综合得分水平条形图（颜色深浅映射得分高低）。"""
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=figsize)
            sorted_df = self.result.sort_values("Score (Ci)", ascending=True)
            scores = sorted_df["Score (Ci)"].values
            labels = sorted_df.index.tolist()

            cmap = plt.get_cmap("RdYlGn")
            norm = plt.Normalize(vmin=scores.min(), vmax=scores.max())
            colors = [cmap(norm(s)) for s in scores]

            bars = ax.barh(
                range(len(labels)), scores, color=colors,
                edgecolor="grey", linewidth=0.6, height=0.65,
            )
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=11)
            ax.set_xlabel(r"相对贴近度 $C_i$", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
            ax.set_xlim(0, min(1.05, scores.max() * 1.22))
            ax.axvline(0.5, color="navy", ls="--", alpha=0.35, lw=1.2,
                       label="基准线 (0.5)")
            ax.legend(fontsize=9)

            for bar, s in zip(bars, scores):
                ax.text(
                    s + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{s:.4f}", va="center", fontsize=9,
                )

            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info("图表已保存: %s", save_path)
        return fig

    def plot_distance_scatter(
        self,
        figsize: Tuple = (8, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制 D+ vs D- 散点图，颜色映射 Ci 得分。"""
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=figsize)
            sc = ax.scatter(
                self._d_pos, self._d_neg,
                c=self._scores, cmap="RdYlGn", s=150,
                edgecolors="black", linewidth=0.8, zorder=5,
            )
            for i, alt in enumerate(self._alternatives):
                ax.annotate(
                    alt, (self._d_pos[i], self._d_neg[i]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9,
                )
            plt.colorbar(sc, ax=ax, label=r"$C_i$ 得分")
            ax.set_xlabel(r"$D^+$（到正理想解距离）", fontsize=12)
            ax.set_ylabel(r"$D^-$（到负理想解距离）", fontsize=12)
            ax.set_title("TOPSIS 理想解距离分布", fontsize=13, fontweight="bold")
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_radar(
        self,
        top_n: int = 5,
        figsize: Tuple = (8, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制前 top_n 个对象的指标雷达图（基于归一化矩阵）。"""
        if self._norm_matrix is None:
            raise RuntimeError("请先调用 compute()。")

        top_alts = self.result.head(top_n).index.tolist()
        idx = [self._alternatives.index(a) for a in top_alts]
        values = self._norm_matrix[idx]

        m = len(self._criteria)
        angles = np.linspace(0, 2 * np.pi, m, endpoint=False).tolist()
        angles += angles[:1]

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
            colors = plt.cm.tab10(np.linspace(0, 1, len(top_alts)))
            for i, (alt, row) in enumerate(zip(top_alts, values)):
                vals = row.tolist() + row[:1].tolist()
                ax.plot(angles, vals, "o-", lw=2, color=colors[i], label=alt)
                ax.fill(angles, vals, alpha=0.08, color=colors[i])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(self._criteria, fontsize=10)
            ax.set_title(f"前 {top_n} 名指标雷达图", fontsize=13,
                         fontweight="bold", pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ── LaTeX 描述 ────────────────────────────────────────────────────────────

    def tex_description(self) -> str:
        """生成 TOPSIS 方法段落的 LaTeX 源码。"""
        return r"""
\subsection{基于TOPSIS的综合评价模型}

\subsubsection{模型原理}

TOPSIS（Technique for Order Preference by Similarity to Ideal Solution）
是由Hwang和Yoon（1981）提出的经典多属性决策方法。
其核心思想为：最优方案应同时最接近正理想解$A^+$（所有指标均取最优值）
且最远离负理想解$A^-$（所有指标均取最劣值）。

\subsubsection{计算步骤}

\noindent\textbf{步骤1\quad 向量归一化}

设决策矩阵 $\boldsymbol{X}=(x_{ij})_{n\times m}$，向量归一化后得：
\begin{equation}
    r_{ij}=\frac{x_{ij}}{\sqrt{\displaystyle\sum_{i=1}^{n}x_{ij}^{2}}},
    \quad i=1,\ldots,n;\;j=1,\ldots,m
    \label{eq:topsis-norm}
\end{equation}

\noindent\textbf{步骤2\quad 构造加权归一化矩阵}

\begin{equation}
    v_{ij}=w_j\cdot r_{ij},\quad \sum_{j=1}^{m}w_j=1
    \label{eq:topsis-weighted}
\end{equation}

\noindent\textbf{步骤3\quad 确定正负理想解}

\begin{align}
    A^{+}&=\left(v_1^{+},\,v_2^{+},\,\ldots,\,v_m^{+}\right) \\
    A^{-}&=\left(v_1^{-},\,v_2^{-},\,\ldots,\,v_m^{-}\right)
\end{align}
其中效益型指标取 $v_j^{+}=\max_i v_{ij}$，成本型指标取 $v_j^{+}=\min_i v_{ij}$。

\noindent\textbf{步骤4\quad 计算欧氏距离}

\begin{equation}
    D_i^{+}=\sqrt{\sum_{j=1}^{m}\left(v_{ij}-v_j^{+}\right)^{2}},\qquad
    D_i^{-}=\sqrt{\sum_{j=1}^{m}\left(v_{ij}-v_j^{-}\right)^{2}}
    \label{eq:topsis-dist}
\end{equation}

\noindent\textbf{步骤5\quad 计算相对贴近度并排序}

\begin{equation}
    C_i=\frac{D_i^{-}}{D_i^{+}+D_i^{-}},\quad 0\leq C_i\leq 1
    \label{eq:topsis-score}
\end{equation}

$C_i$越大表明第$i$个评价对象越接近正理想解，
综合评价水平越高，据此降序排列得到最终综合排名。
"""