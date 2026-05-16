"""
electre.py
ELECTRE I — Elimination Et Choix Traduisant la Realité
淘汰与选择法（一致性/非一致性矩阵）

Reference:
    Roy, B. (1968). Classement et choix en présence de points de vue multiples.
    Revue française d'informatique et de recherche opérationnelle, 2(8), 57-75.
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

_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
}


class ELECTRE(BaseMethod):
    """
    ELECTRE I 淘汰排序法。

    通过计算一致性矩阵（Concordance）和非一致性矩阵（Discordance），
    基于设定的阈值建立方案间的超越关系，进而确定综合排名。

    Parameters
    ----------
    concordance_threshold : float, default 0.7
        一致性阈值 $\\bar{c}$，超越关系成立的一致性最低要求。
    discordance_threshold : float, default 0.3
        非一致性阈值 $\\bar{d}$，超越关系成立的非一致性上限。
    weights : array-like, optional
        指标权重，默认等权。
    indicator_types : list of str, optional
        'positive' 或 'negative'，默认全正向。
    """

    def __init__(
        self,
        concordance_threshold: float = 0.7,
        discordance_threshold: float = 0.3,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        indicator_types: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name="ELECTRE")
        self.concordance_threshold = concordance_threshold
        self.discordance_threshold = discordance_threshold
        self.weights: Optional[np.ndarray] = (
            np.asarray(weights, dtype=float) if weights is not None else None
        )
        self.indicator_types = indicator_types

        self._data: Optional[pd.DataFrame] = None
        self._alternatives: Optional[List] = None
        self._criteria: Optional[List] = None
        self._concordance_matrix: Optional[np.ndarray] = None
        self._discordance_matrix: Optional[np.ndarray] = None
        self._outranking_matrix: Optional[np.ndarray] = None  # 0/1 超越矩阵
        self._net_dominance: Optional[np.ndarray] = None

    def fit(
        self,
        data: pd.DataFrame,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        indicator_types: Optional[List[str]] = None,
    ) -> "ELECTRE":
        if data.isnull().any().any():
            data = data.fillna(data.mean(numeric_only=True))
            logger.warning("缺失值已用列均值填充。")

        self._data = data.copy()
        self._alternatives = list(data.index)
        self._criteria = list(data.columns)
        n, m = data.shape

        if weights is not None:
            self.weights = np.asarray(weights, dtype=float)
        if self.weights is None:
            self.weights = np.ones(m, dtype=float)
        self.weights = self.weights / self.weights.sum()

        if indicator_types is not None:
            self.indicator_types = indicator_types
        if self.indicator_types is None:
            self.indicator_types = ["positive"] * m

        if len(self.weights) != m or len(self.indicator_types) != m:
            raise ValueError("权重或指标类型维度与指标数不匹配。")
        return self

    def compute(self) -> pd.DataFrame:
        """
        执行 ELECTRE I 计算。

        Returns
        -------
        pd.DataFrame
            列：``Dominance (φ+)`` / ``Dominated (φ-)`` / ``Net Flow (φ)`` / ``Rank``。
        """
        if self._data is None:
            raise RuntimeError("请先调用 fit()。")

        X = self._data.values.astype(float)
        n, m = X.shape

        # ── 归一化（min-max）处理成本型指标 ──────────────────────────────
        X_norm = X.copy()
        for j, t in enumerate(self.indicator_types):
            x_min, x_max = X[:, j].min(), X[:, j].max()
            rng = x_max - x_min
            if rng < 1e-12:
                X_norm[:, j] = 0.0
            elif t == "negative":
                X_norm[:, j] = (x_max - X[:, j]) / rng
            else:
                X_norm[:, j] = (X[:, j] - x_min) / rng

        # ── 一致性矩阵 C[i,k] = Σ w_j for j where x_ij >= x_kj ─────────
        self._concordance_matrix = np.zeros((n, n))
        for i in range(n):
            for k in range(n):
                if i == k:
                    continue
                better = X_norm[i] >= X_norm[k]
                self._concordance_matrix[i, k] = self.weights[better].sum()

        # ── 非一致性矩阵 D[i,k] = max_j(|x_kj - x_ij| / range_j) ────────
        #   其中求最大的 j 满足 x_kj > x_ij（方案 k 在 j 上优于 i）
        col_ranges = X_norm.max(axis=0) - X_norm.min(axis=0)
        col_ranges[col_ranges < 1e-12] = 1.0
        self._discordance_matrix = np.zeros((n, n))
        for i in range(n):
            for k in range(n):
                if i == k:
                    continue
                diff = X_norm[k] - X_norm[i]  # positive where k > i
                dominated_cols = diff > 0
                if not dominated_cols.any():
                    self._discordance_matrix[i, k] = 0.0
                else:
                    self._discordance_matrix[i, k] = (diff[dominated_cols]).max()

        # ── 超越矩阵（同时满足两个阈值条件）────────────────────────────
        c_cond = self._concordance_matrix >= self.concordance_threshold
        d_cond = self._discordance_matrix <= self.discordance_threshold
        self._outranking_matrix = (c_cond & d_cond).astype(int)
        np.fill_diagonal(self._outranking_matrix, 0)

        # ── 净支配值（类似 PROMETHEE 的净流）───────────────────────────
        phi_plus = self._outranking_matrix.sum(axis=1).astype(float)   # i 超越几个
        phi_minus = self._outranking_matrix.sum(axis=0).astype(float)  # 几个超越 i
        self._net_dominance = phi_plus - phi_minus

        ranks = (
            pd.Series(self._net_dominance, index=self._alternatives)
            .rank(ascending=False, method="min")
            .astype(int)
        )

        self.result = pd.DataFrame(
            {
                "φ+ (Dominance)": phi_plus.astype(int),
                "φ- (Dominated)": phi_minus.astype(int),
                "Net Flow φ": self._net_dominance,
                "Rank": ranks,
            },
            index=self._alternatives,
        ).sort_values("Rank")

        self.metadata.update(
            {
                "concordance_threshold": self.concordance_threshold,
                "discordance_threshold": self.discordance_threshold,
                "best": self.result.index[0],
            }
        )
        logger.info(
            "ELECTRE 计算完成 → 最优: %s (φ=%.1f)",
            self.result.index[0], self._net_dominance.max(),
        )
        return self.result

    def get_concordance_matrix(self) -> pd.DataFrame:
        """返回一致性矩阵 DataFrame。"""
        if self._concordance_matrix is None:
            raise RuntimeError("请先调用 compute()。")
        return pd.DataFrame(
            self._concordance_matrix,
            index=self._alternatives, columns=self._alternatives,
        ).round(4)

    def get_discordance_matrix(self) -> pd.DataFrame:
        """返回非一致性矩阵 DataFrame。"""
        if self._discordance_matrix is None:
            raise RuntimeError("请先调用 compute()。")
        return pd.DataFrame(
            self._discordance_matrix,
            index=self._alternatives, columns=self._alternatives,
        ).round(4)

    def get_outranking_matrix(self) -> pd.DataFrame:
        """返回超越关系矩阵（0/1）DataFrame。"""
        if self._outranking_matrix is None:
            raise RuntimeError("请先调用 compute()。")
        return pd.DataFrame(
            self._outranking_matrix,
            index=self._alternatives, columns=self._alternatives,
        )

    def summary(self) -> Dict[str, Any]:
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")
        return {
            "method": "ELECTRE I",
            "concordance_threshold": self.concordance_threshold,
            "discordance_threshold": self.discordance_threshold,
            "n_alternatives": len(self._alternatives),
            "n_criteria": len(self._criteria),
            "weights": dict(zip(self._criteria, self.weights.tolist())),
            "net_dominance": dict(zip(self._alternatives,
                                       self._net_dominance.tolist())),
            "ranking": self.result["Rank"].to_dict(),
            "best": self.result.index[0],
            "outranking_pairs": int(self._outranking_matrix.sum()),
        }

    def plot_matrices(
        self,
        figsize: Tuple = (14, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """并排绘制一致性、非一致性与超越关系矩阵热力图。"""
        if self._concordance_matrix is None:
            raise RuntimeError("请先调用 compute()。")

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        matrices = [
            (self._concordance_matrix, "YlGn", "一致性矩阵 C"),
            (self._discordance_matrix, "YlOrRd", "非一致性矩阵 D"),
            (self._outranking_matrix.astype(float), "Blues", "超越关系矩阵 E"),
        ]
        for ax, (mat, cmap, title) in zip(axes, matrices):
            sns.heatmap(
                pd.DataFrame(mat, index=self._alternatives,
                             columns=self._alternatives),
                annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=45)

        fig.suptitle(
            f"ELECTRE 关系矩阵 (c̄={self.concordance_threshold}, "
            f"d̄={self.discordance_threshold})",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def tex_description(self) -> str:
        return r"""
\subsection{基于ELECTRE I的超越关系评价模型}

\subsubsection{模型原理}

ELECTRE I（Roy, 1968）通过构建一致性矩阵和非一致性矩阵，
判断方案间的超越关系（即某方案是否综合优于另一方案），
适用于存在不可比情况的决策问题。

\subsubsection{计算步骤}

\noindent\textbf{步骤1\quad 构造一致性矩阵}

方案 $i$ 相对于方案 $k$ 的一致性指数为：
\begin{equation}
    c_{ik} = \sum_{j:\,x_{ij} \geq x_{kj}} w_j
    \label{eq:electre-concordance}
\end{equation}
即在方案 $i$ 不劣于 $k$ 的所有指标上，对应权重之和。

\noindent\textbf{步骤2\quad 构造非一致性矩阵}

\begin{equation}
    d_{ik} = \max_{j:\,x_{kj} > x_{ij}}
    \frac{x_{kj} - x_{ij}}{\max_{i}x_{ij} - \min_{i}x_{ij}}
    \label{eq:electre-discordance}
\end{equation}
反映在方案 $k$ 优于 $i$ 的指标中，最大归一化差距。

\noindent\textbf{步骤3\quad 建立超越关系}

若同时满足：
\begin{equation}
    c_{ik} \geq \bar{c} \quad \text{且} \quad d_{ik} \leq \bar{d}
\end{equation}
则称方案 $i$ 超越方案 $k$（记为 $iSk$）。

\noindent\textbf{步骤4\quad 计算净支配流并排序}

\begin{equation}
    \varphi_i = \varphi_i^+ - \varphi_i^-
\end{equation}
其中 $\varphi_i^+$（$\varphi_i^-$）为方案 $i$ 超越（被超越）的方案数。
$\varphi_i$ 越大，综合评价越优。
"""