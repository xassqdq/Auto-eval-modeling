"""
vikor.py
VIKOR — VIseKriterijumska Optimizacija I Kompromisno Resenje
多准则妥协排序法

Reference:
    Opricovic, S. (1998). Multicriteria optimization of civil engineering systems.
    Faculty of Civil Engineering, Belgrade, 2(1), 5-21.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

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


class VIKOR(BaseMethod):
    """
    VIKOR 多准则妥协排序法。

    在多准则不可公度（指标量纲不同）且相互冲突时，
    通过引入群体效用和个体遗憾两个度量寻找妥协解。

    Parameters
    ----------
    v : float, default 0.5
        群体效用权重 $v\\in[0,1]$。
        $v>0.5$ 偏重多数原则；$v<0.5$ 偏重个体遗憾最小。
    weights : array-like, optional
        指标权重向量，默认等权。
    indicator_types : list of str, optional
        'positive' 或 'negative'，默认全正向。
    """

    def __init__(
        self,
        v: float = 0.5,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        indicator_types: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name="VIKOR")
        if not 0 <= v <= 1:
            raise ValueError(f"v 必须在 [0,1] 范围内，当前值: {v}")
        self.v = v
        self.weights: Optional[np.ndarray] = (
            np.asarray(weights, dtype=float) if weights is not None else None
        )
        self.indicator_types = indicator_types

        self._data: Optional[pd.DataFrame] = None
        self._alternatives: Optional[List] = None
        self._criteria: Optional[List] = None
        self._f_best: Optional[np.ndarray] = None   # f*
        self._f_worst: Optional[np.ndarray] = None  # f-
        self._S: Optional[np.ndarray] = None  # 效用值
        self._R: Optional[np.ndarray] = None  # 遗憾值
        self._Q: Optional[np.ndarray] = None  # 妥协排序值

    def fit(
        self,
        data: pd.DataFrame,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        indicator_types: Optional[List[str]] = None,
    ) -> "VIKOR":
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
        执行 VIKOR 计算。

        Returns
        -------
        pd.DataFrame
            列：``S`` / ``R`` / ``Q`` / ``Rank_S`` / ``Rank_R`` / ``Rank_Q``，
            按 Q 值升序排列（Q 越小越优）。
        """
        if self._data is None:
            raise RuntimeError("请先调用 fit()。")

        X = self._data.values.astype(float)
        n, m = X.shape

        # ① 确定最优最劣参考值
        self._f_best = np.where(
            [t == "positive" for t in self.indicator_types],
            X.max(axis=0), X.min(axis=0),
        )
        self._f_worst = np.where(
            [t == "positive" for t in self.indicator_types],
            X.min(axis=0), X.max(axis=0),
        )

        # ② 计算规范化差值（分母为零时置 0）
        denom = self._f_best - self._f_worst
        # 避免除零：若某指标所有值相同，该项差值贡献为 0
        denom_safe = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        norm_diff = np.abs(X - self._f_best) / np.abs(denom_safe)
        # 若指标范围为 0，强制置 0
        zero_range = np.abs(denom) < 1e-12
        norm_diff[:, zero_range] = 0.0

        weighted_diff = self.weights * norm_diff  # (n, m)

        # ③ 效用值 S_i（加权和）和遗憾值 R_i（加权最大）
        self._S = weighted_diff.sum(axis=1)
        self._R = weighted_diff.max(axis=1)

        # ④ 妥协排序值 Q_i
        S_star, S_minus = self._S.min(), self._S.max()
        R_star, R_minus = self._R.min(), self._R.max()

        Q_S = (self._S - S_star) / (S_minus - S_star + 1e-12)
        Q_R = (self._R - R_star) / (R_minus - R_star + 1e-12)
        self._Q = self.v * Q_S + (1 - self.v) * Q_R

        # ⑤ 分别排名
        s_series = pd.Series(self._S, index=self._alternatives)
        r_series = pd.Series(self._R, index=self._alternatives)
        q_series = pd.Series(self._Q, index=self._alternatives)

        rank_s = s_series.rank(ascending=True, method="min").astype(int)
        rank_r = r_series.rank(ascending=True, method="min").astype(int)
        rank_q = q_series.rank(ascending=True, method="min").astype(int)

        self.result = pd.DataFrame(
            {
                "S (效用值)": np.round(self._S, 6),
                "R (遗憾值)": np.round(self._R, 6),
                "Q (妥协值)": np.round(self._Q, 6),
                "Rank_S": rank_s,
                "Rank_R": rank_r,
                "Rank_Q": rank_q,
            },
            index=self._alternatives,
        ).sort_values("Q (妥协值)")

        # ⑥ 妥协解条件检验
        self._check_compromise_solution()

        self.metadata.update(
            {
                "S_star": float(S_star),
                "S_minus": float(S_minus),
                "R_star": float(R_star),
                "R_minus": float(R_minus),
                "v": self.v,
                "best_Q": self.result.index[0],
            }
        )
        logger.info("VIKOR 计算完成 → 妥协最优: %s", self.result.index[0])
        return self.result

    def _check_compromise_solution(self) -> Dict[str, bool]:
        """检验 VIKOR 妥协解的两个充分条件。"""
        if self.result is None:
            return {}
        sorted_alts = self.result.index.tolist()
        a1, a2 = sorted_alts[0], sorted_alts[1] if len(sorted_alts) > 1 else sorted_alts[0]
        n = len(sorted_alts)

        # 条件1：可接受优势
        dq = self.result.loc[a2, "Q (妥协值)"] - self.result.loc[a1, "Q (妥协值)"]
        dq_threshold = 1.0 / max(n - 1, 1)
        cond1 = bool(dq >= dq_threshold)

        # 条件2：可接受稳定性
        a1_rank_s = self.result.loc[a1, "Rank_S"]
        a1_rank_r = self.result.loc[a1, "Rank_R"]
        cond2 = bool(a1_rank_s == 1 or a1_rank_r == 1)

        self.metadata.update(
            {
                "condition1_advantage": cond1,
                "condition2_stability": cond2,
                "compromise_valid": cond1 and cond2,
                "dq": float(dq),
                "dq_threshold": float(dq_threshold),
            }
        )
        if not (cond1 and cond2):
            logger.warning(
                "VIKOR 妥协解条件未完全满足 (C1=%s, C2=%s)，"
                "建议考虑妥协解集合。",
                cond1, cond2,
            )
        return {"condition1": cond1, "condition2": cond2}

    def summary(self) -> Dict[str, Any]:
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")
        return {
            "method": "VIKOR",
            "v_parameter": self.v,
            "n_alternatives": len(self._alternatives),
            "n_criteria": len(self._criteria),
            "weights": dict(zip(self._criteria, self.weights.tolist())),
            "f_best": dict(zip(self._criteria, self._f_best.round(6).tolist())),
            "f_worst": dict(zip(self._criteria, self._f_worst.round(6).tolist())),
            "S_values": self.result["S (效用值)"].to_dict(),
            "R_values": self.result["R (遗憾值)"].to_dict(),
            "Q_values": self.result["Q (妥协值)"].to_dict(),
            "ranking_Q": self.result["Rank_Q"].to_dict(),
            "best_compromise": self.result.index[0],
            "compromise_valid": self.metadata.get("compromise_valid", None),
        }

    def plot_srq(
        self,
        figsize: Tuple = (12, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制 S、R、Q 三组排序对比折线图。"""
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")

        with plt.rc_context(_STYLE):
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            order = self.result.sort_values("Q (妥协值)").index.tolist()

            for ax, col, color, title in zip(
                axes,
                ["S (效用值)", "R (遗憾值)", "Q (妥协值)"],
                ["#3498db", "#e74c3c", "#2ecc71"],
                ["效用值 S", "遗憾值 R", "妥协值 Q"],
            ):
                vals = self.result.loc[order, col].values
                ax.bar(range(len(order)), vals, color=color, alpha=0.8,
                       edgecolor="black", linewidth=0.5)
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.set_ylabel("数值", fontsize=9)

            fig.suptitle(f"VIKOR 评价结果（v={self.v}）",
                         fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def tex_description(self) -> str:
        return r"""
\subsection{基于VIKOR的多准则妥协排序模型}

\subsubsection{模型原理}

VIKOR方法（Opricovic, 1998）通过引入\textbf{群体效用}和\textbf{个体遗憾}
两个度量指标，在存在不可公度且相互冲突的准则时寻求妥协解。

\subsubsection{计算步骤}

\noindent\textbf{步骤1\quad 确定最优最劣参考值}

对效益型指标 $f_j^* = \max_i f_{ij}$，$f_j^- = \min_i f_{ij}$；
成本型反之。

\noindent\textbf{步骤2\quad 计算效用值与遗憾值}

\begin{align}
    S_i &= \sum_{j=1}^{m} w_j \frac{\left|f_j^* - f_{ij}\right|}
            {\left|f_j^* - f_j^-\right|}
    \label{eq:vikor-S} \\
    R_i &= \max_{j}\left\{ w_j \frac{\left|f_j^* - f_{ij}\right|}
            {\left|f_j^* - f_j^-\right|} \right\}
    \label{eq:vikor-R}
\end{align}

\noindent\textbf{步骤3\quad 计算妥协排序值}

\begin{equation}
    Q_i = v\frac{S_i - S^*}{S^- - S^*} + (1-v)\frac{R_i - R^*}{R^- - R^*}
    \label{eq:vikor-Q}
\end{equation}
其中 $S^*=\min_i S_i$，$S^-=\max_i S_i$，$R^*$、$R^-$类似，
$v\in[0,1]$ 为决策者对群体效用的偏好权重（本文取 $v=0.5$）。

\noindent\textbf{步骤4\quad 妥协解条件检验}

\begin{itemize}
    \item \textbf{条件1（可接受优势）}：$Q(a^{(2)}) - Q(a^{(1)}) \geq DQ$，
          其中 $DQ = 1/(n-1)$；
    \item \textbf{条件2（可接受稳定性）}：$a^{(1)}$在$S$或$R$排名中亦为第一。
\end{itemize}

按 $Q_i$ 升序排列，$Q_i$ 最小者为综合最优妥协方案。
"""