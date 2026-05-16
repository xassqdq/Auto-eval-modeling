"""
dynamic.py
动态时序综合评价封装

将静态评价方法（TOPSIS / GRA）扩展为多时间截面的动态评价，
支持时间权重自动计算（时间度函数）与综合动态排名。

Reference:
    刘思峰, 党耀国, 方志耕 (2010). 灰色系统理论及其应用（第5版）.
    科学出版社, 北京.
    郭亚军 (2007). 综合评价理论、方法及应用. 科学出版社, 北京.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .topsis import TOPSIS
from .gra import GRA

logger = logging.getLogger(__name__)

_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


# ──────────────────────────────────────────────────────────────────────────────
# 时间权重工具
# ──────────────────────────────────────────────────────────────────────────────

def compute_time_weights(
    n_periods: int,
    method: str = "exponential",
    decay: float = 0.85,
    custom_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    计算时间权重向量（越近期权重越高）。

    Parameters
    ----------
    n_periods : int
        时间截面数量 T。
    method : str
        权重计算方法：
        - ``'equal'``：等权，每期权重 = 1/T
        - ``'linear'``：线性递增，最新期权重最高
        - ``'exponential'``：指数衰减（最常用）
        - ``'time_degree'``：时间度函数（郭亚军方法）
        - ``'custom'``：自定义权重
    decay : float, default 0.85
        指数衰减系数 α ∈ (0,1)，仅 ``method='exponential'`` 时有效。
    custom_weights : list of float, optional
        自定义权重，``method='custom'`` 时必须提供。

    Returns
    -------
    np.ndarray, shape (n_periods,)
        归一化时间权重向量，下标 0 对应最早期，下标 -1 对应最近期。
    """
    T = n_periods
    if method == "equal":
        w = np.ones(T)
    elif method == "linear":
        w = np.arange(1, T + 1, dtype=float)
    elif method == "exponential":
        if not 0 < decay < 1:
            raise ValueError("decay 须在 (0,1) 之间。")
        # w_t ∝ α^(T-t)：最近期 t=T，权重最大
        w = np.array([decay ** (T - t - 1) for t in range(T)])
    elif method == "time_degree":
        # 郭亚军时间度函数：λ_t = t(t+1) / Σ_{k=1}^{T} k(k+1)
        w = np.array([t * (t + 1) for t in range(1, T + 1)], dtype=float)
    elif method == "custom":
        if custom_weights is None:
            raise ValueError("custom 模式需提供 custom_weights。")
        w = np.asarray(custom_weights, dtype=float)
        if len(w) != T:
            raise ValueError(f"custom_weights 长度 {len(w)} ≠ 时间截面数 {T}。")
    else:
        raise ValueError(f"未知时间权重方法: '{method}'。")

    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("时间权重之和须为正数。")
    return w / w_sum


# ──────────────────────────────────────────────────────────────────────────────
# 动态 TOPSIS
# ──────────────────────────────────────────────────────────────────────────────

class DynamicTOPSIS:
    """
    动态 TOPSIS 多时段综合评价。

    对每个时间截面独立运行 TOPSIS，
    然后通过时间权重聚合各期得分，得到最终动态综合排名。

    Parameters
    ----------
    weights : array-like, optional
        指标权重（各期相同）。
    indicator_types : list of str, optional
        指标方向（各期相同）。
    time_weight_method : str, default 'exponential'
        时间权重计算方法，参见 ``compute_time_weights``。
    time_decay : float, default 0.85
        指数时间衰减系数。
    normalize : bool, default True
        TOPSIS 向量归一化开关。

    Examples
    --------
    >>> import pandas as pd
    >>> # 3 个时期，每期 4 个对象 × 3 个指标
    >>> period_data = {
    ...     2021: pd.DataFrame({"A": [1,2,3,4], "B": [4,3,2,1], "C": [2,3,1,4]},
    ...                        index=["城市1","城市2","城市3","城市4"]),
    ...     2022: pd.DataFrame({"A": [2,3,4,5], "B": [3,2,2,1], "C": [3,4,2,5]},
    ...                        index=["城市1","城市2","城市3","城市4"]),
    ...     2023: pd.DataFrame({"A": [3,4,5,6], "B": [2,1,1,1], "C": [4,5,3,6]},
    ...                        index=["城市1","城市2","城市3","城市4"]),
    ... }
    >>> model = DynamicTOPSIS(indicator_types=["positive","negative","positive"])
    >>> model.fit(period_data)
    >>> result = model.compute()
    """

    def __init__(
        self,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        indicator_types: Optional[List[str]] = None,
        time_weight_method: str = "exponential",
        time_decay: float = 0.85,
        normalize: bool = True,
    ) -> None:
        self.weights = weights
        self.indicator_types = indicator_types
        self.time_weight_method = time_weight_method
        self.time_decay = time_decay
        self.normalize = normalize

        self._period_data: Optional[Dict[Any, pd.DataFrame]] = None
        self._period_labels: Optional[List] = None
        self._time_weights: Optional[np.ndarray] = None
        self._period_results: Optional[Dict[Any, pd.DataFrame]] = None
        self._period_scores: Optional[pd.DataFrame] = None  # (n_alt × n_period)
        self.result: Optional[pd.DataFrame] = None

    def fit(
        self,
        period_data: Dict[Any, pd.DataFrame],
        time_weights: Optional[List[float]] = None,
    ) -> "DynamicTOPSIS":
        """
        绑定多时段数据。

        Parameters
        ----------
        period_data : dict {period_label: DataFrame}
            有序字典，键为时间标签（年份等），值为该期决策矩阵。
            所有期次的行索引（评价对象）和列（指标）须完全相同。
        time_weights : list of float, optional
            自定义时间权重，覆盖 time_weight_method 设置。
        """
        if not period_data:
            raise ValueError("period_data 不能为空。")

        self._period_labels = list(period_data.keys())
        self._period_data = period_data
        T = len(self._period_labels)

        # 校验各期数据一致性
        ref_idx = list(period_data.values())[0].index
        ref_cols = list(period_data.values())[0].columns
        for label, df in period_data.items():
            if not df.index.equals(ref_idx):
                raise ValueError(
                    f"第 {label} 期数据的行索引与第一期不一致。"
                )
            if not df.columns.equals(ref_cols):
                raise ValueError(
                    f"第 {label} 期数据的列名与第一期不一致。"
                )

        # 时间权重
        if time_weights is not None:
            self._time_weights = np.asarray(time_weights, dtype=float)
            self._time_weights = self._time_weights / self._time_weights.sum()
        else:
            self._time_weights = compute_time_weights(
                T, method=self.time_weight_method, decay=self.time_decay
            )

        logger.info(
            "DynamicTOPSIS.fit: %d 个时间截面，%d 个对象，时间权重=%s",
            T, len(ref_idx), np.round(self._time_weights, 4).tolist(),
        )
        return self

    def compute(self) -> pd.DataFrame:
        """
        执行动态 TOPSIS 计算。

        Returns
        -------
        pd.DataFrame
            列：各期得分 + ``Dynamic_Score`` + ``Dynamic_Rank``，
            按动态综合得分降序排列。
        """
        if self._period_data is None:
            raise RuntimeError("请先调用 fit()。")

        self._period_results = {}
        all_scores: Dict[Any, pd.Series] = {}

        for label, df in self._period_data.items():
            topsis = TOPSIS(
                weights=self.weights,
                indicator_types=self.indicator_types,
                normalize=self.normalize,
            )
            topsis.fit(df)
            period_result = topsis.compute()
            self._period_results[label] = period_result
            all_scores[label] = period_result["Score (Ci)"]

        # 汇总各期得分矩阵
        self._period_scores = pd.DataFrame(all_scores)
        self._period_scores.columns = [
            f"Score_{lbl}" for lbl in self._period_labels
        ]

        # 动态综合得分 = Σ λ_t * C_it
        dyn_scores = (
            self._period_scores.values * self._time_weights[np.newaxis, :]
        ).sum(axis=1)

        dyn_ranks = (
            pd.Series(dyn_scores, index=self._period_scores.index)
            .rank(ascending=False, method="min")
            .astype(int)
        )

        self.result = self._period_scores.copy()
        self.result["Dynamic_Score"] = np.round(dyn_scores, 6)
        self.result["Dynamic_Rank"] = dyn_ranks
        self.result = self.result.sort_values("Dynamic_Rank")

        logger.info(
            "DynamicTOPSIS 计算完成 → 动态最优: %s",
            self.result.index[0],
        )
        return self.result

    def get_period_result(self, label: Any) -> pd.DataFrame:
        """获取指定时间截面的 TOPSIS 详细结果。"""
        if self._period_results is None:
            raise RuntimeError("请先调用 compute()。")
        if label not in self._period_results:
            raise KeyError(f"未知时间截面标签: {label}")
        return self._period_results[label]

    def plot_score_trajectory(
        self,
        top_n: Optional[int] = None,
        figsize: Tuple = (12, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制各评价对象跨时期得分折线图（时序轨迹）。"""
        if self._period_scores is None:
            raise RuntimeError("请先调用 compute()。")

        alts = self.result.head(top_n).index.tolist() if top_n else \
               self._period_scores.index.tolist()
        periods = self._period_labels
        x_ticks = range(len(periods))
        colors = plt.cm.tab10(np.linspace(0, 1, len(alts)))

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=figsize)
            for alt, color in zip(alts, colors):
                scores = [
                    self._period_results[lbl].loc[alt, "Score (Ci)"]
                    for lbl in periods
                ]
                ax.plot(x_ticks, scores, "o-", lw=2.5, ms=7,
                        color=color, label=alt)
                ax.annotate(
                    f"{scores[-1]:.3f}",
                    (x_ticks[-1], scores[-1]),
                    textcoords="offset points",
                    xytext=(8, 0), fontsize=8, color=color,
                )
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(p) for p in periods], fontsize=10)
            ax.set_xlabel("时间截面", fontsize=12)
            ax.set_ylabel(r"TOPSIS 得分 $C_i$", fontsize=12)
            title = f"动态 TOPSIS 时序轨迹"
            if top_n:
                title += f"（前 {top_n} 名）"
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
                      fontsize=9, framealpha=0.7)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_dynamic_heatmap(
        self,
        figsize: Tuple = (10, 7),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制各对象各期得分热力图（对象 × 时期）。"""
        if self._period_scores is None:
            raise RuntimeError("请先调用 compute()。")

        import seaborn as sns
        sorted_alts = self.result.index.tolist()
        score_cols = [c for c in self._period_scores.columns]
        hmap_data = self._period_scores.loc[sorted_alts, score_cols]
        hmap_data.columns = [str(lbl) for lbl in self._period_labels]

        with plt.rc_context({"figure.dpi": 120}):
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                hmap_data, annot=True, fmt=".3f", cmap="RdYlGn",
                linewidths=0.5, vmin=0, vmax=1,
                cbar_kws={"label": "TOPSIS 得分"},
                ax=ax,
            )
            ax.set_title("动态 TOPSIS — 时序得分热力图",
                         fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("时间截面", fontsize=11)
            ax.set_ylabel("评价对象（按动态排名）", fontsize=11)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def summary(self) -> Dict[str, Any]:
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")
        return {
            "method": "DynamicTOPSIS",
            "n_periods": len(self._period_labels),
            "period_labels": self._period_labels,
            "time_weights": dict(
                zip(self._period_labels, np.round(self._time_weights, 4).tolist())
            ),
            "time_weight_method": self.time_weight_method,
            "dynamic_scores": self.result["Dynamic_Score"].to_dict(),
            "dynamic_ranking": self.result["Dynamic_Rank"].to_dict(),
            "best": self.result.index[0],
            "period_scores": {
                lbl: res["Score (Ci)"].to_dict()
                for lbl, res in self._period_results.items()
            },
        }

    def tex_description(self) -> str:
        return r"""
\subsection{动态 TOPSIS 多时段综合评价模型}

\subsubsection{模型框架}

静态 TOPSIS 仅能评价单一时间截面，当存在多个时期的纵向数据时，
需引入\textbf{时间权重}对各期评价结果进行加权聚合，
以反映时间趋势对综合评价的影响（郭亚军，2007）。

\subsubsection{计算流程}

\noindent\textbf{步骤1\quad 各期独立 TOPSIS 评价}

对 $T$ 个时间截面 $t = 1, 2, \ldots, T$，
分别计算各评价对象在第 $t$ 期的 TOPSIS 相对贴近度 $C_i^{(t)}$。

\noindent\textbf{步骤2\quad 确定时间权重}

采用指数衰减时间权重，赋予近期数据更高权重：
\begin{equation}
    \lambda_t = \frac{\alpha^{T-t}}{\displaystyle\sum_{k=1}^{T}\alpha^{T-k}},
    \quad \alpha \in (0,1),\quad \sum_{t=1}^{T}\lambda_t = 1
    \label{eq:dynamic-time-weight}
\end{equation}

\noindent\textbf{步骤3\quad 动态综合得分}

\begin{equation}
    C_i^{\text{dyn}} = \sum_{t=1}^{T} \lambda_t \cdot C_i^{(t)}
    \label{eq:dynamic-score}
\end{equation}

按 $C_i^{\text{dyn}}$ 降序排列，得到最终动态综合排名。
"""


# ──────────────────────────────────────────────────────────────────────────────
# 动态 GRA
# ──────────────────────────────────────────────────────────────────────────────

class DynamicGRA:
    """
    动态灰色关联分析多时段综合评价。

    对每个时间截面独立运行 GRA，
    通过时间权重聚合各期关联度，得到动态综合排名。

    Parameters
    ----------
    rho : float, default 0.5
        GRA 分辨系数。
    weights : array-like, optional
        指标权重（各期相同）。
    indicator_types : list of str, optional
        指标方向（各期相同）。
    time_weight_method : str, default 'exponential'
        时间权重方法。
    time_decay : float, default 0.85
        时间衰减系数。
    """

    def __init__(
        self,
        rho: float = 0.5,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        indicator_types: Optional[List[str]] = None,
        time_weight_method: str = "exponential",
        time_decay: float = 0.85,
    ) -> None:
        self.rho = rho
        self.weights = weights
        self.indicator_types = indicator_types
        self.time_weight_method = time_weight_method
        self.time_decay = time_decay

        self._period_data: Optional[Dict[Any, pd.DataFrame]] = None
        self._period_labels: Optional[List] = None
        self._time_weights: Optional[np.ndarray] = None
        self._period_results: Optional[Dict[Any, pd.DataFrame]] = None
        self._period_grades: Optional[pd.DataFrame] = None
        self.result: Optional[pd.DataFrame] = None

    def fit(
        self,
        period_data: Dict[Any, pd.DataFrame],
        time_weights: Optional[List[float]] = None,
    ) -> "DynamicGRA":
        """绑定多时段数据（接口同 DynamicTOPSIS.fit）。"""
        if not period_data:
            raise ValueError("period_data 不能为空。")

        self._period_labels = list(period_data.keys())
        self._period_data = period_data
        T = len(self._period_labels)

        ref_idx = list(period_data.values())[0].index
        ref_cols = list(period_data.values())[0].columns
        for label, df in period_data.items():
            if not df.index.equals(ref_idx) or not df.columns.equals(ref_cols):
                raise ValueError(f"第 {label} 期数据结构与第一期不一致。")

        if time_weights is not None:
            self._time_weights = np.asarray(time_weights, dtype=float)
            self._time_weights = self._time_weights / self._time_weights.sum()
        else:
            self._time_weights = compute_time_weights(
                T, method=self.time_weight_method, decay=self.time_decay
            )
        return self

    def compute(self) -> pd.DataFrame:
        """
        执行动态 GRA 计算。

        Returns
        -------
        pd.DataFrame
            列：各期关联度 + ``Dynamic_Grade`` + ``Dynamic_Rank``。
        """
        if self._period_data is None:
            raise RuntimeError("请先调用 fit()。")

        self._period_results = {}
        all_grades: Dict[Any, pd.Series] = {}

        for label, df in self._period_data.items():
            gra = GRA(
                rho=self.rho,
                weights=self.weights,
                indicator_types=self.indicator_types,
            )
            gra.fit(df)
            period_result = gra.compute()
            self._period_results[label] = period_result
            all_grades[label] = period_result["Grey Relational Grade"]

        self._period_grades = pd.DataFrame(all_grades)
        self._period_grades.columns = [
            f"Grade_{lbl}" for lbl in self._period_labels
        ]

        dyn_grades = (
            self._period_grades.values * self._time_weights[np.newaxis, :]
        ).sum(axis=1)

        dyn_ranks = (
            pd.Series(dyn_grades, index=self._period_grades.index)
            .rank(ascending=False, method="min")
            .astype(int)
        )

        self.result = self._period_grades.copy()
        self.result["Dynamic_Grade"] = np.round(dyn_grades, 6)
        self.result["Dynamic_Rank"] = dyn_ranks
        self.result = self.result.sort_values("Dynamic_Rank")

        logger.info(
            "DynamicGRA 计算完成 → 动态最优: %s",
            self.result.index[0],
        )
        return self.result

    def plot_grade_trajectory(
        self,
        top_n: Optional[int] = None,
        figsize: Tuple = (12, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制各对象跨时期关联度轨迹折线图。"""
        if self._period_grades is None:
            raise RuntimeError("请先调用 compute()。")

        alts = self.result.head(top_n).index.tolist() if top_n else \
               self._period_grades.index.tolist()
        periods = self._period_labels
        colors = plt.cm.tab10(np.linspace(0, 1, len(alts)))

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=figsize)
            for alt, color in zip(alts, colors):
                grades = [
                    self._period_results[lbl].loc[alt, "Grey Relational Grade"]
                    for lbl in periods
                ]
                ax.plot(range(len(periods)), grades, "s-",
                        lw=2.5, ms=7, color=color, label=alt)
                ax.annotate(
                    f"{grades[-1]:.3f}",
                    (len(periods) - 1, grades[-1]),
                    textcoords="offset points",
                    xytext=(8, 0), fontsize=8, color=color,
                )
            ax.set_xticks(range(len(periods)))
            ax.set_xticklabels([str(p) for p in periods], fontsize=10)
            ax.set_xlabel("时间截面", fontsize=12)
            ax.set_ylabel("灰色关联度 $\\gamma_i$", fontsize=12)
            title = "动态 GRA 时序轨迹"
            if top_n:
                title += f"（前 {top_n} 名）"
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
                      fontsize=9, framealpha=0.7)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def summary(self) -> Dict[str, Any]:
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")
        return {
            "method": "DynamicGRA",
            "rho": self.rho,
            "n_periods": len(self._period_labels),
            "period_labels": self._period_labels,
            "time_weights": dict(
                zip(self._period_labels, np.round(self._time_weights, 4).tolist())
            ),
            "dynamic_grades": self.result["Dynamic_Grade"].to_dict(),
            "dynamic_ranking": self.result["Dynamic_Rank"].to_dict(),
            "best": self.result.index[0],
        }

    def tex_description(self) -> str:
        return r"""
\subsection{动态灰色关联分析多时段评价模型}

\subsubsection{模型框架}

将灰色关联分析（GRA）推广至多时间截面，
通过时间权重 $\boldsymbol{\lambda}$ 对各期灰色关联度进行加权聚合，
以捕捉评价对象的动态演变趋势。

\subsubsection{动态综合关联度}

设第 $t$ 期对象 $i$ 的灰色关联度为 $\gamma_i^{(t)}$，
动态综合关联度为：
\begin{equation}
    \gamma_i^{\text{dyn}} = \sum_{t=1}^{T} \lambda_t \cdot \gamma_i^{(t)},
    \quad \sum_{t=1}^{T}\lambda_t = 1
    \label{eq:dynamic-gra}
\end{equation}

按 $\gamma_i^{\text{dyn}}$ 降序排列得到最终动态综合排名。
"""


# ──────────────────────────────────────────────────────────────────────────────
# 统一动态评价接口
# ──────────────────────────────────────────────────────────────────────────────

class DynamicEvaluation:
    """
    统一动态多时段综合评价接口。

    支持同时运行 DynamicTOPSIS 与 DynamicGRA，
    并通过 Borda 计数或简单平均对两种方法的动态排名进行融合，
    输出最终一致性排名。

    Parameters
    ----------
    methods : list of str, default ['topsis', 'gra']
        启用的评价方法，可选 ``'topsis'`` / ``'gra'``。
    fusion : str, default 'borda'
        排名融合方式：``'borda'``（Borda 计数）或 ``'average_score'``（得分平均）。
    topsis_kwargs : dict, optional
        传递给 DynamicTOPSIS 的构造参数。
    gra_kwargs : dict, optional
        传递给 DynamicGRA 的构造参数。
    """

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        fusion: str = "borda",
        topsis_kwargs: Optional[Dict[str, Any]] = None,
        gra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.methods = methods or ["topsis", "gra"]
        self.fusion = fusion
        self.topsis_kwargs = topsis_kwargs or {}
        self.gra_kwargs = gra_kwargs or {}

        self._dyn_topsis: Optional[DynamicTOPSIS] = None
        self._dyn_gra: Optional[DynamicGRA] = None
        self._alternatives: Optional[List] = None
        self.result: Optional[pd.DataFrame] = None

    def fit(
        self,
        period_data: Dict[Any, pd.DataFrame],
        time_weights: Optional[List[float]] = None,
    ) -> "DynamicEvaluation":
        """绑定多时段数据并初始化子模型。"""
        alts = list(list(period_data.values())[0].index)
        self._alternatives = alts

        if "topsis" in self.methods:
            self._dyn_topsis = DynamicTOPSIS(**self.topsis_kwargs)
            self._dyn_topsis.fit(period_data, time_weights=time_weights)

        if "gra" in self.methods:
            self._dyn_gra = DynamicGRA(**self.gra_kwargs)
            self._dyn_gra.fit(period_data, time_weights=time_weights)

        return self

    def compute(self) -> pd.DataFrame:
        """
        执行融合动态评价计算。

        Returns
        -------
        pd.DataFrame
            列：各方法动态得分/关联度 + 各方法动态排名 + ``Fused_Rank``，
            按融合排名升序排列。
        """
        if not self.methods:
            raise RuntimeError("未指定评价方法。")

        all_ranks: Dict[str, pd.Series] = {}
        all_scores: Dict[str, pd.Series] = {}

        if self._dyn_topsis is not None:
            topsis_result = self._dyn_topsis.compute()
            all_scores["TOPSIS_Dynamic_Score"] = topsis_result["Dynamic_Score"]
            all_ranks["TOPSIS_Rank"] = topsis_result["Dynamic_Rank"]

        if self._dyn_gra is not None:
            gra_result = self._dyn_gra.compute()
            all_scores["GRA_Dynamic_Grade"] = gra_result["Dynamic_Grade"]
            all_ranks["GRA_Rank"] = gra_result["Dynamic_Rank"]

        combined = pd.DataFrame(
            {**all_scores, **all_ranks}, index=self._alternatives
        )

        # 融合排名
        if self.fusion == "borda":
            # Borda 计数：得分 = Σ (n - rank_k + 1)
            n = len(self._alternatives)
            rank_cols = list(all_ranks.keys())
            borda_scores = sum(
                (n - combined[rc] + 1) for rc in rank_cols
            )
            fused_rank = (
                borda_scores.rank(ascending=False, method="min")
                .astype(int)
            )
            combined["Borda_Score"] = borda_scores.astype(int)
            combined["Fused_Rank"] = fused_rank

        elif self.fusion == "average_score":
            # 归一化各方法得分后取平均
            score_cols = list(all_scores.keys())
            norm_scores = pd.DataFrame(index=self._alternatives)
            for sc in score_cols:
                s = combined[sc]
                rng = s.max() - s.min()
                norm_scores[sc] = (s - s.min()) / rng if rng > 1e-10 else 0.5
            avg_score = norm_scores.mean(axis=1)
            combined["Average_Score"] = np.round(avg_score, 6)
            combined["Fused_Rank"] = (
                avg_score.rank(ascending=False, method="min").astype(int)
            )
        else:
            raise ValueError(f"未知融合方式: '{self.fusion}'。")

        self.result = combined.sort_values("Fused_Rank")
        logger.info(
            "DynamicEvaluation 融合完成 (%s) → 最优: %s",
            self.fusion, self.result.index[0],
        )
        return self.result

    def summary(self) -> Dict[str, Any]:
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")
        s: Dict[str, Any] = {
            "method": "DynamicEvaluation",
            "sub_methods": self.methods,
            "fusion": self.fusion,
            "n_alternatives": len(self._alternatives),
            "fused_ranking": self.result["Fused_Rank"].to_dict(),
            "best": self.result.index[0],
        }
        if self._dyn_topsis is not None:
            s["topsis_summary"] = self._dyn_topsis.summary()
        if self._dyn_gra is not None:
            s["gra_summary"] = self._dyn_gra.summary()
        return s

    def plot_rank_comparison(
        self,
        figsize: Tuple = (12, 7),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制各方法动态排名对比矩阵热力图 + 最终融合排名。"""
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")

        import seaborn as sns
        rank_cols = [c for c in self.result.columns if "Rank" in c]
        sorted_alts = self.result.index.tolist()
        rank_data = self.result.loc[sorted_alts, rank_cols]

        col_labels = {
            "TOPSIS_Rank": "TOPSIS\n动态排名",
            "GRA_Rank": "GRA\n动态排名",
            "Fused_Rank": "融合\n排名",
        }
        rank_data = rank_data.rename(
            columns={c: col_labels.get(c, c) for c in rank_data.columns}
        )

        with plt.rc_context({"figure.dpi": 120}):
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                rank_data, annot=True, fmt="d",
                cmap="YlOrRd_r",
                linewidths=0.5, linecolor="grey",
                cbar_kws={"label": "排名（越小越优）"},
                ax=ax,
            )
            ax.set_title(
                f"动态综合评价排名对比（融合方式：{self.fusion}）",
                fontsize=13, fontweight="bold", pad=12,
            )
            ax.set_xlabel("评价方法", fontsize=11)
            ax.set_ylabel("评价对象（按融合排名）", fontsize=11)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def tex_description(self) -> str:
        return r"""
\subsection{多方法融合动态综合评价模型}

\subsubsection{框架设计}

为增强综合评价结果的稳健性，本文同时采用动态TOPSIS和动态灰色关联分析
两种方法对各评价对象进行独立评价，并通过\textbf{Borda计数法}
对两组动态排名进行融合，最终得到一致性排名结果。

\subsubsection{Borda 计数融合}

设共有 $K$ 种评价方法，第 $k$ 种方法对对象 $i$ 的排名为 $r_i^{(k)}$，
则 Borda 得分为：
\begin{equation}
    B_i = \sum_{k=1}^{K}\left(n - r_i^{(k)} + 1\right)
    \label{eq:borda}
\end{equation}
其中 $n$ 为评价对象总数。
$B_i$ 越大表示在多种方法中排名越靠前，
最终按 $B_i$ 降序排列得到融合综合排名。
该方法有效降低了单一方法可能存在的偏差，
使综合排名更加客观稳健。
"""