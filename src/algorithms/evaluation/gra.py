"""
gra.py
GRA — Grey Relational Analysis
灰色关联分析

Reference:
    Deng, J.L. (1989). Introduction to grey system theory.
    The Journal of Grey System, 1(1), 1-24.
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
    "axes.grid": True,
    "grid.alpha": 0.3,
}


class GRA(BaseMethod):
    """
    灰色关联分析（Grey Relational Analysis）。

    通过计算各评价对象与参考序列之间的灰色关联度，
    对评价对象进行综合排序。

    Parameters
    ----------
    rho : float, default 0.5
        分辨系数（辨别系数），取值范围 $(0,1)$，通常取 0.5。
        值越小，分辨力越强。
    weights : array-like, optional
        指标权重向量，默认等权。
    indicator_types : list of str, optional
        'positive' 或 'negative'，用于构造参考序列，默认全正向。
    reference_mode : str, default 'optimal'
        参考序列构造方式：
        - ``'optimal'``：正向取最大值，负向取最小值（最优参考）
        - ``'custom'``：由 ``reference`` 参数指定
    """

    def __init__(
        self,
        rho: float = 0.5,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        indicator_types: Optional[List[str]] = None,
        reference_mode: str = "optimal",
    ) -> None:
        super().__init__(name="GRA")
        if not 0 < rho < 1:
            raise ValueError(f"分辨系数 rho 必须在 (0,1) 内，当前: {rho}")
        self.rho = rho
        self.weights: Optional[np.ndarray] = (
            np.asarray(weights, dtype=float) if weights is not None else None
        )
        self.indicator_types = indicator_types
        self.reference_mode = reference_mode

        self._data: Optional[pd.DataFrame] = None
        self._alternatives: Optional[List] = None
        self._criteria: Optional[List] = None
        self._reference: Optional[np.ndarray] = None
        self._norm_matrix: Optional[np.ndarray] = None
        self._diff_matrix: Optional[np.ndarray] = None    # 差值序列
        self._coeff_matrix: Optional[np.ndarray] = None  # 关联系数矩阵
        self._grades: Optional[np.ndarray] = None         # 关联度

    def fit(
        self,
        data: pd.DataFrame,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        indicator_types: Optional[List[str]] = None,
        reference: Optional[Union[List[float], np.ndarray]] = None,
    ) -> "GRA":
        """
        绑定数据与参数。

        Parameters
        ----------
        data : pd.DataFrame
            决策矩阵（已预处理，值均非负）。
        reference : array-like, optional
            自定义参考序列，仅当 ``reference_mode='custom'`` 时有效。
        """
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

        if reference is not None and self.reference_mode == "custom":
            self._reference = np.asarray(reference, dtype=float)
            if len(self._reference) != m:
                raise ValueError("自定义参考序列长度须与指标数一致。")

        if len(self.weights) != m or len(self.indicator_types) != m:
            raise ValueError("权重或指标类型维度与指标数不匹配。")
        return self

    def compute(self) -> pd.DataFrame:
        """
        执行灰色关联分析。

        Returns
        -------
        pd.DataFrame
            列：``Grey Relational Grade`` / ``Rank``，按关联度降序排列。
        """
        if self._data is None:
            raise RuntimeError("请先调用 fit()。")

        X = self._data.values.astype(float)
        n, m = X.shape

        # ① 均值化（min-max 线性标准化至 [0,1]）
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        rng = x_max - x_min
        rng[rng < 1e-12] = 1.0
        X_norm = (X - x_min) / rng
        # 负向指标反向（越小越好 → 变为越大越好）
        for j, t in enumerate(self.indicator_types):
            if t == "negative":
                X_norm[:, j] = 1.0 - X_norm[:, j]
        self._norm_matrix = X_norm

        # ② 参考序列
        if self.reference_mode == "custom" and self._reference is not None:
            ref_norm = (self._reference - x_min) / rng
            for j, t in enumerate(self.indicator_types):
                if t == "negative":
                    ref_norm[j] = 1.0 - ref_norm[j]
            x0 = ref_norm
        else:
            # 最优参考序列：归一化后所有指标均取 1.0
            x0 = np.ones(m)

        # ③ 差值序列 Δ_0i(k) = |x_0(k) - x_i(k)|
        self._diff_matrix = np.abs(X_norm - x0)  # (n, m)

        # ④ 全局最大/最小差
        delta_max = self._diff_matrix.max()
        delta_min = self._diff_matrix.min()

        # ⑤ 灰色关联系数
        numerator = delta_min + self.rho * delta_max
        denominator = self._diff_matrix + self.rho * delta_max
        denominator[denominator < 1e-12] = 1e-12
        self._coeff_matrix = numerator / denominator  # (n, m)

        # ⑥ 灰色关联度（加权平均）
        self._grades = (self._coeff_matrix * self.weights).sum(axis=1)

        ranks = (
            pd.Series(self._grades, index=self._alternatives)
            .rank(ascending=False, method="min")
            .astype(int)
        )

        self.result = pd.DataFrame(
            {
                "Grey Relational Grade": np.round(self._grades, 6),
                "Rank": ranks,
            },
            index=self._alternatives,
        ).sort_values("Rank")

        self.metadata.update(
            {
                "rho": self.rho,
                "reference_mode": self.reference_mode,
                "delta_max": float(delta_max),
                "delta_min": float(delta_min),
                "best": self.result.index[0],
                "worst": self.result.index[-1],
            }
        )
        logger.info(
            "GRA 计算完成 → 最优: %s (grade=%.4f)",
            self.result.index[0], self._grades.max(),
        )
        return self.result

    def get_coefficient_matrix(self) -> pd.DataFrame:
        """获取灰色关联系数矩阵（n×m）。"""
        if self._coeff_matrix is None:
            raise RuntimeError("请先调用 compute()。")
        return pd.DataFrame(
            self._coeff_matrix,
            index=self._alternatives,
            columns=self._criteria,
        ).round(4)

    def summary(self) -> Dict[str, Any]:
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")
        return {
            "method": "GRA",
            "description": "灰色关联分析",
            "rho": self.rho,
            "reference_mode": self.reference_mode,
            "n_alternatives": len(self._alternatives),
            "n_criteria": len(self._criteria),
            "weights": dict(zip(self._criteria, self.weights.tolist())),
            "grades": self.result["Grey Relational Grade"].to_dict(),
            "ranking": self.result["Rank"].to_dict(),
            "best": self.result.index[0],
            "worst": self.result.index[-1],
            "coefficient_matrix": self.get_coefficient_matrix().to_dict(),
        }

    def plot_grades(
        self,
        figsize: Tuple = (10, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制灰色关联度条形图。"""
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=figsize)
            sorted_df = self.result.sort_values("Grey Relational Grade", ascending=True)
            grades = sorted_df["Grey Relational Grade"].values
            labels = sorted_df.index.tolist()

            cmap = plt.get_cmap("Blues")
            norm = plt.Normalize(grades.min() * 0.9, grades.max())
            bars = ax.barh(
                range(len(labels)), grades,
                color=[cmap(norm(g)) for g in grades],
                edgecolor="grey", linewidth=0.5, height=0.65,
            )
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=11)
            ax.set_xlabel("灰色关联度 $\\gamma_i$", fontsize=12)
            ax.set_title(f"灰色关联分析综合评价 (ρ={self.rho})",
                         fontsize=13, fontweight="bold")
            ax.set_xlim(0, grades.max() * 1.2)
            for bar, g in zip(bars, grades):
                ax.text(g + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{g:.4f}", va="center", fontsize=9)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_coefficient_heatmap(
        self,
        figsize: Tuple = (10, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制关联系数热力图。"""
        if self._coeff_matrix is None:
            raise RuntimeError("请先调用 compute()。")

        df = self.get_coefficient_matrix()
        sorted_alts = self.result.index.tolist()
        df = df.loc[sorted_alts]

        with plt.rc_context({"figure.dpi": 120}):
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                df, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, linecolor="grey",
                cbar_kws={"label": "关联系数"},
                ax=ax,
            )
            ax.set_title("灰色关联系数热力图", fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("评价指标", fontsize=11)
            ax.set_ylabel("评价对象（按综合排名）", fontsize=11)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def tex_description(self) -> str:
        return r"""
\subsection{基于灰色关联分析的综合评价模型}

\subsubsection{模型原理}

灰色关联分析（Grey Relational Analysis, GRA）由邓聚龙（1989）提出，
适用于信息不完全、样本量较小的综合评价问题。
其核心是通过计算各评价对象序列与参考序列的几何相似程度来衡量优劣。

\subsubsection{计算步骤}

\noindent\textbf{步骤1\quad 数据归一化}

将原始矩阵进行极值标准化（效益型 $\uparrow$，成本型 $\downarrow$）：
\begin{equation}
    x_i'(k) = \frac{x_i(k) - \min_i x_i(k)}{\max_i x_i(k) - \min_i x_i(k)}
    \label{eq:gra-norm}
\end{equation}

\noindent\textbf{步骤2\quad 构造参考序列}

取最优参考序列 $\boldsymbol{x}_0 = (1, 1, \ldots, 1)$。

\noindent\textbf{步骤3\quad 计算差值序列}

\begin{equation}
    \Delta_{0i}(k) = \left|x_0(k) - x_i'(k)\right|
    \label{eq:gra-diff}
\end{equation}

\noindent\textbf{步骤4\quad 计算灰色关联系数}

\begin{equation}
    \xi_i(k) = \frac{\Delta_{\min} + \rho\,\Delta_{\max}}
                    {\Delta_{0i}(k) + \rho\,\Delta_{\max}},
    \quad \rho \in (0,1)
    \label{eq:gra-coeff}
\end{equation}
其中 $\Delta_{\min} = \min_{i,k}\Delta_{0i}(k)$，
$\Delta_{\max} = \max_{i,k}\Delta_{0i}(k)$，
分辨系数 $\rho$ 通常取 $0.5$。

\noindent\textbf{步骤5\quad 计算综合关联度}

\begin{equation}
    \gamma_i = \sum_{k=1}^{m} w_k \,\xi_i(k)
    \label{eq:gra-grade}
\end{equation}

$\gamma_i$ 越大说明第 $i$ 个对象与最优参考序列越相似，综合评价越优。
"""

# 别名
GreyRelationalAnalysis = GRA