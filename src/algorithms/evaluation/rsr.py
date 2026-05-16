"""
rsr.py
RSR — Rank Sum Ratio（秩和比法）

Reference:
    田凤调 (1993). 秩和比综合评价法. 中国卫生统计, 10(5), 1-4.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..base import BaseMethod

logger = logging.getLogger(__name__)

_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


class RSR(BaseMethod):
    """
    秩和比法（Rank Sum Ratio）。

    将各指标转化为秩，计算加权秩和比（RSR），
    并通过 Probit 回归对评价对象进行分级排序。

    Parameters
    ----------
    n_grades : int, default 4
        分级数量（等级数），对应 Probit 分级边界数量。
    weights : array-like, optional
        指标权重，默认等权。
    indicator_types : list of str, optional
        'positive' 或 'negative'，默认全正向。
    """

    # 等级标签
    _GRADE_LABELS = {
        2: ["较差", "较优"],
        3: ["差", "中", "优"],
        4: ["差", "中下", "中上", "优"],
        5: ["差", "较差", "中", "较优", "优"],
    }

    def __init__(
        self,
        n_grades: int = 4,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        indicator_types: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name="RSR")
        if n_grades not in range(2, 6):
            raise ValueError("n_grades 须在 2~5 之间。")
        self.n_grades = n_grades
        self.weights: Optional[np.ndarray] = (
            np.asarray(weights, dtype=float) if weights is not None else None
        )
        self.indicator_types = indicator_types

        self._data: Optional[pd.DataFrame] = None
        self._alternatives: Optional[List] = None
        self._criteria: Optional[List] = None
        self._rank_matrix: Optional[np.ndarray] = None
        self._rsr: Optional[np.ndarray] = None
        self._probit: Optional[np.ndarray] = None
        self._regression_result = None  # scipy linregress result
        self._grade_thresholds: Optional[np.ndarray] = None
        self._grades: Optional[List[str]] = None

    def fit(
        self,
        data: pd.DataFrame,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        indicator_types: Optional[List[str]] = None,
    ) -> "RSR":
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
        执行 RSR 计算（含 Probit 回归分级）。

        Returns
        -------
        pd.DataFrame
            列：``RSR`` / ``Probit`` / ``RSR_fitted`` / ``Grade`` / ``Rank``。
        """
        if self._data is None:
            raise RuntimeError("请先调用 fit()。")

        X = self._data.values.astype(float)
        n, m = X.shape

        # ① 转化为秩（负向指标秩从小到大赋大秩 → 反向排名）
        self._rank_matrix = np.zeros_like(X)
        for j in range(m):
            col = pd.Series(X[:, j])
            if self.indicator_types[j] == "positive":
                self._rank_matrix[:, j] = col.rank(method="average").values
            else:
                self._rank_matrix[:, j] = col.rank(method="average", ascending=False).values

        # ② 加权秩和比 RSR_i = Σ w_j * R_ij / n
        self._rsr = (self._rank_matrix * self.weights).sum(axis=1) / n

        # ③ 频率 p_i 与 Probit 变换
        sorted_idx = np.argsort(self._rsr)
        p = np.zeros(n)
        for rank_pos, orig_idx in enumerate(sorted_idx):
            p[orig_idx] = (rank_pos + 1) / n
        # Blom 修正：避免 p=0 或 p=1
        p_corrected = (p * n - 0.375) / (n + 0.25)
        p_corrected = np.clip(p_corrected, 1e-6, 1 - 1e-6)
        self._probit = stats.norm.ppf(p_corrected) + 5.0  # 转为 Probit 标准值（+5）

        # ④ RSR 对 Probit 的线性回归
        valid = np.isfinite(self._probit) & np.isfinite(self._rsr)
        reg = stats.linregress(self._probit[valid], self._rsr[valid])
        self._regression_result = reg
        rsr_fitted = reg.slope * self._probit + reg.intercept

        # ⑤ 分级（按等分 Probit 区间）
        probit_min, probit_max = self._probit[valid].min(), self._probit[valid].max()
        grade_bounds = np.linspace(probit_min, probit_max, self.n_grades + 1)
        self._grade_thresholds = grade_bounds
        grade_labels = self._GRADE_LABELS.get(
            self.n_grades, [f"Grade{k}" for k in range(1, self.n_grades + 1)]
        )
        self._grades = [
            grade_labels[min(
                np.searchsorted(grade_bounds[1:-1], p_val),
                self.n_grades - 1,
            )]
            for p_val in self._probit
        ]

        ranks = (
            pd.Series(self._rsr, index=self._alternatives)
            .rank(ascending=False, method="min")
            .astype(int)
        )

        self.result = pd.DataFrame(
            {
                "RSR": np.round(self._rsr, 6),
                "Probit": np.round(self._probit, 4),
                "RSR_fitted": np.round(rsr_fitted, 6),
                "Grade": self._grades,
                "Rank": ranks,
            },
            index=self._alternatives,
        ).sort_values("Rank")

        self.metadata.update(
            {
                "n_grades": self.n_grades,
                "regression_slope": float(reg.slope),
                "regression_intercept": float(reg.intercept),
                "regression_r2": float(reg.rvalue ** 2),
                "best": self.result.index[0],
                "worst": self.result.index[-1],
            }
        )
        logger.info(
            "RSR 计算完成 → 最优: %s (RSR=%.4f), 回归 R²=%.4f",
            self.result.index[0], self._rsr.max(),
            self.metadata["regression_r2"],
        )
        return self.result

    def summary(self) -> Dict[str, Any]:
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")
        return {
            "method": "RSR",
            "description": "秩和比法",
            "n_alternatives": len(self._alternatives),
            "n_criteria": len(self._criteria),
            "n_grades": self.n_grades,
            "weights": dict(zip(self._criteria, self.weights.tolist())),
            "rsr_values": self.result["RSR"].to_dict(),
            "probit_values": self.result["Probit"].to_dict(),
            "grades": self.result["Grade"].to_dict(),
            "ranking": self.result["Rank"].to_dict(),
            "best": self.result.index[0],
            "regression": {
                "slope": self.metadata["regression_slope"],
                "intercept": self.metadata["regression_intercept"],
                "r_squared": self.metadata["regression_r2"],
            },
        }

    def plot_probit_regression(
        self,
        figsize: Tuple = (9, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制 RSR-Probit 回归散点图及回归直线。"""
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")

        reg = self._regression_result
        probit_range = np.linspace(self._probit.min() - 0.2,
                                   self._probit.max() + 0.2, 100)
        fitted_line = reg.slope * probit_range + reg.intercept

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(self._probit, self._rsr, s=80, zorder=5,
                       color="#3498db", edgecolors="black", linewidth=0.7,
                       label="观测值")
            ax.plot(probit_range, fitted_line, "r-", lw=2,
                    label=f"回归直线 (R²={reg.rvalue**2:.4f})")

            for i, alt in enumerate(self._alternatives):
                ax.annotate(alt, (self._probit[i], self._rsr[i]),
                            textcoords="offset points", xytext=(6, 3), fontsize=8)

            ax.set_xlabel("Probit 值", fontsize=12)
            ax.set_ylabel("RSR 值", fontsize=12)
            ax.set_title("RSR—Probit 回归分析", fontsize=13, fontweight="bold")
            ax.legend(fontsize=10)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    """
    rsr.py
    RSR — Rank Sum Ratio（秩和比法）

    Reference:
        田凤调 (1993). 秩和比综合评价法. 中国卫生统计, 10(5), 1-4.
    """

    from __future__ import annotations

    import logging
    from typing import Any, Dict, List, Optional, Tuple, Union

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    from ..base import BaseMethod

    logger = logging.getLogger(__name__)

    _STYLE = {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }

    class RSR(BaseMethod):
        """
        秩和比法（Rank Sum Ratio）。

        将各指标转化为秩，计算加权秩和比（RSR），
        并通过 Probit 回归对评价对象进行分级排序。

        Parameters
        ----------
        n_grades : int, default 4
            分级数量（等级数），对应 Probit 分级边界数量。
        weights : array-like, optional
            指标权重，默认等权。
        indicator_types : list of str, optional
            'positive' 或 'negative'，默认全正向。
        """

        # 等级标签
        _GRADE_LABELS = {
            2: ["较差", "较优"],
            3: ["差", "中", "优"],
            4: ["差", "中下", "中上", "优"],
            5: ["差", "较差", "中", "较优", "优"],
        }

        def __init__(
                self,
                n_grades: int = 4,
                weights: Optional[Union[List[float], np.ndarray]] = None,
                indicator_types: Optional[List[str]] = None,
        ) -> None:
            super().__init__(name="RSR")
            if n_grades not in range(2, 6):
                raise ValueError("n_grades 须在 2~5 之间。")
            self.n_grades = n_grades
            self.weights: Optional[np.ndarray] = (
                np.asarray(weights, dtype=float) if weights is not None else None
            )
            self.indicator_types = indicator_types

            self._data: Optional[pd.DataFrame] = None
            self._alternatives: Optional[List] = None
            self._criteria: Optional[List] = None
            self._rank_matrix: Optional[np.ndarray] = None
            self._rsr: Optional[np.ndarray] = None
            self._probit: Optional[np.ndarray] = None
            self._regression_result = None  # scipy linregress result
            self._grade_thresholds: Optional[np.ndarray] = None
            self._grades: Optional[List[str]] = None

        def fit(
                self,
                data: pd.DataFrame,
                weights: Optional[Union[List[float], np.ndarray]] = None,
                indicator_types: Optional[List[str]] = None,
        ) -> "RSR":
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
            执行 RSR 计算（含 Probit 回归分级）。

            Returns
            -------
            pd.DataFrame
                列：``RSR`` / ``Probit`` / ``RSR_fitted`` / ``Grade`` / ``Rank``。
            """
            if self._data is None:
                raise RuntimeError("请先调用 fit()。")

            X = self._data.values.astype(float)
            n, m = X.shape

            # ① 转化为秩（负向指标秩从小到大赋大秩 → 反向排名）
            self._rank_matrix = np.zeros_like(X)
            for j in range(m):
                col = pd.Series(X[:, j])
                if self.indicator_types[j] == "positive":
                    self._rank_matrix[:, j] = col.rank(method="average").values
                else:
                    self._rank_matrix[:, j] = col.rank(method="average", ascending=False).values

            # ② 加权秩和比 RSR_i = Σ w_j * R_ij / n
            self._rsr = (self._rank_matrix * self.weights).sum(axis=1) / n

            # ③ 频率 p_i 与 Probit 变换
            sorted_idx = np.argsort(self._rsr)
            p = np.zeros(n)
            for rank_pos, orig_idx in enumerate(sorted_idx):
                p[orig_idx] = (rank_pos + 1) / n
            # Blom 修正：避免 p=0 或 p=1
            p_corrected = (p * n - 0.375) / (n + 0.25)
            p_corrected = np.clip(p_corrected, 1e-6, 1 - 1e-6)
            self._probit = stats.norm.ppf(p_corrected) + 5.0  # 转为 Probit 标准值（+5）

            # ④ RSR 对 Probit 的线性回归
            valid = np.isfinite(self._probit) & np.isfinite(self._rsr)
            reg = stats.linregress(self._probit[valid], self._rsr[valid])
            self._regression_result = reg
            rsr_fitted = reg.slope * self._probit + reg.intercept

            # ⑤ 分级（按等分 Probit 区间）
            probit_min, probit_max = self._probit[valid].min(), self._probit[valid].max()
            grade_bounds = np.linspace(probit_min, probit_max, self.n_grades + 1)
            self._grade_thresholds = grade_bounds
            grade_labels = self._GRADE_LABELS.get(
                self.n_grades, [f"Grade{k}" for k in range(1, self.n_grades + 1)]
            )
            self._grades = [
                grade_labels[min(
                    np.searchsorted(grade_bounds[1:-1], p_val),
                    self.n_grades - 1,
                )]
                for p_val in self._probit
            ]

            ranks = (
                pd.Series(self._rsr, index=self._alternatives)
                .rank(ascending=False, method="min")
                .astype(int)
            )

            self.result = pd.DataFrame(
                {
                    "RSR": np.round(self._rsr, 6),
                    "Probit": np.round(self._probit, 4),
                    "RSR_fitted": np.round(rsr_fitted, 6),
                    "Grade": self._grades,
                    "Rank": ranks,
                },
                index=self._alternatives,
            ).sort_values("Rank")

            self.metadata.update(
                {
                    "n_grades": self.n_grades,
                    "regression_slope": float(reg.slope),
                    "regression_intercept": float(reg.intercept),
                    "regression_r2": float(reg.rvalue ** 2),
                    "best": self.result.index[0],
                    "worst": self.result.index[-1],
                }
            )
            logger.info(
                "RSR 计算完成 → 最优: %s (RSR=%.4f), 回归 R²=%.4f",
                self.result.index[0], self._rsr.max(),
                self.metadata["regression_r2"],
            )
            return self.result

        def summary(self) -> Dict[str, Any]:
            if self.result is None:
                raise RuntimeError("请先调用 compute()。")
            return {
                "method": "RSR",
                "description": "秩和比法",
                "n_alternatives": len(self._alternatives),
                "n_criteria": len(self._criteria),
                "n_grades": self.n_grades,
                "weights": dict(zip(self._criteria, self.weights.tolist())),
                "rsr_values": self.result["RSR"].to_dict(),
                "probit_values": self.result["Probit"].to_dict(),
                "grades": self.result["Grade"].to_dict(),
                "ranking": self.result["Rank"].to_dict(),
                "best": self.result.index[0],
                "regression": {
                    "slope": self.metadata["regression_slope"],
                    "intercept": self.metadata["regression_intercept"],
                    "r_squared": self.metadata["regression_r2"],
                },
            }

        def plot_probit_regression(
                self,
                figsize: Tuple = (9, 6),
                save_path: Optional[str] = None,
        ) -> plt.Figure:
            """绘制 RSR-Probit 回归散点图及回归直线。"""
            if self.result is None:
                raise RuntimeError("请先调用 compute()。")

            reg = self._regression_result
            probit_range = np.linspace(self._probit.min() - 0.2,
                                       self._probit.max() + 0.2, 100)
            fitted_line = reg.slope * probit_range + reg.intercept

            with plt.rc_context(_STYLE):
                fig, ax = plt.subplots(figsize=figsize)
                ax.scatter(self._probit, self._rsr, s=80, zorder=5,
                           color="#3498db", edgecolors="black", linewidth=0.7,
                           label="观测值")
                ax.plot(probit_range, fitted_line, "r-", lw=2,
                        label=f"回归直线 (R²={reg.rvalue ** 2:.4f})")

                for i, alt in enumerate(self._alternatives):
                    ax.annotate(alt, (self._probit[i], self._rsr[i]),
                                textcoords="offset points", xytext=(6, 3), fontsize=8)

                ax.set_xlabel("Probit 值", fontsize=12)
                ax.set_ylabel("RSR 值", fontsize=12)
                ax.set_title("RSR—Probit 回归分析", fontsize=13, fontweight="bold")
                ax.legend(fontsize=10)
                plt.tight_layout()
                if save_path:
                    fig.savefig(save_path, dpi=150, bbox_inches="tight")
            return fig


    def tex_description(self) -> str:
        return r"""
    \subsection{基于秩和比法的综合评价与分级模型}

    \subsubsection{模型原理}

    秩和比法（Rank Sum Ratio, RSR）由田凤调（1993）提出，
    将原始数据转化为秩次，通过计算加权秩和比对评价对象进行综合排序，
    并借助 Probit 回归实现客观分级。

    \subsubsection{计算步骤}

    \noindent\textbf{步骤1\quad 编秩}

    对每列指标独立编秩：效益型指标按升序编秩，
    成本型指标按降序编秩（即大值赋小秩）。

    \noindent\textbf{步骤2\quad 计算加权秩和比}

    \begin{equation}
        \mathrm{RSR}_i = \frac{1}{n}\sum_{j=1}^{m} w_j R_{ij},
        \quad 0 < \mathrm{RSR}_i \leq 1
        \label{eq:rsr-value}
    \end{equation}
    其中 $R_{ij}$ 为第 $i$ 个对象第 $j$ 个指标的秩，
    $n$ 为评价对象总数，$\sum_j w_j = 1$。

    \noindent\textbf{步骤3\quad Probit 变换}

    将 RSR 的累积频率 $p_i$（Blom修正）转化为正态分布分位数：
    \begin{equation}
        \text{Probit}_i = \Phi^{-1}(p_i) + 5
        \label{eq:rsr-probit}
    \end{equation}

    \noindent\textbf{步骤4\quad 线性回归建立分级标准}

    以 Probit 为自变量对 RSR 作线性回归：
    \begin{equation}
        \hat{\mathrm{RSR}}_i = a + b \cdot \text{Probit}_i
        \label{eq:rsr-regression}
    \end{equation}
    取各等级分界点对应的 Probit 阈值代入回归方程，
    得到 RSR 分级边界，实现对评价对象的客观定级。
    """