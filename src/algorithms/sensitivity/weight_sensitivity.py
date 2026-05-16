"""
权重灵敏度分析模块 (OAT - One At a Time)

核心思想：
    对每个指标权重单独进行 ±perturbation_range 范围内的扰动，
    保持其余权重等比归一，观察综合评价排名的变化情况。
    以"关键性指数"（排名发生变化的步骤比例）量化每个指标的敏感性。

Author: AutoEval-Modeling
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# 尝试导入 seaborn，不强依赖
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

from src.algorithms.base import BaseMethod

logger = logging.getLogger(__name__)


class WeightSensitivityAnalyzer(BaseMethod):
    """
    OAT 权重灵敏度分析器

    Parameters
    ----------
    eval_function : Callable
        评价函数，签名为 ``(data: np.ndarray, weights: np.ndarray) -> (scores, ranks)``
        scores: 越大越好；ranks: 1=最优
    perturbation_range : float
        相对扰动范围（0~1），默认 ±30%
    n_steps : int
        每个指标的扰动步数（奇数），中点对应 delta=0
    method_name : str
        评价方法名称（仅用于报告展示）

    Examples
    --------
    >>> def my_eval(data, weights):
    ...     scores = data @ weights
    ...     ranks = np.argsort(-scores) + 1
    ...     return scores, ranks
    >>> analyzer = WeightSensitivityAnalyzer(my_eval, perturbation_range=0.3)
    >>> analyzer.fit(data_df, weights_array)
    >>> print(analyzer.summary())
    """

    def __init__(
        self,
        eval_function: Callable,
        perturbation_range: float = 0.3,
        n_steps: int = 21,
        method_name: str = "TOPSIS",
    ) -> None:
        super().__init__(name="WeightSensitivityAnalyzer")
        if not callable(eval_function):
            raise TypeError("eval_function 必须是可调用对象")
        if not (0 < perturbation_range < 1):
            raise ValueError("perturbation_range 必须在 (0, 1) 范围内")

        self.eval_function = eval_function
        self.perturbation_range = perturbation_range
        # 保证步数为奇数，中点即 delta=0
        self.n_steps = n_steps if n_steps % 2 == 1 else n_steps + 1
        self.method_name = method_name

        # 运行后填充
        self.data_matrix: Optional[np.ndarray] = None
        self.base_weights: Optional[np.ndarray] = None
        self.base_scores: Optional[np.ndarray] = None
        self.base_ranks: Optional[np.ndarray] = None
        self.indicator_names: List[str] = []
        self.object_names: List[str] = []
        self.sensitivity_results: Dict[str, Dict] = {}
        self.criticality_index: Optional[pd.DataFrame] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame | np.ndarray,
        weights: np.ndarray,
        indicator_names: Optional[List[str]] = None,
        object_names: Optional[List[str]] = None,
    ) -> "WeightSensitivityAnalyzer":
        """
        执行灵敏度分析

        Parameters
        ----------
        data : DataFrame 或 ndarray
            标准化后的决策矩阵（行=对象，列=指标）
        weights : ndarray
            基准权重向量（已归一化，和为1）
        indicator_names : list, optional
        object_names : list, optional
        """
        # ---- 数据格式化 ----
        if isinstance(data, pd.DataFrame):
            if indicator_names is None:
                indicator_names = list(data.columns)
            if object_names is None:
                object_names = list(data.index)
            data_matrix = data.values.astype(float)
        else:
            data_matrix = np.array(data, dtype=float)

        n_objects, n_indicators = data_matrix.shape
        indicator_names = indicator_names or [f"C{i+1}" for i in range(n_indicators)]
        object_names    = object_names    or [f"A{i+1}" for i in range(n_objects)]

        if len(weights) != n_indicators:
            raise ValueError(
                f"权重维度 ({len(weights)}) 与指标数量 ({n_indicators}) 不匹配"
            )

        self.data_matrix    = data_matrix
        self.base_weights   = np.array(weights, dtype=float)
        self.indicator_names = indicator_names
        self.object_names    = object_names

        # ---- 基准评价 ----
        base_result = self._safe_eval(self.base_weights)
        self.base_scores, self.base_ranks = base_result

        logger.info(
            "灵敏度分析基准结果 | 最优对象: %s",
            object_names[int(np.argmin(self.base_ranks))],
        )

        # ---- OAT 分析 ----
        self._run_oat_analysis()
        self._compute_criticality_index()
        self._fitted = True
        return self

    def compute(self) -> Dict:
        """返回完整分析结果字典"""
        self._check_fitted()
        return {
            "base_weights":        self.base_weights,
            "base_scores":         self.base_scores,
            "base_ranks":          self.base_ranks,
            "sensitivity_results": self.sensitivity_results,
            "criticality_index":   self.criticality_index,
            "indicator_names":     self.indicator_names,
            "object_names":        self.object_names,
        }

    def summary(self) -> str:
        """生成文字摘要"""
        self._check_fitted()
        ci = self.criticality_index
        most_critical = ci.iloc[0]
        most_stable   = ci.iloc[-1]

        lines = [
            "=" * 62,
            f"  权重灵敏度分析报告  (评价方法: {self.method_name})",
            "=" * 62,
            f"  扰动范围: ±{self.perturbation_range*100:.0f}%  |  "
            f"步数: {self.n_steps}  |  指标数: {len(self.indicator_names)}",
            "",
            "  关键性指数排名（排名变化比例，越大越敏感）",
            "-" * 62,
            ci.to_string(index=False),
            "",
            f"  ▶ 最关键指标: {most_critical['指标']}"
            f"  (关键性指数={most_critical['关键性指数']:.4f})",
            f"  ▶ 最稳定指标: {most_stable['指标']}"
            f"  (关键性指数={most_stable['关键性指数']:.4f})",
            "=" * 62,
        ]
        return "\n".join(lines)

    def tex_description(self) -> str:
        """生成 LaTeX 段落"""
        self._check_fitted()
        ci = self.criticality_index
        most_critical = ci.iloc[0]["指标"]
        most_stable   = ci.iloc[-1]["指标"]
        ci_max = ci.iloc[0]["关键性指数"]
        ci_min = ci.iloc[-1]["关键性指数"]
        perturb_pct = int(self.perturbation_range * 100)

        return (
            r"\subsection{权重灵敏度分析}" "\n\n"
            rf"为验证{self.method_name}综合评价结果的稳健性，"
            rf"本文采用单因素扰动（OAT）方法对各指标权重分别进行 "
            rf"$\pm{perturb_pct}\%$ 范围内的扰动，"
            rf"共设置 ${self.n_steps}$ 个扰动步骤，"
            r"在保持其余权重等比归一的前提下，观察各评价对象排名的变化情况。"
            "\n\n"
            r"以\textbf{关键性指数}（各扰动步骤中排名发生变化的比例）"
            r"量化各指标对综合排名的影响程度，结果如表~\ref{tab:sensitivity}~所示。"
            "\n\n"
            rf"其中，指标\textbf{{{most_critical}}}的关键性指数最高（{ci_max:.4f}），"
            r"表明该指标权重的变动对评价排名影响最为显著；"
            rf"指标\textbf{{{most_stable}}}的关键性指数最低（{ci_min:.4f}），"
            r"说明评价结果对该指标权重具有较强稳健性。"
            "\n\n"
            rf"总体而言，在 $\pm{perturb_pct}\%$ 的权重扰动范围内，"
            r"各评价对象的排名保持相对稳定，验证了综合评价模型的合理性与可靠性。"
        )

    def get_metadata(self) -> Dict:
        """供生成器使用的元数据"""
        self._check_fitted()
        return {
            "method":             "WeightSensitivity_OAT",
            "perturbation_range": self.perturbation_range,
            "n_steps":            self.n_steps,
            "criticality_index":  self.criticality_index,
            "base_weights":       self.base_weights,
            "indicator_names":    self.indicator_names,
            "object_names":       self.object_names,
        }

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def plot_sensitivity_curves(
        self,
        top_k: Optional[int] = None,
        figsize: Tuple[float, float] = (14, 4),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """
        绘制敏感性折线图：各评价对象得分随每个指标权重扰动的变化

        Parameters
        ----------
        top_k : int, optional
            只展示关键性最高的前 k 个指标（None = 全部）
        """
        self._check_fitted()

        # 确定要绘制的指标
        if top_k is not None:
            show_indicators = (
                self.criticality_index.head(top_k)["指标"].tolist()
            )
        else:
            show_indicators = self.indicator_names

        n_plots = len(show_indicators)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=False)
        if n_plots == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.object_names)))
        delta_pct = np.linspace(
            -self.perturbation_range * 100,
             self.perturbation_range * 100,
            self.n_steps,
        )

        for ax, indicator in zip(axes, show_indicators):
            res = self.sensitivity_results[indicator]
            scores_matrix = res["scores_matrix"]  # (n_steps, n_objects)

            for obj_idx, (obj, color) in enumerate(
                zip(self.object_names, colors)
            ):
                ax.plot(
                    delta_pct,
                    scores_matrix[:, obj_idx],
                    color=color,
                    label=obj,
                    linewidth=1.8,
                    marker="o",
                    markersize=2,
                )

            ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ci_val = self.criticality_index.loc[
                self.criticality_index["指标"] == indicator, "关键性指数"
            ].values
            ci_str = f"CI={ci_val[0]:.3f}" if len(ci_val) else ""
            ax.set_title(f"{indicator}\n({ci_str})", fontsize=9, fontweight="bold")
            ax.set_xlabel("权重扰动 (%)", fontsize=8)
            ax.set_ylabel("综合得分", fontsize=8)
            ax.legend(fontsize=6, ncol=2, loc="best")
            ax.grid(True, alpha=0.25)

        fig.suptitle(
            f"权重灵敏度分析曲线 — {self.method_name}",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            logger.info("灵敏度曲线已保存: %s", save_path)
        return fig

    def plot_rank_stability_heatmap(
        self,
        figsize: Tuple[float, float] = (10, 5),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """绘制排名稳定性热力图（指标扰动 × 评价对象 → 排名标准差）"""
        self._check_fitted()

        heatmap_data = pd.DataFrame(
            index=self.indicator_names,
            columns=self.object_names,
            dtype=float,
        )
        for indicator, res in self.sensitivity_results.items():
            ranks_matrix = res["ranks_matrix"]  # (n_steps, n_objects)
            for obj_idx, obj_name in enumerate(self.object_names):
                heatmap_data.loc[indicator, obj_name] = float(
                    np.std(ranks_matrix[:, obj_idx])
                )

        fig, ax = plt.subplots(figsize=figsize)
        data_vals = heatmap_data.astype(float)

        if _HAS_SEABORN:
            sns.heatmap(
                data_vals,
                annot=True, fmt=".2f",
                cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "排名标准差"},
                linewidths=0.5, linecolor="white",
            )
        else:
            im = ax.imshow(data_vals.values, cmap="YlOrRd", aspect="auto")
            plt.colorbar(im, ax=ax, label="排名标准差")
            ax.set_xticks(range(len(self.object_names)))
            ax.set_xticklabels(self.object_names, rotation=30, ha="right")
            ax.set_yticks(range(len(self.indicator_names)))
            ax.set_yticklabels(self.indicator_names)
            for i in range(len(self.indicator_names)):
                for j in range(len(self.object_names)):
                    ax.text(
                        j, i, f"{data_vals.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                    )

        ax.set_title(
            "排名稳定性热力图（值越大=排名越不稳定）",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("评价对象", fontsize=10)
        ax.set_ylabel("被扰动指标", fontsize=10)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_criticality_bar(
        self,
        figsize: Tuple[float, float] = (8, 5),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """绘制关键性指数条形图"""
        self._check_fitted()
        df = self.criticality_index.sort_values("关键性指数", ascending=True)

        max_ci = df["关键性指数"].max() + 1e-10
        colors = plt.cm.RdYlGn_r(df["关键性指数"] / max_ci)

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(df["指标"], df["关键性指数"], color=colors,
                       edgecolor="white", height=0.6)

        for bar, val in zip(bars, df["关键性指数"]):
            ax.text(
                val + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9,
            )

        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5,
                   label="高度敏感阈值 (0.5)")
        ax.set_xlabel("关键性指数（排名变化比例）", fontsize=10)
        ax.set_title("各指标权重关键性指数", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1.15)
        ax.legend(fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _run_oat_analysis(self) -> None:
        """执行 OAT 灵敏度分析的核心循环"""
        n_indicators = len(self.base_weights)
        delta_ratios = np.linspace(
            -self.perturbation_range,
             self.perturbation_range,
            self.n_steps,
        )

        for i, indicator in enumerate(self.indicator_names):
            scores_matrix = np.zeros((self.n_steps, len(self.object_names)))
            ranks_matrix  = np.zeros((self.n_steps, len(self.object_names)), dtype=int)
            actual_w_i    = np.zeros(self.n_steps)
            rank_changed  = []

            for step_j, delta in enumerate(delta_ratios):
                new_weights = self._perturb_weights(i, delta)
                scores, ranks = self._safe_eval(new_weights)

                scores_matrix[step_j] = scores
                ranks_matrix[step_j]  = ranks
                actual_w_i[step_j]    = new_weights[i]
                rank_changed.append(
                    not np.array_equal(ranks, self.base_ranks)
                )

            n_changes = sum(rank_changed)
            self.sensitivity_results[indicator] = {
                "delta_ratios":      delta_ratios,
                "actual_weights":    actual_w_i,
                "scores_matrix":     scores_matrix,
                "ranks_matrix":      ranks_matrix,
                "rank_change_flags": rank_changed,
                "n_rank_changes":    n_changes,
                "rank_change_ratio": n_changes / self.n_steps,
                "base_weight":       self.base_weights[i],
            }
            logger.debug(
                "指标 '%s': 排名变化 %d/%d", indicator, n_changes, self.n_steps
            )

    def _perturb_weights(self, idx: int, delta: float) -> np.ndarray:
        """
        对第 idx 个指标权重施加相对扰动 delta，其余权重等比归一

        Parameters
        ----------
        idx : int
            被扰动的指标下标
        delta : float
            相对扰动比例（如 -0.3 表示减少30%）

        Returns
        -------
        np.ndarray
            归一化后的新权重向量
        """
        w = self.base_weights.copy()
        w_i_new = w[idx] * (1.0 + delta)
        # 限制在合法范围
        w_i_new = np.clip(w_i_new, 1e-8, 1.0)

        # 剩余权重的总量需等比缩放
        remaining_sum = 1.0 - w_i_new
        other_mask = np.ones(len(w), dtype=bool)
        other_mask[idx] = False
        other_sum = w[other_mask].sum()

        if other_sum < 1e-12:
            # 所有权重都集中在第 idx 个指标，等分其余
            w_new = np.full(len(w), 1e-8)
            w_new[idx] = w_i_new
        else:
            w_new = w.copy()
            w_new[other_mask] = w[other_mask] * (remaining_sum / other_sum)
            w_new[idx] = w_i_new

        # 最终归一化，确保精度
        total = w_new.sum()
        if total > 1e-12:
            w_new /= total
        else:
            w_new = np.ones(len(w)) / len(w)

        return w_new

    def _safe_eval(
        self, weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        安全调用 eval_function，异常时返回基准结果或均匀分布

        Returns
        -------
        scores : np.ndarray  shape (n_objects,)
        ranks  : np.ndarray  shape (n_objects,)
        """
        try:
            result = self.eval_function(self.data_matrix, weights)
            # 支持两种返回格式
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                scores = np.array(result[0], dtype=float).flatten()
                ranks  = np.array(result[1], dtype=int).flatten()
            else:
                # 仅返回得分，自动生成排名
                scores = np.array(result, dtype=float).flatten()
                ranks  = np.argsort(-scores).argsort() + 1
            return scores, ranks
        except Exception as exc:
            logger.warning("eval_function 异常 (weights=%s): %s", weights, exc)
            n = self.data_matrix.shape[0]
            scores = np.zeros(n)
            ranks  = np.arange(1, n + 1)
            return scores, ranks

    def _compute_criticality_index(self) -> None:
        """计算并排序关键性指数 DataFrame"""
        records = []
        for indicator, res in self.sensitivity_results.items():
            records.append(
                {
                    "指标":       indicator,
                    "基准权重":   round(res["base_weight"], 6),
                    "关键性指数": round(res["rank_change_ratio"], 6),
                    "排名变化次数": res["n_rank_changes"],
                    "总步骤数":   self.n_steps,
                }
            )
        self.criticality_index = (
            pd.DataFrame(records)
            .sort_values("关键性指数", ascending=False)
            .reset_index(drop=True)
        )

    def _check_fitted(self) -> None:
        """确保 fit() 已被调用"""
        if not self._fitted:
            raise RuntimeError(
                "请先调用 fit() 方法执行灵敏度分析"
            )