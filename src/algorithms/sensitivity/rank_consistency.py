"""
多方法排名一致性检验模块

功能：
    1. Kendall W 协调系数  —— 检验多评价者/多方法的整体一致程度
    2. Kendall τ 相关系数  —— 两两方法间排名相关性
    3. Spearman ρ 秩相关   —— 两两方法间得分相关性
    4. Borda 计数综合排名  —— 融合多方法的稳健排名
    5. 可视化：相关矩阵热力图 + Borda 排名柱状图

Author: AutoEval-Modeling
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

from src.algorithms.base import BaseMethod

logger = logging.getLogger(__name__)


class RankConsistencyChecker(BaseMethod):
    """
    多方法排名一致性检验器

    Parameters
    ----------
    method_names : list of str
        各评价方法的名称标签
    significance_level : float
        显著性水平（默认 0.05）

    Examples
    --------
    >>> checker = RankConsistencyChecker(["TOPSIS", "VIKOR", "GRA"])
    >>> ranks_dict = {
    ...     "TOPSIS": np.array([1, 3, 2, 5, 4]),
    ...     "VIKOR":  np.array([1, 2, 3, 5, 4]),
    ...     "GRA":    np.array([2, 3, 1, 4, 5]),
    ... }
    >>> checker.fit(ranks_dict)
    >>> print(checker.summary())
    """

    def __init__(
        self,
        method_names: Optional[List[str]] = None,
        significance_level: float = 0.05,
    ) -> None:
        super().__init__(name="RankConsistencyChecker")
        self.method_names      = method_names or []
        self.significance_level = significance_level

        # 运行后填充
        self.ranks_df:           Optional[pd.DataFrame] = None
        self.kendall_w:          Optional[float] = None
        self.kendall_w_pvalue:   Optional[float] = None
        self.kendall_tau_matrix: Optional[pd.DataFrame] = None
        self.spearman_matrix:    Optional[pd.DataFrame] = None
        self.borda_scores:       Optional[pd.Series] = None
        self.borda_ranks:        Optional[pd.Series] = None
        self.pairwise_results:   Optional[pd.DataFrame] = None
        self.object_names:       List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def fit(
        self,
        ranks: Union[Dict[str, np.ndarray], pd.DataFrame],
        object_names: Optional[List[str]] = None,
    ) -> "RankConsistencyChecker":
        """
        执行一致性检验

        Parameters
        ----------
        ranks : dict or DataFrame
            键为方法名，值为排名数组（1=最优）；
            或 DataFrame（行=评价对象，列=方法名）
        object_names : list, optional
            评价对象名称
        """
        # ---- 统一转为 DataFrame（行=对象，列=方法）----
        if isinstance(ranks, dict):
            self.ranks_df = pd.DataFrame(ranks)
            if self.method_names:
                # 只保留指定方法列
                cols = [m for m in self.method_names if m in self.ranks_df.columns]
                self.ranks_df = self.ranks_df[cols]
        else:
            self.ranks_df = ranks.copy()

        n_objects, n_methods = self.ranks_df.shape
        self.object_names = (
            object_names
            or list(self.ranks_df.index)
            or [f"A{i+1}" for i in range(n_objects)]
        )
        self.ranks_df.index = self.object_names

        if n_methods < 2:
            raise ValueError("至少需要 2 种评价方法才能进行一致性检验")

        logger.info(
            "一致性检验: %d 个对象 × %d 种方法", n_objects, n_methods
        )

        # ---- 依次执行各项检验 ----
        self._compute_kendall_w()
        self._compute_pairwise_correlations()
        self._compute_borda_ranking()
        self._fitted = True
        return self

    def compute(self) -> Dict:
        """返回完整检验结果"""
        self._check_fitted()
        return {
            "ranks_df":          self.ranks_df,
            "kendall_w":         self.kendall_w,
            "kendall_w_pvalue":  self.kendall_w_pvalue,
            "kendall_tau_matrix": self.kendall_tau_matrix,
            "spearman_matrix":   self.spearman_matrix,
            "borda_scores":      self.borda_scores,
            "borda_ranks":       self.borda_ranks,
            "pairwise_results":  self.pairwise_results,
        }

    def summary(self) -> str:
        """生成文字摘要"""
        self._check_fitted()
        n_methods = self.ranks_df.shape[1]
        methods   = list(self.ranks_df.columns)
        w_str     = f"{self.kendall_w:.4f}"
        p_str     = f"{self.kendall_w_pvalue:.4f}"
        is_sig    = self.kendall_w_pvalue < self.significance_level
        sig_str   = "✓ 显著（各方法结果高度一致）" if is_sig else "✗ 不显著（方法间存在分歧）"

        lines = [
            "=" * 62,
            f"  多方法排名一致性检验报告  (α={self.significance_level})",
            "=" * 62,
            f"  参与方法: {', '.join(methods)}",
            "",
            "  ── Kendall W 协调系数 ──────────────────────────────",
            f"  W = {w_str}  (p = {p_str})  → {sig_str}",
            f"  参考: W>0.7 高度一致, 0.5~0.7 一致, <0.5 分歧较大",
            "",
            "  ── Spearman ρ 两两相关矩阵 ─────────────────────────",
            self.spearman_matrix.to_string(),
            "",
            "  ── Kendall τ 两两相关矩阵 ──────────────────────────",
            self.kendall_tau_matrix.to_string(),
            "",
            "  ── Borda 计数综合排名 ───────────────────────────────",
            pd.DataFrame({
                "Borda得分": self.borda_scores,
                "综合排名":  self.borda_ranks,
            }).to_string(),
            "=" * 62,
        ]
        return "\n".join(lines)

    def tex_description(self) -> str:
        """生成 LaTeX 段落"""
        self._check_fitted()
        methods  = list(self.ranks_df.columns)
        methods_str = "、".join(methods)
        w_val    = self.kendall_w
        p_val    = self.kendall_w_pvalue
        is_sig   = p_val < self.significance_level
        sig_str  = "显著" if is_sig else "不显著"
        alpha    = self.significance_level

        return (
            r"\subsection{多方法排名一致性检验}" "\n\n"
            rf"为验证综合评价结果的稳健性，本文对"
            rf"{methods_str}等 {len(methods)} 种评价方法的排名结果进行一致性检验。"
            "\n\n"
            r"\textbf{Kendall W 协调系数}检验结果显示，"
            rf"$W = {w_val:.4f}$，$p = {p_val:.4f}$，"
            rf"在显著性水平 $\alpha = {alpha}$ 下{sig_str}，"
            r"说明各评价方法所给出的排名之间"
            + ("具有较高的一致性。" if is_sig else "存在一定差异，建议结合 Borda 计数法取综合排名。")
            + "\n\n"
            r"各方法间的 Spearman 秩相关系数矩阵如表~\ref{tab:spearman}~所示，"
            r"Borda 计数法综合排名如表~\ref{tab:borda}~所示。"
            r"上述结果表明，本文所建立的综合评价模型具有良好的方法稳健性。"
        )

    def get_metadata(self) -> Dict:
        """供生成器使用的元数据"""
        self._check_fitted()
        return {
            "method":            "RankConsistency",
            "kendall_w":         self.kendall_w,
            "kendall_w_pvalue":  self.kendall_w_pvalue,
            "spearman_matrix":   self.spearman_matrix,
            "kendall_tau_matrix": self.kendall_tau_matrix,
            "borda_scores":      self.borda_scores,
            "borda_ranks":       self.borda_ranks,
            "pairwise_results":  self.pairwise_results,
            "object_names":      self.object_names,
            "method_names":      list(self.ranks_df.columns),
        }

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def plot_correlation_heatmap(
        self,
        corr_type: str = "spearman",
        figsize: Tuple[float, float] = (8, 6),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """
        绘制两两相关系数热力图

        Parameters
        ----------
        corr_type : "spearman" 或 "kendall"
        """
        self._check_fitted()
        mat = (
            self.spearman_matrix
            if corr_type == "spearman"
            else self.kendall_tau_matrix
        )
        title = (
            "Spearman 秩相关系数矩阵"
            if corr_type == "spearman"
            else "Kendall τ 相关系数矩阵"
        )

        fig, ax = plt.subplots(figsize=figsize)
        data_vals = mat.astype(float)

        if _HAS_SEABORN:
            mask = np.triu(np.ones_like(data_vals, dtype=bool), k=1)
            sns.heatmap(
                data_vals,
                annot=True, fmt=".3f",
                cmap="coolwarm", vmin=-1, vmax=1,
                ax=ax, mask=mask,
                square=True, linewidths=0.5,
                cbar_kws={"label": "相关系数"},
            )
        else:
            im = ax.imshow(
                data_vals.values, cmap="coolwarm",
                vmin=-1, vmax=1, aspect="auto",
            )
            plt.colorbar(im, ax=ax, label="相关系数")
            n = len(mat.columns)
            ax.set_xticks(range(n))
            ax.set_xticklabels(mat.columns, rotation=30, ha="right")
            ax.set_yticks(range(n))
            ax.set_yticklabels(mat.index)
            for i in range(n):
                for j in range(n):
                    ax.text(
                        j, i, f"{data_vals.values[i, j]:.3f}",
                        ha="center", va="center", fontsize=9,
                    )

        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_borda_ranking(
        self,
        figsize: Tuple[float, float] = (10, 5),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """绘制 Borda 综合排名条形图（与各方法排名对比）"""
        self._check_fitted()
        methods = list(self.ranks_df.columns)
        n_methods = len(methods)
        n_objects = len(self.object_names)

        # 排序依据 Borda 排名
        order = self.borda_ranks.sort_values().index

        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(n_objects)
        width = 0.8 / (n_methods + 1)
        colors = plt.cm.Set2(np.linspace(0, 1, n_methods + 1))

        for i, method in enumerate(methods):
            ranks_ordered = self.ranks_df.loc[order, method].values
            offset = (i - n_methods / 2) * width
            ax.bar(
                x + offset, ranks_ordered, width,
                label=method, color=colors[i],
                alpha=0.85, edgecolor="white",
            )

        # Borda 排名
        borda_ordered = self.borda_ranks.loc[order].values
        offset = (n_methods - n_methods / 2) * width
        ax.bar(
            x + offset, borda_ordered, width,
            label="Borda综合", color=colors[n_methods],
            alpha=1.0, edgecolor="black", linewidth=1.2,
            hatch="//",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylabel("排名（1=最优）", fontsize=10)
        ax.set_yticks(range(1, n_objects + 1))
        ax.set_title("多方法排名对比与 Borda 综合排名", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.invert_yaxis()  # 排名 1 在上方
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _compute_kendall_w(self) -> None:
        """计算 Kendall W 协调系数及其显著性"""
        ranks_array = self.ranks_df.values.T  # (n_methods, n_objects)
        n_raters, n_objects = ranks_array.shape

        # 计算公式：W = 12S / [k²(n³-n) - 12k*T]
        # S = Σ(Ri - R̄)²，Ri 为各列秩和
        rank_sums = ranks_array.sum(axis=0)       # (n_objects,)
        mean_rank_sum = rank_sums.mean()
        S = np.sum((rank_sums - mean_rank_sum) ** 2)

        k = n_raters
        n = n_objects

        # 修正项 T（处理并列秩）
        T = 0.0
        for row in ranks_array:
            _, counts = np.unique(row, return_counts=True)
            T += np.sum(counts ** 3 - counts) / 12.0

        denom = k ** 2 * (n ** 3 - n) - 12 * k * T
        if abs(denom) < 1e-12:
            self.kendall_w = 0.0
            self.kendall_w_pvalue = 1.0
            logger.warning("Kendall W 分母为零，可能存在完全一致的排名")
            return

        W = 12 * S / denom
        self.kendall_w = float(np.clip(W, 0.0, 1.0))

        # 卡方近似检验统计量
        chi2_stat = k * (n - 1) * self.kendall_w
        df = n - 1
        self.kendall_w_pvalue = float(1 - stats.chi2.cdf(chi2_stat, df))
        logger.debug(
            "Kendall W=%.4f, p=%.4f", self.kendall_w, self.kendall_w_pvalue
        )

    def _compute_pairwise_correlations(self) -> None:
        """计算两两方法间的 Spearman ρ 和 Kendall τ"""
        methods = list(self.ranks_df.columns)
        n = len(methods)
        spearman_mat = pd.DataFrame(
            np.eye(n), index=methods, columns=methods
        )
        kendall_mat = pd.DataFrame(
            np.eye(n), index=methods, columns=methods
        )
        pairwise_records = []

        for m1, m2 in combinations(methods, 2):
            r1 = self.ranks_df[m1].values
            r2 = self.ranks_df[m2].values

            sp_corr, sp_p = stats.spearmanr(r1, r2)
            kt_corr, kt_p = stats.kendalltau(r1, r2)

            spearman_mat.loc[m1, m2] = round(sp_corr, 4)
            spearman_mat.loc[m2, m1] = round(sp_corr, 4)
            kendall_mat.loc[m1, m2]  = round(kt_corr, 4)
            kendall_mat.loc[m2, m1]  = round(kt_corr, 4)

            pairwise_records.append({
                "方法A":        m1,
                "方法B":        m2,
                "Spearman_ρ":  round(sp_corr, 4),
                "Spearman_p":  round(sp_p, 4),
                "Kendall_τ":   round(kt_corr, 4),
                "Kendall_p":   round(kt_p, 4),
                "一致性":       "✓" if sp_p < self.significance_level else "✗",
            })

        self.spearman_matrix  = spearman_mat
        self.kendall_tau_matrix = kendall_mat
        self.pairwise_results = pd.DataFrame(pairwise_records)

    def _compute_borda_ranking(self) -> None:
        """
        Borda 计数法：每个对象在每种方法中的得分 = n - rank + 1
        累加所有方法的得分，降序排列即综合排名
        """
        n_objects = self.ranks_df.shape[0]
        # Borda 得分：排第1名得 n 分，排最后得 1 分
        borda_df = n_objects + 1 - self.ranks_df
        self.borda_scores = borda_df.sum(axis=1).rename("Borda得分")
        self.borda_ranks  = (
            self.borda_scores
            .rank(ascending=False, method="min")
            .astype(int)
            .rename("综合排名")
        )
        logger.debug(
            "Borda 最优对象: %s",
            self.borda_ranks.idxmin()
        )

    def _check_fitted(self) -> None:
        """确保 fit() 已被调用"""
        if not self._fitted:
            raise RuntimeError("请先调用 fit() 方法执行一致性检验")