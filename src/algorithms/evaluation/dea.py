"""
dea.py
DEA — Data Envelopment Analysis
数据包络分析（CCR 模型 + BCC 模型）

Reference:
    Charnes, A., Cooper, W. W., & Rhodes, E. (1978).
    Measuring the efficiency of decision making units.
    European Journal of Operational Research, 2(6), 429-444.

    Banker, R. D., Charnes, A., & Cooper, W. W. (1984).
    Some models for estimating technical and scale inefficiencies in DEA.
    Management Science, 30(9), 1078-1092.
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

try:
    from scipy.optimize import linprog
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    warnings.warn("scipy 未安装，DEA 模型不可用。", ImportWarning)

from ..base import BaseMethod

logger = logging.getLogger(__name__)

_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


# ──────────────────────────────────────────────────────────────────────────────
# 内部 LP 求解器
# ──────────────────────────────────────────────────────────────────────────────

def _solve_ccr_input(
    x0: np.ndarray,
    y0: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    求解 CCR（CRS）输入导向 DEA 线性规划。

    对 DMU_0 求解：
        min  θ
        s.t. X λ ≤ θ x_0
             Y λ ≥ y_0
             λ ≥ 0

    Parameters
    ----------
    x0 : (m,)   目标 DMU 的输入向量
    y0 : (s,)   目标 DMU 的输出向量
    X  : (n, m) 全部 DMU 的输入矩阵
    Y  : (n, s) 全部 DMU 的输出矩阵

    Returns
    -------
    theta : float       效率值
    lambda_ : (n,)      参考集权重
    slacks : (m+s,)     松弛变量 [s_in, s_out]
    """
    n, m = X.shape
    _, s = Y.shape

    # 决策变量：[θ, λ_1, ..., λ_n, s_in_1,...,s_in_m, s_out_1,...,s_out_s]
    # 总变量数：1 + n + m + s
    n_vars = 1 + n + m + s

    # 目标函数：min θ（即最小化第 0 个变量）
    c = np.zeros(n_vars)
    c[0] = 1.0

    # 不等式约束 A_ub x ≤ b_ub
    # 约束1：X λ - θ x_0 + s_in = 0  →  X λ - θ x_0 ≤ 0  (slack 松弛)
    # 约束2：-Y λ + s_out = -y_0      →  -Y λ ≤ -y_0
    # 使用等式约束更精准：A_eq x = b_eq

    # 等式约束：
    # [−x0 | X.T | I_m | 0  ] [θ; λ; s_in; s_out] = 0    (m 行，输入平衡)
    # [  0  | Y.T | 0   | I_s] [θ; λ; s_in; s_out] = y0   (s 行，输出平衡)
    A_eq = np.zeros((m + s, n_vars))
    # 输入：-θ x_0 + X λ + s_in = 0
    A_eq[:m, 0] = -x0                          # θ 系数
    A_eq[:m, 1:1+n] = X.T                      # λ 系数
    A_eq[:m, 1+n:1+n+m] = np.eye(m)            # s_in 系数

    # 输出：Y λ - s_out = y_0
    A_eq[m:, 1:1+n] = Y.T                      # λ 系数
    A_eq[m:, 1+n+m:] = -np.eye(s)              # s_out 系数（负向松弛）

    b_eq = np.concatenate([np.zeros(m), y0])

    # 变量界限：θ 无上界，λ ≥ 0，松弛变量 ≥ 0
    bounds = [(None, None)] + [(0, None)] * (n + m + s)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = linprog(
            c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
            method="highs",
            options={"disp": False, "presolve": True},
        )

    if res.success:
        theta = float(res.x[0])
        lambda_ = res.x[1:1+n]
        slacks = res.x[1+n:]
        return max(0.0, min(1.0, theta)), lambda_, slacks
    else:
        logger.debug("CCR LP 求解失败: %s", res.message)
        return 1.0, np.zeros(n), np.zeros(m + s)


def _solve_bcc_input(
    x0: np.ndarray,
    y0: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    求解 BCC（VRS）输入导向 DEA 线性规划。
    在 CCR 基础上增加凸性约束：sum(λ) = 1。
    """
    n, m = X.shape
    _, s = Y.shape
    n_vars = 1 + n + m + s

    c = np.zeros(n_vars)
    c[0] = 1.0

    # 等式约束（同 CCR）+ 凸性约束
    A_eq = np.zeros((m + s + 1, n_vars))
    A_eq[:m, 0] = -x0
    A_eq[:m, 1:1+n] = X.T
    A_eq[:m, 1+n:1+n+m] = np.eye(m)
    A_eq[m:m+s, 1:1+n] = Y.T
    A_eq[m:m+s, 1+n+m:] = -np.eye(s)
    # 凸性约束：sum λ_j = 1
    A_eq[m+s, 1:1+n] = 1.0

    b_eq = np.concatenate([np.zeros(m), y0, [1.0]])

    bounds = [(None, None)] + [(0, None)] * (n + m + s)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = linprog(
            c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
            method="highs",
            options={"disp": False, "presolve": True},
        )

    if res.success:
        theta = float(res.x[0])
        lambda_ = res.x[1:1+n]
        slacks = res.x[1+n:]
        return max(0.0, min(1.0, theta)), lambda_, slacks
    else:
        logger.debug("BCC LP 求解失败: %s", res.message)
        return 1.0, np.zeros(n), np.zeros(m + s)


# ──────────────────────────────────────────────────────────────────────────────
# 主类
# ──────────────────────────────────────────────────────────────────────────────

class DEA(BaseMethod):
    """
    数据包络分析（Data Envelopment Analysis）。

    同时计算 CCR（规模报酬不变）和 BCC（规模报酬可变）模型效率值，
    并导出规模效率与规模报酬类型。

    Parameters
    ----------
    model : str, default 'both'
        运行的模型：``'ccr'`` / ``'bcc'`` / ``'both'``。
    orientation : str, default 'input'
        导向类型：``'input'``（输入导向，目前仅支持此选项）。
    input_cols : list of str, optional
        输入指标列名，若为 None 则自动从数据中推断（需配合 ``output_cols``）。
    output_cols : list of str, optional
        输出指标列名。

    Notes
    -----
    - DEA 要求所有投入/产出值严格为正，否则内部加微小扰动。
    - 建议样本量 ≥ max(3×m, 3×s)（m 输入数，s 输出数）。
    """

    def __init__(
        self,
        model: str = "both",
        orientation: str = "input",
        input_cols: Optional[List[str]] = None,
        output_cols: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name="DEA")
        if not _SCIPY_OK:
            raise ImportError("DEA 模型需要 scipy，请执行 pip install scipy。")
        if model not in ("ccr", "bcc", "both"):
            raise ValueError("model 须为 'ccr'、'bcc' 或 'both'。")
        if orientation != "input":
            raise NotImplementedError("当前版本仅支持输入导向（input orientation）。")
        self.model = model
        self.orientation = orientation
        self.input_cols = input_cols
        self.output_cols = output_cols

        self._dmus: Optional[List] = None
        self._X: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._input_names: Optional[List[str]] = None
        self._output_names: Optional[List[str]] = None

        self._ccr_efficiency: Optional[np.ndarray] = None
        self._bcc_efficiency: Optional[np.ndarray] = None
        self._scale_efficiency: Optional[np.ndarray] = None
        self._rts: Optional[List[str]] = None  # 规模报酬类型
        self._ccr_lambdas: Optional[np.ndarray] = None
        self._bcc_lambdas: Optional[np.ndarray] = None
        self._ccr_slacks: Optional[np.ndarray] = None
        self._bcc_slacks: Optional[np.ndarray] = None

    # ── 公共 API ──────────────────────────────────────────────────────────────

    def fit(
        self,
        data: pd.DataFrame,
        input_cols: Optional[List[str]] = None,
        output_cols: Optional[List[str]] = None,
    ) -> "DEA":
        """
        绑定数据。

        Parameters
        ----------
        data : pd.DataFrame
            行为 DMU，列包含输入与输出指标。
        input_cols : list of str
            输入指标列名，优先于构造函数参数。
        output_cols : list of str
            输出指标列名。
        """
        if input_cols is not None:
            self.input_cols = input_cols
        if output_cols is not None:
            self.output_cols = output_cols

        if self.input_cols is None or self.output_cols is None:
            raise ValueError("必须指定 input_cols 和 output_cols。")

        missing = set(self.input_cols + self.output_cols) - set(data.columns)
        if missing:
            raise ValueError(f"数据中缺少以下列：{missing}")

        if data.isnull().any().any():
            data = data.fillna(data.mean(numeric_only=True))
            logger.warning("缺失值已用列均值填充。")

        self._dmus = list(data.index)
        self._X = data[self.input_cols].values.astype(float)
        self._Y = data[self.output_cols].values.astype(float)
        self._input_names = self.input_cols
        self._output_names = self.output_cols

        n, m = self._X.shape
        _, s = self._Y.shape

        # 确保正值（DEA 要求）
        eps = 1e-6
        self._X = np.where(self._X <= 0, eps, self._X)
        self._Y = np.where(self._Y <= 0, eps, self._Y)

        self.metadata.update(
            {
                "n_dmus": n,
                "n_inputs": m,
                "n_outputs": s,
                "input_cols": self.input_cols,
                "output_cols": self.output_cols,
                "model": self.model,
            }
        )
        logger.info(
            "DEA.fit: %d DMU, %d 输入, %d 输出。",
            n, m, s,
        )
        return self

    def compute(self) -> pd.DataFrame:
        """
        执行 DEA 全流程计算。

        Returns
        -------
        pd.DataFrame
            列（视 model 参数）：
            ``CCR_Efficiency`` / ``BCC_Efficiency`` / ``Scale_Efficiency`` /
            ``Returns_to_Scale`` / ``CCR_Efficient`` / ``BCC_Efficient`` / ``Rank``
        """
        if self._X is None:
            raise RuntimeError("请先调用 fit()。")

        n = len(self._dmus)
        m_in = self._X.shape[1]
        m_out = self._Y.shape[1]

        run_ccr = self.model in ("ccr", "both")
        run_bcc = self.model in ("bcc", "both")

        if run_ccr:
            self._ccr_efficiency = np.zeros(n)
            self._ccr_lambdas = np.zeros((n, n))
            self._ccr_slacks = np.zeros((n, m_in + m_out))

        if run_bcc:
            self._bcc_efficiency = np.zeros(n)
            self._bcc_lambdas = np.zeros((n, n))
            self._bcc_slacks = np.zeros((n, m_in + m_out))

        for i in range(n):
            x0 = self._X[i]
            y0 = self._Y[i]

            if run_ccr:
                theta, lam, slk = _solve_ccr_input(x0, y0, self._X, self._Y)
                self._ccr_efficiency[i] = theta
                self._ccr_lambdas[i] = lam
                self._ccr_slacks[i] = slk

            if run_bcc:
                theta, lam, slk = _solve_bcc_input(x0, y0, self._X, self._Y)
                self._bcc_efficiency[i] = theta
                self._bcc_lambdas[i] = lam
                self._bcc_slacks[i] = slk

        # 规模效率 SE = CCR / BCC（仅当两模型均运行）
        result_dict: Dict[str, Any] = {}
        tol = 1e-4  # 效率值 ≥ 1-tol 视为有效

        if run_ccr:
            result_dict["CCR_Efficiency"] = np.round(self._ccr_efficiency, 6)
            result_dict["CCR_Efficient"] = (self._ccr_efficiency >= 1 - tol)

        if run_bcc:
            result_dict["BCC_Efficiency"] = np.round(self._bcc_efficiency, 6)
            result_dict["BCC_Efficient"] = (self._bcc_efficiency >= 1 - tol)

        if run_ccr and run_bcc:
            denom = np.where(self._bcc_efficiency < 1e-8, 1e-8, self._bcc_efficiency)
            self._scale_efficiency = np.clip(
                self._ccr_efficiency / denom, 0.0, 1.0
            )
            result_dict["Scale_Efficiency"] = np.round(self._scale_efficiency, 6)
            self._rts = self._determine_rts()
            result_dict["Returns_to_Scale"] = self._rts

        # 综合排名（以 CCR 为主，BCC 次之）
        if run_ccr:
            score_for_rank = self._ccr_efficiency
        else:
            score_for_rank = self._bcc_efficiency

        ranks = (
            pd.Series(score_for_rank, index=self._dmus)
            .rank(ascending=False, method="min")
            .astype(int)
        )
        result_dict["Rank"] = ranks

        self.result = pd.DataFrame(result_dict, index=self._dmus).sort_values("Rank")

        self.metadata.update(
            {
                "n_ccr_efficient": int((self._ccr_efficiency >= 1 - tol).sum())
                if run_ccr else None,
                "n_bcc_efficient": int((self._bcc_efficiency >= 1 - tol).sum())
                if run_bcc else None,
                "mean_ccr": float(self._ccr_efficiency.mean()) if run_ccr else None,
                "mean_bcc": float(self._bcc_efficiency.mean()) if run_bcc else None,
            }
        )

        logger.info(
            "DEA 计算完成 → CCR 有效 DMU 数: %s，均值效率: %.4f",
            self.metadata.get("n_ccr_efficient", "N/A"),
            self.metadata.get("mean_ccr") or 0.0,
        )
        return self.result

    def _determine_rts(self) -> List[str]:
        """
        判断规模报酬类型（IRS / CRS / DRS）。

        规则：
        - SE ≈ 1 且 CCR 有效 → CRS（规模报酬不变）
        - SE < 1 且 BCC λ 和 > 1 → DRS（规模报酬递减）
        - SE < 1 且 BCC λ 和 < 1 → IRS（规模报酬递增）
        """
        tol = 1e-4
        rts = []
        for i in range(len(self._dmus)):
            se = self._scale_efficiency[i]
            if se >= 1 - tol:
                rts.append("CRS")
            else:
                lam_sum = self._bcc_lambdas[i].sum()
                if lam_sum > 1 + tol:
                    rts.append("DRS")
                elif lam_sum < 1 - tol:
                    rts.append("IRS")
                else:
                    rts.append("CRS")
        return rts

    def get_reference_set(self, dmu: str) -> Dict[str, float]:
        """返回指定 DMU 的 BCC 参考集（λ > 0 的有效 DMU）。"""
        if self._bcc_lambdas is None:
            raise RuntimeError("请先运行 BCC 模型（model='bcc' 或 'both'）。")
        idx = self._dmus.index(dmu)
        lambdas = self._bcc_lambdas[idx]
        ref_set = {
            self._dmus[j]: round(float(lambdas[j]), 4)
            for j in range(len(self._dmus))
            if lambdas[j] > 1e-4
        }
        return ref_set

    def get_slack_analysis(self) -> pd.DataFrame:
        """返回 BCC 模型的松弛变量分析表（输入超额 + 输出不足）。"""
        if self._bcc_slacks is None:
            raise RuntimeError("请先运行 BCC 模型。")
        m_in = len(self._input_names)
        slack_cols = (
            [f"s_in({c})" for c in self._input_names]
            + [f"s_out({c})" for c in self._output_names]
        )
        return pd.DataFrame(
            np.round(self._bcc_slacks, 4),
            index=self._dmus,
            columns=slack_cols,
        )

    def summary(self) -> Dict[str, Any]:
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")
        s = {
            "method": "DEA",
            "description": "数据包络分析",
            "model": self.model,
            "orientation": self.orientation,
            "n_dmus": len(self._dmus),
            "n_inputs": len(self._input_names),
            "n_outputs": len(self._output_names),
            "input_cols": self._input_names,
            "output_cols": self._output_names,
        }
        if self._ccr_efficiency is not None:
            s["ccr_efficiency"] = dict(
                zip(self._dmus, np.round(self._ccr_efficiency, 4).tolist())
            )
            s["n_ccr_efficient"] = self.metadata["n_ccr_efficient"]
            s["mean_ccr_efficiency"] = self.metadata["mean_ccr"]
        if self._bcc_efficiency is not None:
            s["bcc_efficiency"] = dict(
                zip(self._dmus, np.round(self._bcc_efficiency, 4).tolist())
            )
            s["n_bcc_efficient"] = self.metadata["n_bcc_efficient"]
            s["mean_bcc_efficiency"] = self.metadata["mean_bcc"]
        if self._scale_efficiency is not None:
            s["scale_efficiency"] = dict(
                zip(self._dmus, np.round(self._scale_efficiency, 4).tolist())
            )
            s["returns_to_scale"] = dict(zip(self._dmus, self._rts))
        return s

    # ── 可视化 ────────────────────────────────────────────────────────────────

    def plot_efficiency(
        self,
        figsize: Tuple = (12, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制 CCR/BCC 效率值对比水平条形图。"""
        if self.result is None:
            raise RuntimeError("请先调用 compute()。")

        sorted_dmus = self.result.index.tolist()
        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=figsize)
            y_pos = np.arange(len(sorted_dmus))
            bar_h = 0.35

            run_ccr = "CCR_Efficiency" in self.result.columns
            run_bcc = "BCC_Efficiency" in self.result.columns

            if run_ccr and run_bcc:
                ccr_vals = self.result.loc[sorted_dmus, "CCR_Efficiency"].values
                bcc_vals = self.result.loc[sorted_dmus, "BCC_Efficiency"].values
                ax.barh(y_pos + bar_h / 2, ccr_vals, height=bar_h,
                        color="#3498db", alpha=0.85, label="CCR 效率",
                        edgecolor="black", linewidth=0.5)
                ax.barh(y_pos - bar_h / 2, bcc_vals, height=bar_h,
                        color="#e67e22", alpha=0.85, label="BCC 效率",
                        edgecolor="black", linewidth=0.5)
                ax.legend(fontsize=10)
            elif run_ccr:
                vals = self.result.loc[sorted_dmus, "CCR_Efficiency"].values
                ax.barh(y_pos, vals, height=bar_h * 2,
                        color="#3498db", alpha=0.85, label="CCR 效率")
            else:
                vals = self.result.loc[sorted_dmus, "BCC_Efficiency"].values
                ax.barh(y_pos, vals, height=bar_h * 2,
                        color="#e67e22", alpha=0.85, label="BCC 效率")

            ax.axvline(1.0, color="red", ls="--", lw=1.4,
                       alpha=0.7, label="效率前沿 (=1)")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_dmus, fontsize=10)
            ax.set_xlabel("效率值", fontsize=12)
            ax.set_title("DEA 效率评价结果", fontsize=14, fontweight="bold")
            ax.set_xlim(0, 1.15)
            ax.legend(fontsize=9, loc="lower right")
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_efficiency_scatter(
        self,
        figsize: Tuple = (8, 7),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制 CCR vs BCC 效率散点图，区分有效/无效 DMU。"""
        if self._ccr_efficiency is None or self._bcc_efficiency is None:
            raise RuntimeError("需要同时运行 CCR 和 BCC 模型（model='both'）。")

        tol = 1e-4
        ccr_eff = self._ccr_efficiency
        bcc_eff = self._bcc_efficiency
        both_eff = (ccr_eff >= 1 - tol) & (bcc_eff >= 1 - tol)
        only_bcc = (~(ccr_eff >= 1 - tol)) & (bcc_eff >= 1 - tol)
        neither = ~both_eff & ~only_bcc

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=figsize)

            for mask, color, label, marker in [
                (both_eff, "#2ecc71", "CCR+BCC 双有效", "★"),
                (only_bcc, "#f39c12", "仅 BCC 有效", "o"),
                (neither, "#e74c3c", "无效 DMU", "x"),
            ]:
                idx = np.where(mask)[0]
                if len(idx) > 0:
                    ax.scatter(
                        ccr_eff[idx], bcc_eff[idx],
                        s=120, c=color, label=label,
                        edgecolors="black", linewidth=0.7, zorder=5,
                        marker="*" if label.startswith("CCR") else "o"
                        if label.startswith("仅") else "x",
                    )
                    for i in idx:
                        ax.annotate(
                            self._dmus[i],
                            (ccr_eff[i], bcc_eff[i]),
                            textcoords="offset points",
                            xytext=(6, 3), fontsize=8,
                        )

            ax.plot([0, 1.05], [0, 1.05], "k--", lw=1, alpha=0.4, label="对角线")
            ax.axvline(1.0, color="blue", ls=":", alpha=0.4, lw=1)
            ax.axhline(1.0, color="blue", ls=":", alpha=0.4, lw=1)
            ax.set_xlim(0, 1.1)
            ax.set_ylim(0, 1.1)
            ax.set_xlabel("CCR 效率值", fontsize=12)
            ax.set_ylabel("BCC 效率值", fontsize=12)
            ax.set_title("DEA CCR-BCC 效率分布", fontsize=13, fontweight="bold")
            ax.legend(fontsize=9)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_rts_pie(
        self,
        figsize: Tuple = (7, 7),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """绘制规模报酬类型分布饼图。"""
        if self._rts is None:
            raise RuntimeError("需要同时运行 CCR 和 BCC 模型。")

        from collections import Counter
        counts = Counter(self._rts)
        labels = list(counts.keys())
        sizes = list(counts.values())
        color_map = {"CRS": "#2ecc71", "IRS": "#3498db", "DRS": "#e74c3c"}
        colors = [color_map.get(l, "#95a5a6") for l in labels]

        with plt.rc_context({"figure.dpi": 120}):
            fig, ax = plt.subplots(figsize=figsize)
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                textprops={"fontsize": 12},
            )
            for at in autotexts:
                at.set_fontsize(11)
            ax.set_title("规模报酬类型分布", fontsize=13, fontweight="bold", pad=20)
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def tex_description(self) -> str:
        return r"""
\subsection{基于数据包络分析的效率评价模型}

\subsubsection{模型原理}

数据包络分析（Data Envelopment Analysis, DEA）由Charnes、Cooper和Rhodes（1978）
提出，是一种基于线性规划的非参数效率评价方法，
用于评价具有多投入多产出的同类决策单元（DMU）的相对效率。

\subsubsection{CCR 模型（规模报酬不变）}

设有 $n$ 个 DMU，第 $i$ 个 DMU 有输入向量 $\boldsymbol{x}_i \in \mathbb{R}^m$
和产出向量 $\boldsymbol{y}_i \in \mathbb{R}^s$。
CCR 输入导向模型对 DMU$_0$ 的线性规划形式为：

\begin{equation}
\begin{aligned}
    \min_{\theta,\,\boldsymbol{\lambda}} &\quad \theta \\
    \text{s.t.} &\quad X\boldsymbol{\lambda} \leq \theta\boldsymbol{x}_0 \\
                &\quad Y\boldsymbol{\lambda} \geq \boldsymbol{y}_0 \\
                &\quad \boldsymbol{\lambda} \geq \boldsymbol{0}
\end{aligned}
\label{eq:dea-ccr}
\end{equation}

最优解 $\theta^* \in (0,1]$，$\theta^*=1$ 表示 DMU$_0$ 处于 CCR 有效前沿。

\subsubsection{BCC 模型（规模报酬可变）}

BCC 模型（Banker等，1984）在 CCR 基础上增加凸性约束：
\begin{equation}
    \sum_{j=1}^{n}\lambda_j = 1,\quad \boldsymbol{\lambda} \geq \boldsymbol{0}
    \label{eq:dea-bcc-conv}
\end{equation}
BCC 效率值 $\theta^*_{\text{BCC}} \geq \theta^*_{\text{CCR}}$，
两者之比即为\textbf{规模效率}：
\begin{equation}
    \text{SE}_i = \frac{\theta^*_{\text{CCR},i}}{\theta^*_{\text{BCC},i}}
    \label{eq:dea-scale}
\end{equation}

\subsubsection{规模报酬判断}

\begin{itemize}
    \item SE$_i = 1$：规模报酬不变（CRS）
    \item SE$_i < 1$ 且 $\sum\lambda_j > 1$：规模报酬递减（DRS）
    \item SE$_i < 1$ 且 $\sum\lambda_j < 1$：规模报酬递增（IRS）
\end{itemize}
"""