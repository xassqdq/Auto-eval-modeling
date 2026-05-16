# -*- coding: utf-8 -*-
"""
组合赋权法模块
=============

将多种主观/客观权重结合为综合权重的四种方法：

1. MultiplicativeCombination — 乘法合成归一化
2. LinearCombination         — 线性加权组合
3. GameTheoryCombination     — 博弈论组合赋权
4. MinDeviationCombination   — 离差最小化组合赋权

所有方法接受多组权重向量列表作为输入，产出单一综合权重向量。
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
from scipy.optimize import minimize, LinearConstraint

from ..base import BaseMethod

logger = logging.getLogger(__name__)


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


def _validate_weight_matrix(
    weights_list: List[np.ndarray],
    method_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    内部工具：校验多组权重并转换为 (K, n) 矩阵。

    Parameters
    ----------
    weights_list : list of array-like, each shape (n,)
    method_names : list of str, optional

    Returns
    -------
    W : np.ndarray, shape (K, n)  — K 组权重，n 个指标
    names : list of str
    """
    if len(weights_list) < 2:
        raise ValueError("组合赋权至少需要 2 组权重")

    W_rows = []
    for i, w in enumerate(weights_list):
        w_arr = np.asarray(w, dtype=float).ravel()
        if np.any(w_arr < 0):
            raise ValueError(f"第 {i+1} 组权重含负值")
        s = w_arr.sum()
        if abs(s - 1.0) > 1e-4:
            warnings.warn(
                f"第 {i+1} 组权重之和 = {s:.6f} ≠ 1，已自动归一化。"
            )
            w_arr = w_arr / s
        W_rows.append(w_arr)

    W = np.vstack(W_rows)    # (K, n)

    # 校验同维度
    n_cols = [row.shape[0] for row in W_rows]
    if len(set(n_cols)) != 1:
        raise ValueError(
            f"各组权重维度不一致: {n_cols}。"
            "所有权重组必须具有相同的指标数 n。"
        )

    K = len(W_rows)
    if method_names is None:
        names = [f"方法{i+1}" for i in range(K)]
    elif len(method_names) != K:
        warnings.warn(
            f"method_names 长度 {len(method_names)} ≠ 权重组数 {K}，"
            "已使用默认名称。"
        )
        names = [f"方法{i+1}" for i in range(K)]
    else:
        names = list(method_names)

    return W, names


# ═══════════════════════════════════════════════════════════════
#  公共 Mixin：weight_comparison_plot
# ═══════════════════════════════════════════════════════════════

class _WeightPlotMixin:
    """提供多方法权重对比可视化的公共绘图接口。"""

    def weight_comparison_plot(
        self,
        figsize: Tuple[int, int] = (13, 5),
        indicator_names: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        绘制三联图：各组权重条形对比 + 最终权重柱图 + 雷达图。

        Parameters
        ----------
        figsize : tuple
        indicator_names : list of str, optional
            覆盖内置指标名称

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._check_computed()
        r          = self._results
        W          = r["W"]          # (K, n)
        w_final    = r["weights"]    # (n,)
        method_names = r["method_names"]
        ind_names  = indicator_names or r.get("indicator_names") or [
            f"C{j+1}" for j in range(W.shape[1])
        ]
        K, n = W.shape

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs  = fig.add_gridspec(1, 3, width_ratios=[2, 1.2, 1.2])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2], polar=True)

        palette = plt.cm.tab10.colors

        # ── 子图1: 多组权重对比条形图 ──
        x     = np.arange(n)
        width = 0.8 / (K + 1)
        for i, (row, mname) in enumerate(zip(W, method_names)):
            offset = (i - K / 2 + 0.5) * width
            ax1.bar(x + offset, row, width,
                    color=palette[i % 10], alpha=0.8,
                    label=mname, edgecolor="white")
        # 最终组合权重折线
        ax1.plot(x, w_final, "k-o", lw=2, ms=5,
                 zorder=5, label="组合权重")
        ax1.axhline(1 / n, color="grey", ls=":", lw=1.2, label="均等权重")
        ax1.set_xticks(x)
        ax1.set_xticklabels(ind_names, rotation=45, ha="right", fontsize=9)
        ax1.set_ylabel("权重值")
        ax1.set_title("各方法权重与组合权重对比")
        ax1.legend(fontsize=8, loc="upper right")

        # ── 子图2: 最终权重水平条形图 ──
        sorted_idx  = np.argsort(w_final)
        s_names     = [ind_names[i] for i in sorted_idx]
        s_weights   = w_final[sorted_idx]
        colors      = plt.cm.RdYlGn(np.linspace(0.25, 0.85, n))
        ax2.barh(s_names, s_weights, color=colors[::-1], edgecolor="white")
        ax2.axvline(1 / n, color="grey", ls="--", lw=1)
        for xi, val in enumerate(s_weights):
            ax2.text(val + 0.002, xi, f"{val:.4f}", va="center", fontsize=8)
        ax2.set_xlabel("组合权重")
        ax2.set_title("组合权重排序")
        ax2.set_xlim(0, max(w_final) * 1.35)

        # ── 子图3: 雷达图 ──
        angles   = np.linspace(0, 2 * np.pi, n, endpoint=False)
        angles_c = np.concatenate([angles, [angles[0]]])

        for i, (row, mname) in enumerate(zip(W, method_names)):
            vals_c = np.concatenate([row, [row[0]]])
            ax3.plot(angles_c, vals_c, lw=1.5,
                     color=palette[i % 10], label=mname, alpha=0.7)
            ax3.fill(angles_c, vals_c, alpha=0.06,
                     color=palette[i % 10])

        vals_final_c = np.concatenate([w_final, [w_final[0]]])
        ax3.plot(angles_c, vals_final_c, "k-", lw=2.5, label="组合权重")
        ax3.fill(angles_c, vals_final_c, alpha=0.12, color="black")

        ax3.set_xticks(angles)
        ax3.set_xticklabels(ind_names, fontsize=8)
        ax3.set_title("权重雷达图", pad=15)
        ax3.legend(
            fontsize=7, loc="upper right",
            bbox_to_anchor=(1.35, 1.15),
        )

        fig.suptitle(f"{self.name} — 权重对比可视化",
                     fontsize=13, fontweight="bold")
        return fig

    def plot(self, **kwargs) -> plt.Figure:
        """默认 plot 调用权重对比图。"""
        return self.weight_comparison_plot(**kwargs)


# ═══════════════════════════════════════════════════════════════
#  1. MultiplicativeCombination — 乘法合成归一化
# ═══════════════════════════════════════════════════════════════

class MultiplicativeCombination(_WeightPlotMixin, BaseMethod):
    """
    乘法合成归一化 (Multiplicative Synthesis)

    将多组权重向量逐元素相乘后归一化，适用于各方法权重
    量纲一致、希望对权重较小的指标给予"惩罚放大"的场景。

    .. math::
        w_j^{\\text{combined}} = \\frac{\\prod_{k=1}^{K} (w_j^{(k)})^{\\alpha_k}}
        {\\sum_{l=1}^{n} \\prod_{k=1}^{K} (w_l^{(k)})^{\\alpha_k}}

    当 :math:`\\alpha_k = 1/K`（默认）时为等权几何平均。
    用户也可指定各方法的元权重 :math:`\\alpha_k`（和为 1）。

    Parameters
    ----------
    method_names : list of str, optional
        各权重方法的名称标签
    meta_weights : array-like of shape (K,), optional
        各方法的元权重（指数），默认等权 1/K
    indicator_names : list of str, optional
    epsilon : float, default 1e-12
        防止零权重导致乘积为 0

    Examples
    --------
    >>> import numpy as np
    >>> from src.algorithms.weights import MultiplicativeCombination
    >>>
    >>> w_ahp    = np.array([0.50, 0.30, 0.20])
    >>> w_entropy= np.array([0.35, 0.40, 0.25])
    >>> mc = MultiplicativeCombination(
    ...     method_names=["AHP", "熵权法"],
    ...     indicator_names=["C1", "C2", "C3"],
    ... )
    >>> mc.fit([w_ahp, w_entropy]).compute().summary()
    """

    def __init__(
        self,
        method_names:     Optional[List[str]] = None,
        meta_weights:     Optional[Union[np.ndarray, List]] = None,
        indicator_names:  Optional[List[str]] = None,
        epsilon:          float = 1e-12,
    ) -> None:
        super().__init__(
            name="乘法合成组合赋权",
            description="多组权重逐元素几何加权乘积后归一化",
        )
        self.method_names_init = method_names
        self.meta_weights_init = meta_weights
        self.indicator_names   = indicator_names
        self.epsilon           = epsilon
        self._W: Optional[np.ndarray] = None

    def fit(
        self,
        weights_list: List[Union[np.ndarray, List]],
        method_names:    Optional[List[str]] = None,
        indicator_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "MultiplicativeCombination":
        """
        载入多组权重。

        Parameters
        ----------
        weights_list : list of array-like, each shape (n,)
        """
        mn = method_names or self.method_names_init
        self._W, self._method_names = _validate_weight_matrix(
            weights_list, mn
        )
        if indicator_names is not None:
            self.indicator_names = indicator_names
        if self.indicator_names is None:
            n = self._W.shape[1]
            self.indicator_names = [f"C{j+1}" for j in range(n)]
        self._fitted = True
        return self

    def compute(self) -> "MultiplicativeCombination":
        """执行乘法合成计算。"""
        self._check_fitted()
        K, n = self._W.shape

        # ── 元权重（指数）──
        if self.meta_weights_init is not None:
            alpha = np.asarray(self.meta_weights_init, dtype=float)
            if len(alpha) != K:
                raise ValueError(
                    f"meta_weights 长度 {len(alpha)} ≠ 权重方法数 {K}"
                )
            if abs(alpha.sum() - 1.0) > 1e-4:
                warnings.warn("meta_weights 之和不为 1，已自动归一化。")
                alpha = alpha / alpha.sum()
        else:
            alpha = np.ones(K) / K    # 等权

        # ── 乘法合成（避免零值）──
        W_safe = np.clip(self._W, self.epsilon, None)
        # log-domain 计算
        log_prod = (alpha[:, None] * np.log(W_safe)).sum(axis=0)
        product  = np.exp(log_prod)
        weights  = product / product.sum()

        self._results = {
            "weights":        weights,
            "product":        product,
            "alpha":          alpha,
            "W":              self._W,
            "method_names":   self._method_names,
            "indicator_names": list(self.indicator_names),
            "K":              K,
            "n":              n,
        }
        self._computed = True
        return self

    def summary(self) -> Dict[str, Any]:
        self._check_computed()
        r = self._results
        rows = {"指标": self.indicator_names}
        for i, (mn, row) in enumerate(zip(r["method_names"], r["W"])):
            rows[f"{mn} (α={r['alpha'][i]:.3f})"] = np.round(row, 4)
        rows["组合权重 w_j"] = np.round(r["weights"], 6)

        df = pd.DataFrame(rows)
        r["weight_df"] = df

        print("=" * 62)
        print("  乘法合成组合赋权 — 计算结果")
        print("=" * 62)
        print(f"  方法数 K={r['K']}，指标数 n={r['n']}")
        print(
            "  元权重 α = ["
            + ", ".join(f"{a:.4f}" for a in r["alpha"]) + "]"
        )
        print("-" * 62)
        print(df.to_string(index=False))
        print("=" * 62)
        return r

    def tex_description(self) -> str:
        self._check_computed()
        r = self._results
        w_str   = ", ".join(f"{w:.4f}" for w in r["weights"])
        alpha_str = ", ".join(f"{a:.3f}" for a in r["alpha"])
        method_str = "、".join(r["method_names"])
        tex = (
            r"\subsubsection{乘法合成组合赋权}" "\n\n"
            f"采用{method_str}共 {r['K']} 种方法分别计算权重，"
            f"各方法元权重为 $(\\alpha_1,\\ldots,\\alpha_K) = ({alpha_str})$，"
            "通过乘法合成：\n\n"
            r"\begin{equation}" "\n"
            r"  w_j = \frac{\prod_{k=1}^{K} (w_j^{(k)})^{\alpha_k}}"
            r"{\sum_{l=1}^{n} \prod_{k=1}^{K} (w_l^{(k)})^{\alpha_k}}"
            "\n"
            r"\end{equation}" "\n\n"
            f"最终综合权重向量 $\\mathbf{{w}} = ({w_str})^\\top$。\n"
        )
        return tex

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._results.get("weights")


# ═══════════════════════════════════════════════════════════════
#  2. LinearCombination — 线性加权组合
# ═══════════════════════════════════════════════════════════════

class LinearCombination(_WeightPlotMixin, BaseMethod):
    """
    线性加权组合 (Linear Weighted Combination)

    最直觉的组合方式：为各赋权方法指定组合系数（元权重），
    对多组权重做线性加权平均：

    .. math::
        w_j^{\\text{combined}} = \\sum_{k=1}^{K} \\alpha_k \\cdot w_j^{(k)}

    其中 :math:`\\sum_{k=1}^{K} \\alpha_k = 1`，:math:`\\alpha_k \\geq 0`。

    Parameters
    ----------
    method_names : list of str, optional
    meta_weights : array-like of shape (K,), optional
        各方法的线性组合系数，默认等权 1/K
    indicator_names : list of str, optional
    auto_optimize : bool, default False
        若为 True，则通过最大化加权方差（"最大信息量"准则）
        自动优化元权重 alpha（此时忽略手动指定的 meta_weights）

    Examples
    --------
    >>> from src.algorithms.weights import LinearCombination
    >>> import numpy as np
    >>>
    >>> lc = LinearCombination(
    ...     method_names=["AHP", "熵权法", "CRITIC"],
    ...     meta_weights=[0.4, 0.35, 0.25],
    ...     indicator_names=["技术", "经济", "社会", "环境"],
    ... )
    >>> lc.fit([w1, w2, w3]).compute().summary()
    """

    def __init__(
        self,
        method_names:    Optional[List[str]] = None,
        meta_weights:    Optional[Union[np.ndarray, List]] = None,
        indicator_names: Optional[List[str]] = None,
        auto_optimize:   bool = False,
    ) -> None:
        super().__init__(
            name="线性加权组合赋权",
            description="对多组权重做线性加权平均合成综合权重",
        )
        self.method_names_init = method_names
        self.meta_weights_init = meta_weights
        self.indicator_names   = indicator_names
        self.auto_optimize     = auto_optimize
        self._W: Optional[np.ndarray] = None

    def fit(
        self,
        weights_list:    List[Union[np.ndarray, List]],
        method_names:    Optional[List[str]] = None,
        indicator_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "LinearCombination":
        mn = method_names or self.method_names_init
        self._W, self._method_names = _validate_weight_matrix(
            weights_list, mn
        )
        if indicator_names is not None:
            self.indicator_names = indicator_names
        if self.indicator_names is None:
            n = self._W.shape[1]
            self.indicator_names = [f"C{j+1}" for j in range(n)]
        self._fitted = True
        return self

    def compute(self) -> "LinearCombination":
        """计算线性组合权重。"""
        self._check_fitted()
        K, n = self._W.shape

        if self.auto_optimize:
            alpha, opt_info = self._optimize_alpha(self._W)
        else:
            if self.meta_weights_init is not None:
                alpha = np.asarray(self.meta_weights_init, dtype=float)
                if len(alpha) != K:
                    raise ValueError(
                        f"meta_weights 长度 {len(alpha)} ≠ K={K}"
                    )
                if abs(alpha.sum() - 1.0) > 1e-4:
                    warnings.warn("meta_weights 之和不为 1，已自动归一化。")
                    alpha = alpha / alpha.sum()
            else:
                alpha = np.ones(K) / K
            opt_info = None

        # w = α^T · W，其中 W shape=(K, n)
        weights = alpha @ self._W    # shape (n,)
        weights = np.clip(weights, 0.0, None)
        weights /= weights.sum()

        self._results = {
            "weights":        weights,
            "alpha":          alpha,
            "W":              self._W,
            "method_names":   self._method_names,
            "indicator_names": list(self.indicator_names),
            "K":              K,
            "n":              n,
            "auto_optimize":  self.auto_optimize,
            "opt_info":       opt_info,
        }
        self._computed = True
        return self

    @staticmethod
    def _optimize_alpha(W: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        最大化加权方差准则：
        目标函数 max Σ_j [w_j * σ_j^2(W)]，等价于最大化权重向量的加权离散度。
        这里用近似：最大化 Σ_j (Σ_k α_k w_jk)^2 - (Σ_j Σ_k α_k w_jk)^2 / n
        即最大化综合权重的方差。
        """
        K, n = W.shape

        def neg_variance(alpha: np.ndarray) -> float:
            w_comb = alpha @ W    # (n,)
            return -float(np.var(w_comb))

        # 约束：Σ α_k = 1
        constraints = [{"type": "eq", "fun": lambda a: a.sum() - 1.0}]
        bounds      = [(0.0, 1.0)] * K
        x0          = np.ones(K) / K

        result = minimize(
            neg_variance, x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success:
            warnings.warn(
                f"元权重优化未收敛: {result.message}，使用等权替代。"
            )
            return np.ones(K) / K, result

        alpha_opt = result.x
        alpha_opt = np.clip(alpha_opt, 0.0, 1.0)
        alpha_opt /= alpha_opt.sum()
        return alpha_opt, result

    def summary(self) -> Dict[str, Any]:
        self._check_computed()
        r   = self._results
        rows = {"指标": self.indicator_names}
        for i, (mn, row) in enumerate(zip(r["method_names"], r["W"])):
            rows[f"{mn} (α={r['alpha'][i]:.3f})"] = np.round(row, 4)
        rows["组合权重 w_j"] = np.round(r["weights"], 6)
        df = pd.DataFrame(rows)
        r["weight_df"] = df

        opt_note = ""
        if r["auto_optimize"] and r["opt_info"] is not None:
            opt_note = f"\n  （元权重通过最大化加权方差自动优化，"
            opt_note += f"状态: {'成功' if r['opt_info'].success else '未收敛'}）"

        print("=" * 60)
        print("  线性加权组合赋权 — 计算结果" + opt_note)
        print("=" * 60)
        print(f"  方法数 K={r['K']}，指标数 n={r['n']}")
        print("  元权重 α = [" +
              ", ".join(f"{a:.4f}" for a in r["alpha"]) + "]")
        print("-" * 60)
        print(df.to_string(index=False))
        print("=" * 60)
        return r

    def tex_description(self) -> str:
        self._check_computed()
        r = self._results
        w_str     = ", ".join(f"{w:.4f}" for w in r["weights"])
        alpha_str = ", ".join(f"{a:.3f}" for a in r["alpha"])
        method_str = "、".join(r["method_names"])
        opt_note   = (
            "元权重通过最大化综合权重方差准则自动优化。\n\n"
            if r["auto_optimize"] else
            "元权重由用户指定。\n\n"
        )
        tex = (
            r"\subsubsection{线性加权组合赋权}" "\n\n"
            f"采用{method_str}共 {r['K']} 种方法分别计算权重，"
            f"{opt_note}"
            f"各方法元权重 $(\\alpha_1,\\ldots,\\alpha_K) = ({alpha_str})$，"
            "线性加权合成综合权重：\n\n"
            r"\begin{equation}" "\n"
            r"  w_j = \sum_{k=1}^{K} \alpha_k \cdot w_j^{(k)},"
            r"\quad \sum_{k=1}^{K} \alpha_k = 1,\; \alpha_k \geq 0"
            "\n"
            r"\end{equation}" "\n\n"
            f"最终综合权重向量 $\\mathbf{{w}} = ({w_str})^\\top$。\n"
        )
        return tex

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._results.get("weights")


# ═══════════════════════════════════════════════════════════════
#  3. GameTheoryCombination — 博弈论组合赋权
# ═══════════════════════════════════════════════════════════════

class GameTheoryCombination(_WeightPlotMixin, BaseMethod):
    """
    博弈论组合赋权 (Game Theory Based Combination)

    将多种赋权方法视为"博弈参与者"，通过求解纳什均衡
    确定各赋权方法的最优组合系数，使综合权重与各单一权重
    的整体偏差最小（均衡博弈论视角）。

    优化目标
    --------
    设各方法权重向量矩阵 :math:`\\mathbf{W} = [w^{(1)}, \\ldots, w^{(K)}]^\\top`，
    组合系数 :math:`\\boldsymbol{\\alpha} = (\\alpha_1, \\ldots, \\alpha_K)^\\top`，

    最小化综合权重 :math:`w = \\alpha^\\top W` 与所有单一权重的加权偏差：

    .. math::
        \\min_{\\alpha} \\sum_{k=1}^{K}
        \\left\\| \\sum_{l=1}^{K} \\alpha_l w^{(l)} - w^{(k)} \\right\\|^2

    约束：:math:`\\sum_k \\alpha_k = 1`，:math:`\\alpha_k \\geq 0`。

    最优解通过求解线性方程组（KKT 条件）得到。

    Parameters
    ----------
    method_names : list of str, optional
    indicator_names : list of str, optional
    solver : {'kkt', 'scipy'}
        * ``kkt``   — 直接求解 KKT 线性系统（精确解）
        * ``scipy`` — scipy 优化器（数值解，更鲁棒）

    References
    ----------
    Wang, T. C., & Lee, H. D. (2009).
    Developing a fuzzy TOPSIS approach based on subjective weights
    and objective weights. *Expert Systems with Applications*, 36(5).

    Examples
    --------
    >>> from src.algorithms.weights import GameTheoryCombination
    >>> gtc = GameTheoryCombination(
    ...     method_names=["AHP", "熵权法"],
    ...     indicator_names=["C1","C2","C3","C4"],
    ... )
    >>> gtc.fit([w_ahp, w_entropy]).compute().summary()
    """

    def __init__(
        self,
        method_names:    Optional[List[str]] = None,
        indicator_names: Optional[List[str]] = None,
        solver:          str = "kkt",
    ) -> None:
        super().__init__(
            name="博弈论组合赋权",
            description="博弈论纳什均衡确定各赋权方法的最优组合系数",
        )
        self.method_names_init = method_names
        self.indicator_names   = indicator_names
        self.solver            = solver
        self._W: Optional[np.ndarray] = None

    def fit(
        self,
        weights_list:    List[Union[np.ndarray, List]],
        method_names:    Optional[List[str]] = None,
        indicator_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "GameTheoryCombination":
        mn = method_names or self.method_names_init
        self._W, self._method_names = _validate_weight_matrix(
            weights_list, mn
        )
        if indicator_names is not None:
            self.indicator_names = indicator_names
        if self.indicator_names is None:
            n = self._W.shape[1]
            self.indicator_names = [f"C{j+1}" for j in range(n)]
        self._fitted = True
        return self

    def compute(self) -> "GameTheoryCombination":
        """计算博弈论组合系数与综合权重。"""
        self._check_fitted()
        K, n = self._W.shape

        if self.solver == "kkt":
            alpha, converged = self._solve_kkt(self._W)
        else:
            alpha, converged = self._solve_scipy(self._W)

        if not converged:
            warnings.warn(
                "博弈论组合赋权求解未完全收敛，结果仅供参考。"
            )

        alpha = np.clip(alpha, 0.0, 1.0)
        alpha /= alpha.sum()

        weights = alpha @ self._W
        weights = np.clip(weights, 0.0, None)
        weights /= weights.sum()

        # 计算综合权重与各单一权重的偏差
        deviations = np.array([
            float(np.linalg.norm(weights - self._W[k]))
            for k in range(K)
        ])

        self._results = {
            "weights":        weights,
            "alpha":          alpha,
            "deviations":     deviations,
            "W":              self._W,
            "method_names":   self._method_names,
            "indicator_names": list(self.indicator_names),
            "K":              K,
            "n":              n,
            "converged":      converged,
            "solver":         self.solver,
        }
        self._computed = True
        return self

    @staticmethod
    def _solve_kkt(W: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        直接求解 KKT 一阶条件线性系统。

        最优化问题等价于最小二乘：
            min_{α} ||W^T α - w||^2 的加权版，
        其 KKT 系统为：
            [2 * W W^T | ones_K ] [α]   [sum_k W[k]]
            [ones_K^T  |    0   ] [λ] = [    1     ]
        """
        K = W.shape[0]
        # G = W @ W^T，shape (K, K)
        G = W @ W.T

        # 构造 KKT 矩阵
        # [2G  1] [α]   [2G * ones / K]
        # [1^T 0] [λ] = [     1       ]
        ones_K = np.ones((K, 1))
        zero   = np.zeros((1, 1))

        # 右端向量：最小化 Σ_k ||α^T W - W[k]||^2 → ∂/∂α = 0
        # ∂/∂α_i = 2 Σ_k (Σ_l α_l w^(l) - w^(k)) w^(i) = 0
        # ⟹ 2G α = 2G (1/K) ones → 解为 α_init = 1/K
        # 实际求解含等式约束版本
        A = np.block([
            [2.0 * G, ones_K],
            [ones_K.T, zero],
        ])
        # 右端 b：2 * G @ (ones/K) 对应的等式  +  约束 Σα=1
        rhs_top = 2.0 * G.sum(axis=1) / K    # (K,)
        b = np.concatenate([rhs_top, [1.0]])

        try:
            sol = np.linalg.solve(A, b)
            alpha = sol[:K]
            converged = True
        except np.linalg.LinAlgError:
            # 矩阵奇异时退化为均等权重
            logger.warning("KKT 矩阵奇异，退化为均等权重。")
            alpha = np.ones(K) / K
            converged = False

        return alpha, converged

    @staticmethod
    def _solve_scipy(W: np.ndarray) -> Tuple[np.ndarray, bool]:
        """scipy 数值求解版本（更鲁棒）。"""
        K = W.shape[0]

        def objective(alpha: np.ndarray) -> float:
            w_comb = alpha @ W    # (n,)
            return float(sum(
                np.sum((w_comb - W[k]) ** 2)
                for k in range(K)
            ))

        def jac(alpha: np.ndarray) -> np.ndarray:
            w_comb = alpha @ W
            residuals = np.array([w_comb - W[k] for k in range(K)])   # (K, n)
            # ∂f/∂α_i = 2 Σ_k <w_comb - W[k], W[i]>
            grad = 2.0 * np.array([
                sum(np.dot(residuals[k], W[i]) for k in range(K))
                for i in range(K)
            ])
            return grad

        constraints = [{"type": "eq", "fun": lambda a: a.sum() - 1.0}]
        bounds      = [(0.0, 1.0)] * K
        x0          = np.ones(K) / K

        result = minimize(
            objective, x0, jac=jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        return result.x, result.success

    def summary(self) -> Dict[str, Any]:
        self._check_computed()
        r = self._results
        rows = {"指标": self.indicator_names}
        for i, (mn, row) in enumerate(zip(r["method_names"], r["W"])):
            rows[f"{mn} (α={r['alpha'][i]:.4f})"] = np.round(row, 4)
        rows["组合权重 w_j"] = np.round(r["weights"], 6)
        df = pd.DataFrame(rows)
        r["weight_df"] = df

        print("=" * 65)
        print("  博弈论组合赋权 — 计算结果")
        print("=" * 65)
        print(f"  方法数 K={r['K']}，指标数 n={r['n']}")
        print(f"  求解器: {r['solver']}，收敛: {r['converged']}")
        print("  最优元权重 α = [" +
              ", ".join(f"{a:.4f}" for a in r["alpha"]) + "]")
        print("  各方法偏差 = [" +
              ", ".join(f"{d:.4f}" for d in r["deviations"]) + "]")
        print("-" * 65)
        print(df.to_string(index=False))
        print("=" * 65)
        return r

    def tex_description(self) -> str:
        self._check_computed()
        r = self._results
        w_str     = ", ".join(f"{w:.4f}" for w in r["weights"])
        alpha_str = ", ".join(f"{a:.4f}" for a in r["alpha"])
        method_str = "、".join(r["method_names"])
        tex = (
            r"\subsubsection{博弈论组合赋权}" "\n\n"
            f"采用{method_str}共 {r['K']} 种方法分别计算权重，"
            "以博弈论纳什均衡思想确定各方法的最优组合系数，"
            "最小化综合权重与所有单一权重的整体偏差：\n\n"
            r"\begin{equation}" "\n"
            r"  \min_{\alpha} \sum_{k=1}^{K}"
            r"  \left\| \sum_{l=1}^{K} \alpha_l w^{(l)} - w^{(k)} \right\|^2,"
            r"\quad"
            r"  \text{s.t.} \sum_k \alpha_k = 1,\; \alpha_k \geq 0"
            "\n"
            r"\end{equation}" "\n\n"
            f"求得最优元权重 $\\boldsymbol{{\\alpha}} = ({alpha_str})$，"
            "综合权重向量：\n\n"
            r"\begin{equation}" "\n"
            f"  \\mathbf{{w}} = ({w_str})^\\top\n"
            r"\end{equation}" "\n"
        )
        return tex

    def plot(self, figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
        """在基类可视化基础上额外添加偏差分析图。"""
        self._check_computed()
        r   = self._results
        fig = self.weight_comparison_plot(figsize=figsize)

        # 额外追加偏差信息到第一子图标题
        ax0 = fig.get_axes()[0]
        dev_str = "  ".join(
            f"{mn}:{dv:.4f}"
            for mn, dv in zip(r["method_names"], r["deviations"])
        )
        ax0.set_xlabel(f"偏差 → {dev_str}", fontsize=8, color="grey")
        return fig

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._results.get("weights")


# ═══════════════════════════════════════════════════════════════
#  4. MinDeviationCombination — 离差最小化组合赋权
# ═══════════════════════════════════════════════════════════════

class MinDeviationCombination(_WeightPlotMixin, BaseMethod):
    """
    离差最小化组合赋权 (Minimum Deviation Combination)

    从**评价结果层面**而非权重层面出发，给定决策矩阵，
    寻找组合系数 :math:`\\alpha_k` 使各评价对象在综合权重下
    的排序与各单一权重方案下排序的离差之和最小。

    当没有决策矩阵时，退化为最小化综合权重向量与各方法
    权重向量之间的 L2 离差（等同于博弈论中的平均方案）。

    完整优化模型
    ------------
    给定决策矩阵 :math:`X`（已正向化），综合得分：

    .. math::
        S_i(\\alpha) = \\sum_{j=1}^{n} \\left(\\sum_{k=1}^{K}
        \\alpha_k w_j^{(k)}\\right) x_{ij}

    目标：最小化各单一方法得分与综合得分的总偏差：

    .. math::
        \\min_{\\alpha} \\sum_{k=1}^{K}
        \\sum_{i=1}^{m} \\left( S_i(\\alpha) - S_i^{(k)} \\right)^2

    无 X 时退化为权重空间的离差最小化：

    .. math::
        \\min_{\\alpha} \\sum_{k=1}^{K}
        \\left\\| \\sum_{l=1}^{K} \\alpha_l w^{(l)} - w^{(k)} \\right\\|^2

    Parameters
    ----------
    method_names : list of str, optional
    indicator_names : list of str, optional
    decision_matrix : np.ndarray, shape (m, n), optional
        已正向化的决策矩阵（提供时使用评价结果层面优化）
    reg_lambda : float, default 0.0
        L2 正则化系数（防止元权重过于极端）

    Examples
    --------
    >>> from src.algorithms.weights import MinDeviationCombination
    >>> mdc = MinDeviationCombination(
    ...     method_names=["AHP", "熵权法"],
    ...     decision_matrix=X_normalized,
    ... )
    >>> mdc.fit([w_ahp, w_entropy]).compute().summary()
    """

    def __init__(
        self,
        method_names:     Optional[List[str]] = None,
        indicator_names:  Optional[List[str]] = None,
        decision_matrix:  Optional[np.ndarray] = None,
        reg_lambda:       float = 0.0,
    ) -> None:
        super().__init__(
            name="离差最小化组合赋权",
            description="最小化综合得分与各单一方法得分偏差确定最优组合系数",
        )
        self.method_names_init = method_names
        self.indicator_names   = indicator_names
        self.decision_matrix   = decision_matrix
        self.reg_lambda        = reg_lambda
        self._W: Optional[np.ndarray] = None

    def fit(
        self,
        weights_list:    List[Union[np.ndarray, List]],
        method_names:    Optional[List[str]] = None,
        indicator_names: Optional[List[str]] = None,
        decision_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "MinDeviationCombination":
        mn = method_names or self.method_names_init
        self._W, self._method_names = _validate_weight_matrix(
            weights_list, mn
        )
        if indicator_names is not None:
            self.indicator_names = indicator_names
        if self.indicator_names is None:
            n = self._W.shape[1]
            self.indicator_names = [f"C{j+1}" for j in range(n)]
        if decision_matrix is not None:
            self.decision_matrix = np.asarray(decision_matrix, dtype=float)

        self._fitted = True
        return self

    def compute(self) -> "MinDeviationCombination":
        """计算离差最小化组合权重。"""
        self._check_fitted()
        K, n = self._W.shape

        if self.decision_matrix is not None:
            X    = self.decision_matrix
            if X.shape[1] != n:
                raise ValueError(
                    f"决策矩阵列数 {X.shape[1]} ≠ 指标数 n={n}"
                )
            alpha, converged, mode = self._solve_with_X(
                self._W, X, self.reg_lambda
            )
        else:
            # 退化为权重空间离差最小化（与博弈论相同的 KKT 解）
            alpha, converged = GameTheoryCombination._solve_kkt(self._W)
            mode = "weight_space"

        alpha = np.clip(alpha, 0.0, 1.0)
        alpha /= alpha.sum()

        weights = alpha @ self._W
        weights = np.clip(weights, 0.0, None)
        weights /= weights.sum()

        # ── 计算各方法综合得分（若有 X）──
        score_info = None
        if self.decision_matrix is not None:
            X      = self.decision_matrix
            scores = {
                mn: (X @ self._W[k]).tolist()
                for k, mn in enumerate(self._method_names)
            }
            scores["组合权重"] = (X @ weights).tolist()
            score_info = scores

        self._results = {
            "weights":        weights,
            "alpha":          alpha,
            "W":              self._W,
            "method_names":   self._method_names,
            "indicator_names": list(self.indicator_names),
            "K":              K,
            "n":              n,
            "converged":      converged,
            "mode":           mode,
            "score_info":     score_info,
        }
        self._computed = True
        return self

    @staticmethod
    def _solve_with_X(
        W: np.ndarray, X: np.ndarray, reg_lambda: float
    ) -> Tuple[np.ndarray, bool, str]:
        """
        基于决策矩阵的评价结果层面离差最小化。

        对象得分 S_i(α) = X @ (α^T W)^T = X @ W^T @ α
        各方法得分 S^(k) = X @ W[k]
        """
        K  = W.shape[0]
        # S_k = X @ W[k]  shape (m,)
        # S(α) = X @ (W^T α) = (X @ W^T) α = A α, where A = X @ W^T  shape (m, K)
        A   = X @ W.T   # (m, K)

        # min Σ_k ||A α - A[:,k]||^2 + λ ||α||^2
        # = min α^T (K * A^T A + λ I) α - 2 α^T A^T Σ_k A[:,k] + const
        G   = K * (A.T @ A) + reg_lambda * np.eye(K)   # (K, K)
        b   = A.T @ A.sum(axis=1)                        # (K,)

        def objective(alpha: np.ndarray) -> float:
            s_comb = A @ alpha
            s_each = [A[:, k] for k in range(K)]
            val    = sum(np.sum((s_comb - sk) ** 2) for sk in s_each)
            if reg_lambda > 0:
                val += reg_lambda * np.dot(alpha, alpha)
            return float(val)

        def jac(alpha: np.ndarray) -> np.ndarray:
            s_comb = A @ alpha
            s_each = [A[:, k] for k in range(K)]
            grad = sum(
                2 * A.T @ (s_comb - sk) for sk in s_each
            )
            if reg_lambda > 0:
                grad = grad + 2 * reg_lambda * alpha
            return grad

        constraints = [{"type": "eq", "fun": lambda a: a.sum() - 1.0}]
        bounds      = [(0.0, 1.0)] * K
        x0          = np.ones(K) / K

        result = minimize(
            objective, x0, jac=jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        return result.x, result.success, "score_space"

    def summary(self) -> Dict[str, Any]:
        self._check_computed()
        r = self._results
        rows = {"指标": self.indicator_names}
        for i, (mn, row) in enumerate(zip(r["method_names"], r["W"])):
            rows[f"{mn} (α={r['alpha'][i]:.4f})"] = np.round(row, 4)
        rows["组合权重 w_j"] = np.round(r["weights"], 6)
        df = pd.DataFrame(rows)
        r["weight_df"] = df

        print("=" * 65)
        print("  离差最小化组合赋权 — 计算结果")
        print("=" * 65)
        print(f"  方法数 K={r['K']}，指标数 n={r['n']}")
        print(f"  优化模式: {r['mode']}，收敛: {r['converged']}")
        print("  最优元权重 α = [" +
              ", ".join(f"{a:.4f}" for a in r["alpha"]) + "]")
        print("-" * 65)
        print(df.to_string(index=False))

        if r.get("score_info"):
            print("\n  各方法综合得分对比：")
            score_df = pd.DataFrame(r["score_info"])
            score_df.index = [f"对象{i+1}" for i in range(len(score_df))]
            print(score_df.round(4).to_string())
        print("=" * 65)
        return r

    def tex_description(self) -> str:
        self._check_computed()
        r = self._results
        w_str     = ", ".join(f"{w:.4f}" for w in r["weights"])
        alpha_str = ", ".join(f"{a:.4f}" for a in r["alpha"])
        method_str = "、".join(r["method_names"])

        if r["mode"] == "score_space":
            obj_desc = (
                "综合得分与各单一赋权方法得分的总离差：\n\n"
                r"\begin{equation}" "\n"
                r"  \min_{\alpha} \sum_{k=1}^{K}"
                r"  \sum_{i=1}^{m} \left( S_i(\alpha) - S_i^{(k)} \right)^2,"
                r"\quad S_i(\alpha) = \sum_{j=1}^{n}"
                r"  \left(\sum_{l=1}^{K} \alpha_l w_j^{(l)}\right) x_{ij}"
                "\n"
                r"\end{equation}" "\n\n"
            )
        else:
            obj_desc = (
                "综合权重与各单一权重向量的总 L2 离差：\n\n"
                r"\begin{equation}" "\n"
                r"  \min_{\alpha} \sum_{k=1}^{K}"
                r"  \left\| \sum_{l=1}^{K} \alpha_l w^{(l)} - w^{(k)} \right\|^2"
                "\n"
                r"\end{equation}" "\n\n"
            )

        tex = (
            r"\subsubsection{离差最小化组合赋权}" "\n\n"
            f"采用{method_str}共 {r['K']} 种方法分别计算权重，"
            f"通过最小化{obj_desc}"
            r"约束 $\sum_k \alpha_k = 1$，$\alpha_k \geq 0$，"
            f"求得最优元权重 $\\boldsymbol{{\\alpha}} = ({alpha_str})$，"
            "综合权重向量：\n\n"
            r"\begin{equation}" "\n"
            f"  \\mathbf{{w}} = ({w_str})^\\top\n"
            r"\end{equation}" "\n"
        )
        return tex

    def plot(self, figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
        """权重对比 + 可选的综合得分对比。"""
        self._check_computed()
        r = self._results

        if r.get("score_info") is None:
            return self.weight_comparison_plot(figsize=figsize)

        # 有决策矩阵时：权重对比（左） + 得分对比（右）
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle("离差最小化组合赋权 — 结果可视化",
                     fontsize=13, fontweight="bold")

        # ── 左图：权重对比 ──
        ax   = axes[0]
        K, n = r["W"].shape
        x    = np.arange(n)
        pal  = plt.cm.tab10.colors
        width = 0.8 / (K + 1)
        for i, (row, mn) in enumerate(zip(r["W"], r["method_names"])):
            offset = (i - K / 2 + 0.5) * width
            ax.bar(x + offset, row, width,
                   color=pal[i % 10], alpha=0.8, label=mn,
                   edgecolor="white")
        ax.plot(x, r["weights"], "k-o", lw=2, ms=5,
                zorder=5, label="组合权重")
        ax.set_xticks(x)
        ax.set_xticklabels(r["indicator_names"], rotation=45, ha="right")
        ax.set_ylabel("权重值")
        ax.set_title("权重方案对比")
        ax.legend(fontsize=8)

        # ── 右图：综合得分对比（折线图）──
        ax      = axes[1]
        scores  = r["score_info"]
        m       = len(next(iter(scores.values())))
        obj_idx = np.arange(m)
        for i, (mname, score_vals) in enumerate(scores.items()):
            ls   = "-" if mname == "组合权重" else "--"
            lw   = 2.5 if mname == "组合权重" else 1.5
            col  = "black" if mname == "组合权重" else pal[i % 10]
            ax.plot(obj_idx, score_vals, ls, lw=lw,
                    color=col, marker="o" if mname == "组合权重" else "s",
                    ms=5, label=mname)
        ax.set_xticks(obj_idx)
        ax.set_xticklabels([f"对象{i+1}" for i in range(m)],
                           rotation=30, ha="right")
        ax.set_ylabel("综合得分")
        ax.set_title("各方法综合得分对比")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        return fig

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self._results.get("weights")