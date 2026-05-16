"""
可视化图表生成器
================================================
根据评价算法的输出结果自动生成系列学术图表，
包括权重条形图、综合得分排序图、雷达图、
热力图（相关性/关联系数）、灵敏度折线图、
PCA 碎石图、箱线图等。

设计原则
--------
- 所有图表统一配置字体、颜色主题、坐标轴风格。
- 每个 plot_*() 方法返回 (fig, ax) 或 (fig, axes) 元组，
  调用者可进一步自定义后再保存。
- save_all() 将已生成的所有图表批量保存。
- 支持中/英文标签切换。
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# matplotlib 必须在导入 pyplot 之前设置后端（无显示环境时使用 Agg）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  全局样式配置
# ══════════════════════════════════════════════════════════════════════════════

# 自定义配色方案（专业蓝绿渐变 + 暖色强调）
_PALETTE_BLUE  = ["#1f4e79", "#2e75b6", "#5ba3d9", "#9dc3e6", "#c5dff4"]
_PALETTE_GRAD  = ["#1f4e79", "#2e75b6", "#5ba3d9", "#e07b39", "#c55a11"]
_PALETTE_QUAL  = [
    "#2e75b6", "#e07b39", "#70ad47", "#ffc000",
    "#7030a0", "#c00000", "#00b0f0", "#92d050",
]
_CMAP_BLUE = LinearSegmentedColormap.from_list(
    "AutoEval_Blue", ["#c5dff4", "#1f4e79"]
)
_CMAP_RWG  = LinearSegmentedColormap.from_list(
    "AutoEval_RWG", ["#c00000", "#ffffff", "#70ad47"]
)


def _setup_style(font_size: int = 11) -> None:
    """配置全局 matplotlib 样式（含中文字体自动检测）。"""
    font_candidates = [
        "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
        "PingFang SC", "STHeiti", "Noto Sans CJK SC", "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen_font = "DejaVu Sans"
    for font in font_candidates:
        if font in available:
            chosen_font = font
            break

    plt.rcParams.update({
        "font.family":          chosen_font,
        "font.size":            font_size,
        "axes.unicode_minus":   False,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "axes.grid.axis":       "y",
        "grid.alpha":           0.3,
        "grid.linestyle":       "--",
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",
        "legend.framealpha":    0.85,
        "legend.fontsize":      font_size - 1,
    })
    logger.debug("绘图样式已配置 [font=%s, size=%d]", chosen_font, font_size)


_setup_style()


# ══════════════════════════════════════════════════════════════════════════════
#  主类
# ══════════════════════════════════════════════════════════════════════════════

class PlotBuilder:
    """
    学术评价图表生成器。

    Parameters
    ----------
    language : str
        ``"cn"`` 或 ``"en"``，控制坐标轴、图例等文字语言。
    output_dir : str or Path
        图片默认保存目录。
    fig_format : str
        图片格式（``"png"`` / ``"pdf"`` / ``"svg"``）。
    font_size : int
        全局字体大小（默认 11pt）。
    dpi : int
        保存分辨率（默认 300）。
    """

    def __init__(
        self,
        language: str = "cn",
        output_dir: Union[str, Path] = "output/figures",
        fig_format: str = "png",
        font_size: int = 11,
        dpi: int = 300,
    ) -> None:
        self.language   = language.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_format = fig_format.lower().lstrip(".")
        self.dpi        = dpi

        _setup_style(font_size)

        # 已生成图表注册表 {name: (fig, filename)}
        self._figures: Dict[str, Tuple[plt.Figure, str]] = {}

        logger.info(
            "PlotBuilder 初始化 [language=%s, output_dir=%s, format=%s]",
            language, output_dir, fig_format,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────────────────────────────────

    def _register(self, name: str, fig: plt.Figure, filename: str) -> None:
        """注册已生成的图表。"""
        self._figures[name] = (fig, filename)

    def _t(self, cn_text: str, en_text: str) -> str:
        """根据当前语言返回对应文本。"""
        return cn_text if self.language == "cn" else en_text

    @staticmethod
    def _add_value_labels(
        ax: plt.Axes,
        bars,
        fmt: str = "{:.4f}",
        fontsize: int = 9,
        color: str = "black",
        va: str = "bottom",
        offset: float = 0.002,
    ) -> None:
        """在条形图各柱顶部添加数值标签。"""
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + offset,
                fmt.format(height),
                ha="center", va=va,
                fontsize=fontsize, color=color,
            )

    @staticmethod
    def _style_ax(ax: plt.Axes, title: str = "", xlabel: str = "",
                  ylabel: str = "") -> None:
        """为 Axes 统一设置标题与坐标轴标签。"""
        if title:
            ax.set_title(title, fontweight="bold", pad=10)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. 权重条形图
    # ──────────────────────────────────────────────────────────────────────────

    def plot_weights(
        self,
        weights: Union[np.ndarray, List[float]],
        labels: List[str],
        method_name: str = "",
        figsize: Tuple[float, float] = (8, 4),
        color_gradient: bool = True,
        show_value: bool = True,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制指标权重条形图。

        Parameters
        ----------
        weights : array-like
            权重向量。
        labels : List[str]
            指标名称列表。
        method_name : str
            赋权方法名称（用于标题）。
        figsize : tuple
            图片尺寸（英寸）。
        color_gradient : bool
            是否按权重大小渐变着色。
        show_value : bool
            是否在柱顶显示数值。
        filename : str, optional
            输出文件名（不含格式后缀），默认自动生成。

        Returns
        -------
        (fig, ax)
        """
        w = np.asarray(weights)
        n = len(labels)

        fig, ax = plt.subplots(figsize=figsize)

        # 颜色
        if color_gradient:
            norm_w   = (w - w.min()) / (w.max() - w.min() + 1e-10)
            colors   = [_CMAP_BLUE(v) for v in norm_w]
        else:
            colors = _PALETTE_BLUE[:n] if n <= 5 else _PALETTE_QUAL

        x_pos = np.arange(n)
        bars  = ax.bar(x_pos, w, color=colors, width=0.6, edgecolor="white",
                       linewidth=0.8, zorder=3)

        if show_value:
            self._add_value_labels(ax, bars, fmt="{:.4f}", fontsize=9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30 if n > 6 else 0, ha="right")

        title = self._t(
            f"{'【' + method_name + '】' if method_name else ''}指标权重分布",
            f"{'[' + method_name + '] ' if method_name else ''}Indicator Weights",
        )
        self._style_ax(
            ax, title=title,
            xlabel=self._t("指标", "Indicator"),
            ylabel=self._t("权重", "Weight"),
        )
        ax.set_ylim(0, w.max() * 1.18)

        # 水平参考线（均等权重）
        ax.axhline(1.0 / n, color="red", linestyle="--", linewidth=0.8,
                   alpha=0.6, label=self._t(f"均等权重 (1/{n})", f"Equal (1/{n})"))
        ax.legend(loc="upper right")

        plt.tight_layout()
        fname = filename or f"weights_{method_name or 'eval'}"
        self._register(fname, fig, f"{fname}.{self.fig_format}")
        logger.debug("权重条形图已生成 [n=%d]", n)
        return fig, ax

    def plot_weight_comparison(
        self,
        weights_dict: Dict[str, np.ndarray],
        labels: List[str],
        figsize: Tuple[float, float] = (10, 5),
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        多种赋权方法权重对比条形图（分组柱状图）。

        Parameters
        ----------
        weights_dict : Dict[str, array-like]
            键为方法名称，值为权重向量。
        labels : List[str]
            指标名称列表。
        """
        methods = list(weights_dict.keys())
        n_methods = len(methods)
        n_inds    = len(labels)

        fig, ax = plt.subplots(figsize=figsize)
        x_pos   = np.arange(n_inds)
        width   = 0.8 / n_methods

        for k, (method, w) in enumerate(weights_dict.items()):
            w_arr = np.asarray(w)
            offset = (k - n_methods / 2 + 0.5) * width
            color  = _PALETTE_QUAL[k % len(_PALETTE_QUAL)]
            bars   = ax.bar(x_pos + offset, w_arr, width=width * 0.9,
                            label=method, color=color, edgecolor="white",
                            linewidth=0.6, zorder=3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30 if n_inds > 6 else 0, ha="right")
        self._style_ax(
            ax,
            title=self._t("多方法权重对比", "Multi-Method Weight Comparison"),
            xlabel=self._t("指标", "Indicator"),
            ylabel=self._t("权重", "Weight"),
        )
        ax.legend(loc="upper right", ncol=min(n_methods, 3))
        plt.tight_layout()

        fname = filename or "weight_comparison"
        self._register(fname, fig, f"{fname}.{self.fig_format}")
        return fig, ax

    # ──────────────────────────────────────────────────────────────────────────
    # 2. 综合得分排序图
    # ──────────────────────────────────────────────────────────────────────────

    def plot_ranking(
        self,
        scores: Union[np.ndarray, List[float]],
        labels: List[str],
        method_name: str = "TOPSIS",
        score_label: str = "",
        figsize: Tuple[float, float] = (9, 5),
        highlight_top: int = 3,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制评价对象综合得分水平条形图（按得分降序排列）。

        Parameters
        ----------
        scores : array-like
            各评价对象综合得分。
        labels : List[str]
            评价对象名称（与 scores 对应）。
        method_name : str
            评价方法名（用于标题和 X 轴标签）。
        score_label : str
            得分含义描述（可选）。
        highlight_top : int
            高亮前 N 名（默认前 3 名）。
        filename : str, optional
            输出文件名。

        Returns
        -------
        (fig, ax)
        """
        s      = np.asarray(scores, dtype=float)
        order  = np.argsort(s)              # 升序（水平条形图从下往上画）
        s_sorted = s[order]
        l_sorted = [labels[i] for i in order]
        n = len(labels)

        fig, ax = plt.subplots(figsize=figsize)

        colors = []
        for i in range(n):
            rank = n - i                    # 水平条形图最上面排名第1
            if rank <= highlight_top:
                intensity = 1.0 - (rank - 1) / max(highlight_top, 1) * 0.4
                colors.append(_CMAP_BLUE(intensity))
            else:
                colors.append("#b8d4ea")

        bars = ax.barh(range(n), s_sorted, color=colors,
                       edgecolor="white", linewidth=0.8, height=0.65, zorder=3)

        # 数值标签
        for i, (bar, val) in enumerate(zip(bars, s_sorted)):
            ax.text(
                val + s_sorted.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="left", fontsize=9,
            )

        ax.set_yticks(range(n))
        ax.set_yticklabels(l_sorted)
        ax.set_xlim(0, s_sorted.max() * 1.18)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.grid(axis="y", linestyle="", alpha=0)

        sl = score_label or self._t(f"{method_name} 综合得分", f"{method_name} Score")
        self._style_ax(
            ax,
            title=self._t("综合评价结果排序", "Comprehensive Evaluation Ranking"),
            xlabel=sl,
            ylabel=self._t("评价对象", "Object"),
        )

        # 图例（高亮说明）
        patch = mpatches.Patch(
            color=_CMAP_BLUE(1.0),
            label=self._t(f"前{highlight_top}名", f"Top {highlight_top}"),
        )
        ax.legend(handles=[patch], loc="lower right")

        plt.tight_layout()
        fname = filename or f"ranking_{method_name.lower()}"
        self._register(fname, fig, f"{fname}.{self.fig_format}")
        logger.debug("排序图已生成 [n=%d, method=%s]", n, method_name)
        return fig, ax

    # ──────────────────────────────────────────────────────────────────────────
    # 3. 雷达图
    # ──────────────────────────────────────────────────────────────────────────

    def plot_radar(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: List[str],
        objects: List[str],
        figsize: Tuple[float, float] = (7, 7),
        fill_alpha: float = 0.15,
        max_objects: int = 8,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制多对象雷达图（蜘蛛网图）。

        Parameters
        ----------
        data : array-like, shape (n_objects, n_indicators)
            各评价对象在各指标上的值（建议已归一化至 [0,1]）。
        labels : List[str]
            指标名称列表（雷达轴标签）。
        objects : List[str]
            评价对象名称列表。
        figsize : tuple
            图片尺寸。
        fill_alpha : float
            填充透明度。
        max_objects : int
            最多绘制对象数（超出则取前 N 个）。
        filename : str, optional
            输出文件名。

        Returns
        -------
        (fig, ax)
        """
        mat = np.asarray(data)
        n_ind = len(labels)
        n_obj = min(len(objects), max_objects)
        mat   = mat[:n_obj]
        objs  = objects[:n_obj]

        # 雷达图角度
        angles = np.linspace(0, 2 * np.pi, n_ind, endpoint=False).tolist()
        angles += angles[:1]                            # 闭合

        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw={"projection": "polar"})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # 轴标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)

        # 径向网格
        ax.set_rlabel_position(30)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
        ax.set_ylim(0, 1)

        for i, (obj, row) in enumerate(zip(objs, mat)):
            values = row.tolist() + row[:1].tolist()   # 闭合
            color  = _PALETTE_QUAL[i % len(_PALETTE_QUAL)]
            ax.plot(angles, values, "o-", linewidth=1.8,
                    color=color, label=obj, markersize=4)
            ax.fill(angles, values, alpha=fill_alpha, color=color)

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.15),
            framealpha=0.85,
        )
        ax.set_title(
            self._t("各评价对象综合雷达图", "Comprehensive Radar Chart"),
            fontweight="bold", pad=20,
        )

        plt.tight_layout()
        fname = filename or "radar_chart"
        self._register(fname, fig, f"{fname}.{self.fig_format}")
        logger.debug("雷达图已生成 [n_obj=%d, n_ind=%d]", n_obj, n_ind)
        return fig, ax

    # ──────────────────────────────────────────────────────────────────────────
    # 4. 相关性热力图
    # ──────────────────────────────────────────────────────────────────────────

    def plot_correlation_heatmap(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: List[str],
        figsize: Tuple[float, float] = (8, 6),
        annot: bool = True,
        method: str = "pearson",
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制指标相关性热力图。

        Parameters
        ----------
        data : array-like, shape (n_samples, n_indicators)
            原始指标数据矩阵。
        labels : List[str]
            指标名称。
        annot : bool
            是否在格中显示相关系数值。
        method : str
            ``"pearson"`` / ``"spearman"`` / ``"kendall"``。
        filename : str, optional
            输出文件名。
        """
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data, columns=labels)

        corr = df.corr(method=method)

        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # 仅显示下三角

        sns.heatmap(
            corr,
            mask=mask,
            annot=annot,
            fmt=".3f",
            cmap=_CMAP_RWG,
            vmin=-1, vmax=1,
            center=0,
            linewidths=0.5,
            linecolor="white",
            square=True,
            ax=ax,
            annot_kws={"size": 9},
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        title = self._t(
            f"指标{method.capitalize()}相关性矩阵",
            f"Indicator {method.capitalize()} Correlation Matrix",
        )
        ax.set_title(title, fontweight="bold", pad=12)

        plt.tight_layout()
        fname = filename or "corr_heatmap"
        self._register(fname, fig, f"{fname}.{self.fig_format}")
        logger.debug("相关性热力图已生成 [method=%s]", method)
        return fig, ax

    # ──────────────────────────────────────────────────────────────────────────
    # 5. 灵敏度分析折线图
    # ──────────────────────────────────────────────────────────────────────────

    def plot_sensitivity(
        self,
        sens_result: Dict[str, Any],
        objects: List[str],
        figsize_per_row: Tuple[float, float] = (6, 3.5),
        max_cols: int = 3,
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        绘制权重灵敏度分析折线图（每个指标一个子图）。

        Parameters
        ----------
        sens_result : dict
            ``weight_sensitivity()`` 的返回值，包含：

            * ``"perturb_ratios"`` : 1-D array — 扰动比例
            * ``"rank_records"``   : Dict[str, 2-D array] — 指标名→排名矩阵

        objects : List[str]
            评价对象名称列表（用于图例）。
        figsize_per_row : tuple
            每行每个子图的尺寸。
        max_cols : int
            每行最多子图数。
        filename : str, optional
            输出文件名（前缀）。

        Returns
        -------
        (fig, axes)
        """
        rank_records  = sens_result.get("rank_records", {})
        perturb_ratios = np.asarray(sens_result.get("perturb_ratios", []))
        indicator_names = list(rank_records.keys())
        n_ind  = len(indicator_names)

        if n_ind == 0:
            logger.warning("灵敏度结果为空，跳过绘图。")
            return None, None

        n_cols = min(n_ind, max_cols)
        n_rows = math.ceil(n_ind / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_row[0] * n_cols,
                     figsize_per_row[1] * n_rows),
        )
        axes_flat = np.array(axes).flatten()

        x_pct = perturb_ratios * 100  # 转为百分比

        for idx, ind_name in enumerate(indicator_names):
            ax   = axes_flat[idx]
            rm   = rank_records[ind_name]      # (steps, n_objects)
            n_obj = rm.shape[1]

            for obj_i in range(n_obj):
                color = _PALETTE_QUAL[obj_i % len(_PALETTE_QUAL)]
                ax.plot(
                    x_pct, rm[:, obj_i],
                    color=color, linewidth=1.6,
                    label=objects[obj_i] if obj_i < len(objects) else f"Obj{obj_i+1}",
                    marker="o", markersize=3, alpha=0.85,
                )

            ax.set_xlabel(
                self._t("权重扰动 (%)", "Weight Perturbation (%)"), fontsize=9
            )
            ax.set_ylabel(self._t("综合排名", "Rank"), fontsize=9)
            ax.set_title(
                self._t(f"【{ind_name}】权重灵敏度", f"[{ind_name}] Sensitivity"),
                fontsize=10, fontweight="bold",
            )
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.invert_yaxis()               # 排名1在顶部
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            if idx == 0:
                ax.legend(loc="upper right", fontsize=8,
                          ncol=max(1, n_obj // 4))

        # 隐藏多余子图
        for idx in range(n_ind, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(
            self._t("权重灵敏度分析（OAT）", "Weight Sensitivity Analysis (OAT)"),
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        fname = filename or "sensitivity_oat"
        self._register(fname, fig, f"{fname}.{self.fig_format}")
        logger.debug("灵敏度图已生成 [n_indicators=%d]", n_ind)
        return fig, axes

    # ──────────────────────────────────────────────────────────────────────────
    # 6. PCA 碎石图与贡献率图
    # ──────────────────────────────────────────────────────────────────────────

    def plot_pca_scree(
        self,
        eigenvalues: Union[np.ndarray, List[float]],
        explained_var: Union[np.ndarray, List[float]],
        n_selected: int = 2,
        figsize: Tuple[float, float] = (8, 4),
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        绘制 PCA 碎石图（特征值折线）及累计方差贡献率双轴图。

        Parameters
        ----------
        eigenvalues : array-like
            各主成分特征值。
        explained_var : array-like
            各主成分方差贡献率（%）。
        n_selected : int
            最终选取的主成分数（垂直参考线）。
        filename : str, optional
            输出文件名。

        Returns
        -------
        (fig, (ax1, ax2))
        """
        ev  = np.asarray(eigenvalues)
        var = np.asarray(explained_var)
        cum_var = np.cumsum(var)
        n   = len(ev)
        x   = np.arange(1, n + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # ── 左：碎石图 ─────────────────────────────────────────────────────
        ax1.plot(x, ev, "o-", color=_PALETTE_BLUE[1], linewidth=2,
                 markersize=6, zorder=3)
        ax1.axvline(n_selected, color="red", linestyle="--",
                    linewidth=1.2, alpha=0.7,
                    label=self._t(f"选取 {n_selected} 个", f"Select {n_selected}"))
        ax1.set_xticks(x)
        ax1.set_xlabel(self._t("主成分编号", "Principal Component"))
        ax1.set_ylabel(self._t("特征值", "Eigenvalue"))
        ax1.set_title(self._t("碎石图", "Scree Plot"), fontweight="bold")
        ax1.legend(fontsize=9)

        # ── 右：方差贡献率 ──────────────────────────────────────────────────
        bars = ax2.bar(x, var, color=_CMAP_BLUE(np.linspace(0.3, 0.9, n)),
                       edgecolor="white", width=0.65, zorder=3,
                       label=self._t("单项贡献率", "Individual"))
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x, cum_var, "s--", color="#e07b39", linewidth=1.8,
                      markersize=5, label=self._t("累计贡献率", "Cumulative"))
        ax2_twin.axhline(85, color="#c00000", linestyle=":", linewidth=1,
                         alpha=0.7, label="85%")
        ax2_twin.set_ylim(0, 105)
        ax2_twin.set_ylabel(self._t("累计方差贡献率 (%)", "Cumulative Var. (%)"))

        ax2.set_xticks(x)
        ax2.set_xlabel(self._t("主成分编号", "Principal Component"))
        ax2.set_ylabel(self._t("方差贡献率 (%)", "Variance Explained (%)"))
        ax2.set_title(self._t("方差贡献率", "Variance Explained"), fontweight="bold")

        lines1, labs1 = ax2.get_legend_handles_labels()
        lines2, labs2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="center right")

        plt.tight_layout()
        fname = filename or "pca_scree"
        self._register(fname, fig, f"{fname}.{self.fig_format}")
        logger.debug("PCA 碎石图已生成 [n_comp=%d]", n_selected)
        return fig, np.array([ax1, ax2])

    # ──────────────────────────────────────────────────────────────────────────
    # 7. 综合得分气泡散点图（2D 投影）
    # ──────────────────────────────────────────────────────────────────────────

    def plot_score_scatter(
        self,
        scores_x: Union[np.ndarray, List[float]],
        scores_y: Union[np.ndarray, List[float]],
        labels: List[str],
        size_weight: Optional[Union[np.ndarray, List[float]]] = None,
        method_x: str = "Method A",
        method_y: str = "Method B",
        figsize: Tuple[float, float] = (7, 6),
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制两种评价方法得分的二维散点图（用于方法对比）。

        Parameters
        ----------
        scores_x : array-like
            X 轴评价方法得分。
        scores_y : array-like
            Y 轴评价方法得分。
        labels : List[str]
            评价对象标注。
        size_weight : array-like, optional
            气泡大小权重（若提供则绘制气泡图）。
        method_x, method_y : str
            两种方法名称（坐标轴标签）。
        filename : str, optional
            输出文件名。
        """
        sx = np.asarray(scores_x, dtype=float)
        sy = np.asarray(scores_y, dtype=float)

        fig, ax = plt.subplots(figsize=figsize)

        if size_weight is not None:
            sw = np.asarray(size_weight, dtype=float)
            sw = (sw - sw.min()) / (sw.max() - sw.min() + 1e-10) * 400 + 50
        else:
            sw = 80

        scatter = ax.scatter(
            sx, sy, s=sw, c=sx + sy,
            cmap=_CMAP_BLUE, edgecolors="#1f4e79", linewidths=0.7,
            alpha=0.85, zorder=3,
        )
        plt.colorbar(scatter, ax=ax,
                     label=self._t("综合得分（两方法之和）",
                                   "Combined Score (sum of both)"))

        # 添加对象标注
        for i, txt in enumerate(labels):
            ax.annotate(
                txt,
                (sx[i], sy[i]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
                alpha=0.85,
            )

        # 对角参考线
        lim_min = min(sx.min(), sy.min()) * 0.95
        lim_max = max(sx.max(), sy.max()) * 1.05
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                "r--", linewidth=0.8, alpha=0.5,
                label=self._t("对角线（两方法一致）", "Diagonal (agreement)"))

        self._style_ax(
            ax,
            title=self._t("两种评价方法得分对比", "Score Comparison of Two Methods"),
            xlabel=f"{method_x} " + self._t("得分", "Score"),
            ylabel=f"{method_y} " + self._t("得分", "Score"),
        )
        ax.legend(fontsize=9)
        plt.tight_layout()

        fname = filename or f"scatter_{method_x}_{method_y}".replace(" ", "_").lower()
        self._register(fname, fig, f"{fname}.{self.fig_format}")
        logger.debug("得分散点图已生成")
        return fig, ax

    # ──────────────────────────────────────────────────────────────────────────
    # 8. AHP 判断矩阵热力图
    # ──────────────────────────────────────────────────────────────────────────

    def plot_ahp_heatmap(
        self,
        judgment_matrix: np.ndarray,
        labels: List[str],
        figsize: Tuple[float, float] = (7, 5),
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制 AHP 判断矩阵热力图（对数刻度着色）。

        Parameters
        ----------
        judgment_matrix : 2-D array-like
            AHP 正互反判断矩阵。
        labels : List[str]
            指标名称列表。
        filename : str, optional
            输出文件名。
        """
        jm  = np.asarray(judgment_matrix, dtype=float)
        log_jm = np.log2(np.clip(jm, 1e-6, None))   # 以 log2 着色

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(log_jm, cmap=_CMAP_RWG, vmin=-4, vmax=4, aspect="auto")

        # 色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.04)
        cbar.set_label(self._t("log₂(aᵢⱼ)", "log₂(aᵢⱼ)"), fontsize=9)
        cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
        cbar.set_ticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])

        # 格内标注原始值
        n = jm.shape[0]
        for i in range(n):
            for j in range(n):
                val = jm[i, j]
                txt = (f"1/{int(round(1/val))}" if val < 1 and val > 0
                       else f"{val:.1f}" if val != int(val) else str(int(val)))
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if abs(log_jm[i, j]) > 1.5 else "black")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(
            self._t("AHP 判断矩阵热力图", "AHP Judgment Matrix Heatmap"),
            fontweight="bold", pad=12,
        )
        plt.tight_layout()

        fname = filename or "ahp_heatmap"
        self._register(fname, fig, f"{fname}.{self.fig_format}")
        logger.debug("AHP 热力图已生成 [n=%d]", n)
        return fig, ax

    # ──────────────────────────────────────────────────────────────────────────
    # 9. 时序动态评价折线图
    # ──────────────────────────────────────────────────────────────────────────

    def plot_dynamic_trend(
        self,
        scores_time: Union[np.ndarray, pd.DataFrame],
        objects: List[str],
        time_labels: List[str],
        figsize: Tuple[float, float] = (10, 5),
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制各评价对象随时间变化的综合得分折线图（动态评价）。

        Parameters
        ----------
        scores_time : array-like, shape (n_objects, n_time)
            各对象在各时间点的综合得分。
        objects : List[str]
            评价对象名称。
        time_labels : List[str]
            时间点标签（如年份）。
        filename : str, optional
            输出文件名。
        """
        mat = np.asarray(scores_time, dtype=float)
        n_obj, n_time = mat.shape
        x = np.arange(n_time)

        fig, ax = plt.subplots(figsize=figsize)

        for i, obj in enumerate(objects[:n_obj]):
            color  = _PALETTE_QUAL[i % len(_PALETTE_QUAL)]
            marker = ["o", "s", "^", "D", "v", "p", "*", "h"][i % 8]
            ax.plot(
                x, mat[i],
                color=color, linewidth=2, label=obj,
                marker=marker, markersize=6, zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(time_labels, rotation=30 if n_time > 6 else 0,
                           ha="right")
        self._style_ax(
            ax,
            title=self._t("综合评价得分时序变化", "Dynamic Evaluation Score Trend"),
            xlabel=self._t("时间", "Time"),
            ylabel=self._t("综合得分", "Composite Score"),
        )
        ax.legend(loc="upper left", ncol=max(1, n_obj // 5))
        plt.tight_layout()

        fname = filename or "dynamic_trend"
        self._register(fname, fig, f"{fname}.{self.fig_format}")
        logger.debug("动态评价折线图已生成 [n_obj=%d, n_time=%d]", n_obj, n_time)
        return fig, ax

    # ──────────────────────────────────────────────────────────────────────────
    # 10. 熵权法信息熵可视化
    # ──────────────────────────────────────────────────────────────────────────

    def plot_entropy_analysis(
        self,
        entropy_result: Dict[str, Any],
        labels: List[str],
        figsize: Tuple[float, float] = (10, 4),
        filename: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        绘制熵权法分析三联图：信息熵 | 差异系数 | 权重。

        Parameters
        ----------
        entropy_result : dict
            包含 ``"entropy"``, ``"diversity"``, ``"weights"`` 的字典。
        labels : List[str]
            指标名称列表。
        figsize : tuple
            整体图片尺寸。
        filename : str, optional
            输出文件名。

        Returns
        -------
        (fig, axes)
        """
        e_j = np.asarray(entropy_result.get("entropy",  []))
        d_j = np.asarray(entropy_result.get("diversity", 1 - e_j))
        w_j = np.asarray(entropy_result.get("weights",  []))
        n   = len(labels)
        x   = np.arange(n)

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        datasets = [
            (e_j, self._t("信息熵 $e_j$", "Entropy $e_j$"),   _PALETTE_BLUE[2]),
            (d_j, self._t("差异系数 $d_j$", "Diversity $d_j$"), _PALETTE_BLUE[1]),
            (w_j, self._t("权重 $w_j$", "Weight $w_j$"),       _PALETTE_BLUE[0]),
        ]

        for ax, (vals, ylabel, color) in zip(axes, datasets):
            bars = ax.bar(x, vals, color=color, width=0.6,
                          edgecolor="white", linewidth=0.8, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_ylim(0, vals.max() * 1.2)
            self._add_value_labels(ax, bars, fmt="{:.4f}", fontsize=8, offset=vals.max() * 0.01)

        axes[0].set_title(self._t("信息熵", "Entropy"), fontweight="bold")
        axes[1].set_title(self._t("差异系数", "Diversity"), fontweight="bold")
        axes[2].set_title(self._t("指标权重", "Weights"), fontweight="bold")

        fig.suptitle(
            self._t("熵权法分析结果", "Entropy Weight Method Analysis"),
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()

        fname = filename or "entropy_analysis"
        self._register(fname, fig, f"{fname}.{self.fig_format}")
        logger.debug("熵权法分析图已生成 [n=%d]", n)
        return fig, axes

    # ──────────────────────────────────────────────────────────────────────────
    # 批量保存
    # ──────────────────────────────────────────────────────────────────────────

    def save_all(self, close_after: bool = True) -> List[Path]:
        """
        将所有已注册的图表批量保存到输出目录。

        Parameters
        ----------
        close_after : bool
            保存后是否关闭图表（释放内存）。

        Returns
        -------
        List[Path]
            所有已保存文件的路径列表。
        """
        saved_paths = []
        if not self._figures:
            logger.warning("没有已生成的图表，save_all() 跳过。")
            return saved_paths

        for name, (fig, filename) in self._figures.items():
            out_path = self.output_dir / filename
            try:
                fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight",
                            facecolor="white")
                saved_paths.append(out_path)
                logger.info("图表已保存: %s", out_path)
            except Exception as exc:
                logger.error("图表保存失败 [name=%s]: %s", name, exc)
            if close_after:
                plt.close(fig)

        if close_after:
            self._figures.clear()

        logger.info("批量保存完成，共 %d 张图表。", len(saved_paths))
        return saved_paths

    def save_figure(
        self,
        name: str,
        filename: Optional[str] = None,
        close_after: bool = False,
    ) -> Optional[Path]:
        """
        保存指定名称的单张图表。

        Parameters
        ----------
        name : str
            图表注册名（与 plot_*() 中的 filename 参数一致）。
        filename : str, optional
            覆盖输出文件名。
        close_after : bool
            保存后是否关闭图表。

        Returns
        -------
        Path or None
        """
        if name not in self._figures:
            logger.warning("图表 %r 不存在，跳过保存。", name)
            return None

        fig, default_filename = self._figures[name]
        out_filename = filename or default_filename
        out_path = self.output_dir / out_filename
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight",
                    facecolor="white")
        logger.info("图表已保存: %s", out_path)
        if close_after:
            plt.close(fig)
            del self._figures[name]
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # 辅助属性
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def figure_names(self) -> List[str]:
        """已注册图表的名称列表。"""
        return list(self._figures.keys())

    @property
    def figure_count(self) -> int:
        """已注册图表总数。"""
        return len(self._figures)

    def clear(self) -> None:
        """关闭并清除所有已注册图表。"""
        for name, (fig, _) in self._figures.items():
            plt.close(fig)
        self._figures.clear()
        logger.debug("PlotBuilder 图表缓存已清空")

    def __repr__(self) -> str:
        return (
            f"PlotBuilder(language={self.language!r}, "
            f"figures={self.figure_count}, "
            f"output_dir={str(self.output_dir)!r})"
        )