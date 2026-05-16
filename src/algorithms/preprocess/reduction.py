# src/algorithms/preprocess/reduction.py
"""
降维与相关性分析模块
- CorrelationAnalyzer : 计算相关矩阵、识别高相关对（非 BaseMethod，工具类）
- PCAReducer          : 主成分分析（PCA），支持按方差贡献率自动选取主成分数
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import PreprocessBase
from ..base import MethodCategory, MethodResult
from ...utils.logging_config import get_logger

logger = get_logger("auto_eval.reduction")


# ============================================================
#  相关性分析工具（非 BaseMethod）
# ============================================================

@dataclass
class CorrelationReport:
    """
    相关性分析报告数据类。

    Attributes
    ----------
    corr_matrix       : 完整相关矩阵
    high_corr_pairs   : 高相关对列表 [(col_i, col_j, r)]
    max_corr          : 最大相关系数绝对值
    mean_corr         : 所有非对角元素的平均相关系数绝对值
    n_high_corr_pairs : 高相关对数量
    threshold         : 判断高相关的阈值
    recommendation    : 自动生成的处理建议
    """
    corr_matrix:       pd.DataFrame            = field(default_factory=pd.DataFrame)
    high_corr_pairs:   list[tuple[str, str, float]] = field(default_factory=list)
    max_corr:          float                   = 0.0
    mean_corr:         float                   = 0.0
    n_high_corr_pairs: int                     = 0
    threshold:         float                   = 0.85
    recommendation:    str                     = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_corr":          self.max_corr,
            "mean_corr":         self.mean_corr,
            "n_high_corr_pairs": self.n_high_corr_pairs,
            "threshold":         self.threshold,
            "recommendation":    self.recommendation,
            "high_corr_pairs": [
                {"col_i": a, "col_j": b, "r": round(r, 4)}
                for a, b, r in self.high_corr_pairs
            ],
        }


class CorrelationAnalyzer:
    """
    指标相关性分析工具。

    主要功能：
    1. 计算 Pearson / Spearman 相关矩阵
    2. 识别高相关性指标对（|r| > threshold）
    3. 生成可视化数据（热力图矩阵）
    4. 自动给出处理建议（是否需要 PCA 降维）

    不继承 BaseMethod，作为独立工具类调用。
    """

    def __init__(
        self,
        threshold: float = 0.85,
        method: str = "pearson",
    ):
        """
        Parameters
        ----------
        threshold : float
            高相关性判断阈值（|r| > threshold 视为高相关），范围 (0, 1)
        method : str
            相关性计算方法：'pearson'（线性）或 'spearman'（秩相关）
        """
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold={threshold} 需在 (0, 1) 范围内")
        if method not in ("pearson", "spearman"):
            raise ValueError(f"method='{method}' 无效，合法值: 'pearson', 'spearman'")
        self.threshold = threshold
        self.method    = method
        self._report: Optional[CorrelationReport] = None

    def analyze(self, data: pd.DataFrame) -> CorrelationReport:
        """
        执行相关性分析。

        Parameters
        ----------
        data : pd.DataFrame
            数值型数据，行为对象，列为指标

        Returns
        -------
        CorrelationReport
        """
        if data.empty:
            raise ValueError("输入数据不能为空")

        # 仅保留数值列
        num_data = data.select_dtypes(include=["number"])
        if num_data.shape[1] < 2:
            logger.warning("数值列数量 < 2，无法计算相关性")
            return CorrelationReport()

        # 计算相关矩阵
        corr_matrix = num_data.corr(method=self.method)

        # 提取高相关对（上三角，排除对角线）
        cols = corr_matrix.columns.tolist()
        high_corr_pairs: list[tuple[str, str, float]] = []
        abs_off_diagonal: list[float] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr_matrix.iloc[i, j]
                if not np.isnan(r):
                    abs_off_diagonal.append(abs(r))
                    if abs(r) > self.threshold:
                        high_corr_pairs.append((cols[i], cols[j], r))

        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        max_corr  = max(abs_off_diagonal) if abs_off_diagonal else 0.0
        mean_corr = float(np.mean(abs_off_diagonal)) if abs_off_diagonal else 0.0

        # 生成建议
        recommendation = self._generate_recommendation(
            n_high=len(high_corr_pairs),
            max_corr=max_corr,
            n_cols=num_data.shape[1],
        )

        report = CorrelationReport(
            corr_matrix       = corr_matrix,
            high_corr_pairs   = high_corr_pairs,
            max_corr          = round(max_corr, 4),
            mean_corr         = round(mean_corr, 4),
            n_high_corr_pairs = len(high_corr_pairs),
            threshold         = self.threshold,
            recommendation    = recommendation,
        )
        self._report = report
        logger.info(
            f"相关性分析完成 | {self.method} | "
            f"最大相关系数: {max_corr:.4f} | "
            f"高相关对数: {len(high_corr_pairs)}"
        )
        return report

    def get_high_corr_dataframe(self) -> pd.DataFrame:
        """
        以 DataFrame 形式返回高相关对。

        Returns
        -------
        pd.DataFrame
            列：指标A, 指标B, 相关系数, 绝对值
        """
        if self._report is None:
            raise RuntimeError("请先调用 analyze() 方法")
        rows = [
            {
                "指标A": a,
                "指标B": b,
                "相关系数": round(r, 4),
                "绝对值": round(abs(r), 4),
            }
            for a, b, r in self._report.high_corr_pairs
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _generate_recommendation(
        n_high: int,
        max_corr: float,
        n_cols: int,
    ) -> str:
        """根据分析结果自动生成处理建议"""
        if n_high == 0:
            return (
                f"指标间相关性较低（最大 |r| = {max_corr:.4f}），"
                "各指标信息独立，建议直接使用原始指标进行评价。"
            )
        ratio = n_high / (n_cols * (n_cols - 1) / 2)
        if ratio > 0.3 or max_corr > 0.95:
            return (
                f"存在 {n_high} 对高相关指标（最大 |r| = {max_corr:.4f}，"
                f"占比 {ratio:.1%}），指标间存在明显信息重叠。"
                "强烈建议使用 PCA 降维或 CRITIC 法以消除共线性影响。"
            )
        return (
            f"存在 {n_high} 对高相关指标（最大 |r| = {max_corr:.4f}），"
            "建议考虑使用 CRITIC 赋权法或 PCA 降维，以减弱信息重叠带来的偏差。"
        )

    def tex_description(self) -> str:
        if self._report is None:
            return r"\subsubsection{相关性分析}\n尚未执行分析。"

        rpt = self._report
        method_name = "Pearson" if self.method == "pearson" else "Spearman 秩"

        high_corr_tex = ""
        if rpt.high_corr_pairs:
            rows_tex = " \\\\\n    ".join(
                rf"{a} & {b} & {r:.4f}"
                for a, b, r in rpt.high_corr_pairs[:10]  # 最多展示10对
            )
            high_corr_tex = rf"""
主要高相关指标对如表所示（部分），
\begin{{center}}
\begin{{tabular}}{{lll}}
\toprule
指标A & 指标B & 相关系数 \\
\midrule
    {rows_tex} \\
\bottomrule
\end{{tabular}}
\end{{center}}"""

        return rf"""
\subsubsection{{指标相关性分析}}
采用 {method_name} 相关系数对 {len(rpt.corr_matrix.columns)} 个评价指标进行相关性检验，
相关系数矩阵的最大值为 ${rpt.max_corr:.4f}$，平均值为 ${rpt.mean_corr:.4f}$。
以阈值 $|r| > {rpt.threshold}$ 识别高相关指标对，共发现 {rpt.n_high_corr_pairs} 对。
{high_corr_tex}
\textbf{{分析建议}}：{rpt.recommendation}
"""


# ============================================================
#  主成分分析（PCA）
# ============================================================

class PCAReducer(PreprocessBase):
    """
    主成分分析（Principal Component Analysis）降维。

    功能
    ----
    1. 自动确定主成分数量（按累积方差贡献率）
    2. 计算主成分载荷矩阵（各原始指标对主成分的贡献）
    3. 输出变换后的主成分得分矩阵
    4. 生成碎石图数据、累积方差贡献率表

    Parameters（fit 传入）
    ----------------------
    n_components      : int | None，指定主成分数，None 则按 variance_ratio 自动确定
    variance_ratio    : float，累积方差贡献率阈值（自动确定模式），默认 0.85
    standardize_first : bool，是否在 PCA 前对数据做 Z-score 标准化（强烈推荐 True）
    """

    METHOD_NAME_ZH = "主成分分析"
    METHOD_NAME_EN = "PCAReducer"
    METHOD_ABBR    = "PCA"
    CATEGORY       = MethodCategory.REDUCTION

    def __init__(self, language: str = "zh"):
        super().__init__(language)
        self._n_components:      Optional[int]   = None
        self._variance_ratio:    float           = 0.85
        self._standardize_first: bool            = True
        # 训练后的 sklearn PCA 对象
        self._pca_model: Any = None

    # ── fit ────────────────────────────────────────────────

    def fit(
        self,
        data: pd.DataFrame,
        n_components: Optional[int] = None,
        variance_ratio: float = 0.85,
        standardize_first: bool = True,
        **kwargs: Any,
    ) -> "PCAReducer":
        """
        Parameters
        ----------
        data              : 数值 DataFrame（建议先经过 Z-score 标准化）
        n_components      : 保留的主成分数量；None 则由 variance_ratio 自动确定
        variance_ratio    : 累积方差贡献率阈值（0.5~1.0），默认 0.85
        standardize_first : True 则先进行 Z-score 标准化再做 PCA
        """
        self.validate_input(data)

        if n_components is not None and n_components < 1:
            raise ValueError(f"n_components={n_components} 必须 ≥ 1")
        if not (0.5 <= variance_ratio <= 1.0):
            raise ValueError(f"variance_ratio={variance_ratio} 需在 [0.5, 1.0]")

        _, num_cols = self._extract_numeric(data)
        if len(num_cols) < 2:
            raise ValueError("PCA 至少需要 2 个数值列")

        self._raw_data           = data.copy()
        self._indicator_names    = num_cols
        self._object_names       = list(data.index)
        self._n_components       = n_components
        self._variance_ratio     = variance_ratio
        self._standardize_first  = standardize_first
        self._is_fitted          = True

        self.logger.info(
            f"PCAReducer.fit | n_components={n_components} | "
            f"variance_ratio={variance_ratio} | "
            f"standardize={standardize_first}"
        )
        return self

    # ── compute ────────────────────────────────────────────

    def compute(self) -> MethodResult:
        """执行 PCA 分析。"""
        self._check_fitted("compute")
        t0 = self._start_timer()

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError(
                "PCA 降维需要 scikit-learn，请安装: pip install scikit-learn"
            )

        df_num = self._raw_data[self._indicator_names].copy()

        # ── 可选 Z-score 标准化 ────────────────────────────
        if self._standardize_first:
            scaler = StandardScaler()
            X = scaler.fit_transform(df_num.values)
        else:
            X = df_num.values.astype(float)

        n_samples, n_features = X.shape

        # ── 确定主成分数量 ─────────────────────────────────
        if self._n_components is not None:
            k = min(self._n_components, n_samples, n_features)
        else:
            # 先做完整 PCA，再按方差阈值裁剪
            pca_full = PCA(n_components=min(n_samples, n_features))
            pca_full.fit(X)
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            k = int(np.searchsorted(cumsum, self._variance_ratio) + 1)
            k = max(1, min(k, n_features))

        self.logger.info(f"PCA 选取主成分数: k={k}")

        # ── 正式 PCA ───────────────────────────────────────
        pca = PCA(n_components=k, random_state=42)
        scores_matrix = pca.fit_transform(X)
        self._pca_model = pca

        # ── 构建输出 DataFrame ─────────────────────────────
        pc_names = [f"PC{i+1}" for i in range(k)]
        scores_df = pd.DataFrame(
            scores_matrix,
            index=self._object_names,
            columns=pc_names,
        )

        # ── 载荷矩阵（指标 × 主成分）──────────────────────
        loadings_df = pd.DataFrame(
            pca.components_.T,               # shape: (n_features, k)
            index=self._indicator_names,
            columns=pc_names,
        )

        # ── 方差贡献率表 ───────────────────────────────────
        variance_df = self._build_variance_table(pca, k)

        # ── 碎石图数据（仅在完整 PCA 时计算）─────────────
        scree_data = self._build_scree_data(X, n_features)

        # ── 综合得分（以方差贡献率为权重的加权主成分得分）
        weights_arr = pca.explained_variance_ratio_ / pca.explained_variance_ratio_.sum()
        composite_score = pd.Series(
            scores_matrix.dot(weights_arr),
            index=self._object_names,
            name="PCA综合得分",
        )
        rankings = self._rank_scores(composite_score, ascending=False)

        result = self._build_result(
            scores   = composite_score,
            rankings = rankings,
            tables   = {
                "output":       scores_df,
                "loadings":     loadings_df,
                "variance":     variance_df,
                "scree":        scree_data,
            },
            scalars  = {
                "n_components_selected": k,
                "n_features_original":   n_features,
                "variance_ratio_target": self._variance_ratio,
                "variance_ratio_actual": float(
                    pca.explained_variance_ratio_.sum()
                ),
                "standardize_first": self._standardize_first,
            },
            metadata = {
                "output_data":  scores_df,   # 降维后数据
                "pca_model":    pca,
                "indicator_names_original": self._indicator_names,
            },
            elapsed  = self._stop_timer(t0),
        )
        self._result = result
        self.logger.info(
            f"PCA 完成 | k={k} | "
            f"累积方差: {pca.explained_variance_ratio_.sum():.4f} | "
            f"耗时 {result.elapsed_time:.3f}s"
        )
        return result

    # ── 内部工具 ───────────────────────────────────────────

    def _build_variance_table(
        self,
        pca: Any,
        k: int,
    ) -> pd.DataFrame:
        """构建方差贡献率汇总表"""
        ev_ratio    = pca.explained_variance_ratio_
        ev_values   = pca.explained_variance_
        cumulative  = np.cumsum(ev_ratio)

        rows = []
        for i in range(k):
            rows.append({
                "主成分":         f"PC{i+1}",
                "特征值":          round(float(ev_values[i]), 4),
                "方差贡献率":      f"{ev_ratio[i]:.2%}",
                "累积方差贡献率":  f"{cumulative[i]:.2%}",
            })
        return pd.DataFrame(rows)

    def _build_scree_data(
        self,
        X: np.ndarray,
        n_features: int,
    ) -> pd.DataFrame:
        """
        构建碎石图数据（完整特征值列表，用于绘图）。
        """
        try:
            from sklearn.decomposition import PCA as _PCA
            pca_full = _PCA(n_components=min(len(X), n_features))
            pca_full.fit(X)
            ev = pca_full.explained_variance_
            cr = pca_full.explained_variance_ratio_
        except Exception:
            return pd.DataFrame()

        return pd.DataFrame({
            "主成分编号":   list(range(1, len(ev) + 1)),
            "特征值":       np.round(ev, 4).tolist(),
            "方差贡献率":   np.round(cr, 4).tolist(),
            "累积贡献率":   np.round(np.cumsum(cr), 4).tolist(),
        })

    # ── 便捷属性 ───────────────────────────────────────────

    @property
    def n_components(self) -> int:
        """实际选取的主成分数（compute 后可用）"""
        return self.get_result().scalars["n_components_selected"]

    @property
    def loadings(self) -> pd.DataFrame:
        """载荷矩阵（compute 后可用）"""
        return self.get_result().tables["loadings"]

    @property
    def variance_table(self) -> pd.DataFrame:
        """方差贡献率表（compute 后可用）"""
        return self.get_result().tables["variance"]

    @property
    def pca_model(self) -> Any:
        """底层 sklearn PCA 模型（compute 后可用）"""
        return self.get_result().metadata["pca_model"]

    def transform(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        将新数据投影到已训练的主成分空间（需先 compute()）。

        Parameters
        ----------
        new_data : pd.DataFrame
            与训练数据相同列结构的新数据

        Returns
        -------
        pd.DataFrame
            主成分得分 DataFrame
        """
        if self._pca_model is None:
            raise RuntimeError("请先调用 compute() 方法")
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("需要 scikit-learn")

        X = new_data[self._indicator_names].values.astype(float)
        if self._standardize_first:
            # 使用训练集统计量标准化
            pca_meta = self.get_result().metadata
            scaler = StandardScaler()
            scaler.mean_ = self._raw_data[self._indicator_names].mean().values
            scaler.scale_ = self._raw_data[self._indicator_names].std().values
            X = (X - scaler.mean_) / (scaler.scale_ + 1e-12)

        scores = self._pca_model.transform(X)
        k = scores.shape[1]
        return pd.DataFrame(
            scores,
            index=new_data.index,
            columns=[f"PC{i+1}" for i in range(k)],
        )

    # ── summary / tex ──────────────────────────────────────

    def summary(self) -> str:
        r = self.get_result()
        lines = [
            f"\n{'='*65}",
            f"  主成分分析结果摘要 (PCA)",
            f"{'='*65}",
            f"  原始指标数    : {r.scalars['n_features_original']}",
            f"  保留主成分数  : {r.scalars['n_components_selected']}",
            f"  目标累积方差  : {r.scalars['variance_ratio_target']:.2%}",
            f"  实际累积方差  : {r.scalars['variance_ratio_actual']:.4f} "
            f"({r.scalars['variance_ratio_actual']:.2%})",
            f"  耗时          : {r.elapsed_time:.3f}s",
            f"\n  方差贡献率明细：",
        ]
        if "variance" in r.tables:
            lines.append(r.tables["variance"].to_string(index=False))
        lines.append(f"\n  载荷矩阵（前5行）：")
        if "loadings" in r.tables:
            lines.append(r.tables["loadings"].head().to_string())
        lines.append("="*65)
        text = "\n".join(lines)
        print(text)
        return text

    def tex_description(self) -> str:
        result_section = ""
        if self._result is not None:
            k    = self._result.scalars["n_components_selected"]
            cr   = self._result.scalars["variance_ratio_actual"]
            n_f  = self._result.scalars["n_features_original"]

            # 方差贡献率表 LaTeX
            vt = self._result.tables.get("variance", pd.DataFrame())
            if not vt.empty:
                rows_tex = " \\\\\n    ".join(
                    rf"{row['主成分']} & {row['特征值']} & "
                    rf"{row['方差贡献率']} & {row['累积方差贡献率']}"
                    for _, row in vt.iterrows()
                )
                variance_table = rf"""
\begin{{table}}[htbp]
\centering
\caption{{主成分方差贡献率}}
\label{{tab:pca_variance}}
\begin{{tabular}}{{cccc}}
\toprule
主成分 & 特征值 & 方差贡献率 & 累积贡献率 \\
\midrule
    {rows_tex} \\
\bottomrule
\end{{tabular}}
\end{{table}}"""
            else:
                variance_table = ""

            result_section = rf"""
分析结果显示，从原始 {n_f} 个指标中提取 ${k}$ 个主成分，
累积方差贡献率达 ${cr:.2%}$，满足信息保留要求。
{variance_table}"""

        return rf"""
\subsubsection{{主成分分析（PCA）}}
为消除指标间的多重共线性并实现降维，对经标准化处理后的数据矩阵进行
主成分分析。设原始指标数据矩阵为 $X \in \mathbb{{R}}^{{m \times n}}$，
标准化后计算协方差矩阵 $\Sigma = X^T X / (m-1)$，
对其进行特征值分解：
\begin{{equation}}
    \Sigma \boldsymbol{{v}}_k = \lambda_k \boldsymbol{{v}}_k, \quad k = 1, 2, \ldots, p
\end{{equation}}
按特征值从大到小排列，取前 $k$ 个特征向量构成投影矩阵 $W$，
则主成分得分矩阵为：
\begin{{equation}}
    F = XW, \quad W = [\boldsymbol{{v}}_1, \boldsymbol{{v}}_2, \ldots, \boldsymbol{{v}}_k]
\end{{equation}}
选取标准为累积方差贡献率 $\sum_{{i=1}}^k \lambda_i / \sum_{{i=1}}^p \lambda_i \geq {self._variance_ratio:.0%}$。
{result_section}
"""