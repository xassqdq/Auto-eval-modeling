# src/algorithms/preprocess/cleaner.py
"""
数据清洗模块
- MissingValueHandler : 缺失值处理（10种策略）
- OutlierHandler      : 异常值检测与处理（3种检测 × 3种动作）
- DataCleaner         : 组合清洗流水线（缺失值 → 异常值）
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from . import PreprocessBase
from ..base import MethodCategory, MethodResult
from ...utils.logging_config import get_logger

logger = get_logger("auto_eval.cleaner")

# 填充策略合法值
_MISSING_STRATEGIES = frozenset([
    "mean", "median", "mode", "constant",
    "interpolate", "ffill", "bfill", "knn",
    "drop_rows", "drop_cols",
])

# 异常值检测方法
_OUTLIER_METHODS = frozenset(["zscore", "iqr", "mad"])

# 异常值处理动作
_OUTLIER_ACTIONS = frozenset(["clip", "remove", "flag"])


# ============================================================
#  1. 缺失值处理
# ============================================================

class MissingValueHandler(PreprocessBase):
    """
    缺失值处理算法。

    支持的填充策略
    --------------
    mean        : 以该列均值填充
    median      : 以该列中位数填充
    mode        : 以该列众数填充（多个众数时取第一个）
    constant    : 以指定常数填充（需指定 fill_value）
    interpolate : 线性插值（仅对数值列，按行序）
    ffill       : 前向填充（用前一个有效值填充）
    bfill       : 后向填充（用后一个有效值填充）
    knn         : K近邻插补（需安装 scikit-learn）
    drop_rows   : 删除含任意缺失值的行
    drop_cols   : 删除缺失率超过阈值的列

    Parameters（fit 传入）
    ----------------------
    strategy        : 填充策略，默认 'mean'
    fill_value      : 常数填充值（strategy='constant' 时生效）
    drop_threshold  : 列删除阈值（strategy='drop_cols' 时，缺失率>此值的列被删除）
    knn_neighbors   : KNN 邻居数（strategy='knn' 时生效）
    cols            : 指定处理的列名列表，None 表示处理所有数值列
    """

    METHOD_NAME_ZH = "缺失值处理"
    METHOD_NAME_EN = "MissingValueHandler"
    METHOD_ABBR    = "MVH"

    def __init__(self, language: str = "zh"):
        super().__init__(language)
        self._strategy: str = "mean"
        self._fill_value: float = 0.0
        self._drop_threshold: float = 0.5
        self._knn_neighbors: int = 5
        self._cols: Optional[list[str]] = None

    # ── fit ────────────────────────────────────────────────

    def fit(
        self,
        data: pd.DataFrame,
        strategy: str = "mean",
        fill_value: float = 0.0,
        drop_threshold: float = 0.5,
        knn_neighbors: int = 5,
        cols: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "MissingValueHandler":
        """
        Parameters
        ----------
        data            : 输入 DataFrame（可含缺失值）
        strategy        : 填充策略
        fill_value      : 常数值（strategy='constant' 专用）
        drop_threshold  : 缺失率阈值（strategy='drop_cols' 专用），范围 [0, 1]
        knn_neighbors   : KNN 邻居数
        cols            : 需要处理的列名列表，None 表示自动选取数值列
        """
        self.validate_input(data)

        if strategy not in _MISSING_STRATEGIES:
            raise ValueError(
                f"strategy='{strategy}' 无效，合法值: {sorted(_MISSING_STRATEGIES)}"
            )
        if not (0.0 <= drop_threshold <= 1.0):
            raise ValueError(
                f"drop_threshold={drop_threshold} 超出范围 [0, 1]"
            )

        self._raw_data = data.copy()
        self._indicator_names = list(data.columns)
        self._object_names    = list(data.index)
        self._strategy        = strategy
        self._fill_value      = fill_value
        self._drop_threshold  = drop_threshold
        self._knn_neighbors   = knn_neighbors
        self._cols            = cols
        self._is_fitted       = True

        # 预报告缺失情况
        n_missing = int(data.isnull().sum().sum())
        self.logger.info(
            f"MissingValueHandler.fit 完成 | "
            f"strategy='{strategy}' | 共 {n_missing} 个缺失值"
        )
        return self

    # ── compute ────────────────────────────────────────────

    def compute(self) -> MethodResult:
        """执行缺失值填充，返回 MethodResult。"""
        self._check_fitted("compute")
        t0 = self._start_timer()

        df = self._raw_data.copy()

        # 确定需要处理的列
        _, num_cols = self._extract_numeric(df)
        target_cols = (
            [c for c in self._cols if c in df.columns]
            if self._cols else num_cols
        )

        # 统计处理前缺失
        missing_before = df[target_cols].isnull().sum()
        total_before   = int(missing_before.sum())

        # ── 执行填充策略 ──────────────────────────────────

        report_rows = []   # 记录每列的处理情况

        if self._strategy == "drop_rows":
            rows_before = len(df)
            df = df.dropna(subset=target_cols)
            rows_dropped = rows_before - len(df)
            report_rows.append({
                "列": "全部目标列",
                "策略": "删除含缺失值的行",
                "填充值/操作": f"删除 {rows_dropped} 行",
                "处理前缺失数": total_before,
                "处理后缺失数": 0,
            })
            self.logger.info(f"drop_rows: 删除 {rows_dropped} 行")

        elif self._strategy == "drop_cols":
            missing_rate = df[target_cols].isnull().mean()
            drop_cols = missing_rate[missing_rate > self._drop_threshold].index.tolist()
            df = df.drop(columns=drop_cols)
            # 更新 target_cols
            target_cols = [c for c in target_cols if c not in drop_cols]
            for col in drop_cols:
                report_rows.append({
                    "列": col,
                    "策略": "删除列",
                    "填充值/操作": f"缺失率={missing_rate[col]:.2%} > {self._drop_threshold:.2%}",
                    "处理前缺失数": int(missing_before.get(col, 0)),
                    "处理后缺失数": 0,
                })
            self.logger.info(f"drop_cols: 删除 {len(drop_cols)} 列: {drop_cols}")

        elif self._strategy == "knn":
            df = self._knn_fill(df, target_cols)
            for col in target_cols:
                n_filled = int(missing_before.get(col, 0))
                if n_filled > 0:
                    report_rows.append({
                        "列": col,
                        "策略": "KNN填充",
                        "填充值/操作": f"k={self._knn_neighbors}",
                        "处理前缺失数": n_filled,
                        "处理后缺失数": int(df[col].isnull().sum()),
                    })

        else:
            # 逐列处理
            for col in target_cols:
                n_miss = int(df[col].isnull().sum())
                if n_miss == 0:
                    continue

                fill_desc, df = self._fill_column(df, col, self._strategy)
                report_rows.append({
                    "列": col,
                    "策略": self._strategy,
                    "填充值/操作": fill_desc,
                    "处理前缺失数": n_miss,
                    "处理后缺失数": int(df[col].isnull().sum()),
                })

        # 统计处理后缺失
        remaining_cols = [c for c in target_cols if c in df.columns]
        total_after = int(df[remaining_cols].isnull().sum().sum())

        report_df = (
            pd.DataFrame(report_rows)
            if report_rows
            else pd.DataFrame(columns=["列", "策略", "填充值/操作", "处理前缺失数", "处理后缺失数"])
        )

        result = self._build_result(
            tables={
                "output":  df,
                "missing_report": report_df,
            },
            scalars={
                "missing_before": total_before,
                "missing_after":  total_after,
                "n_cols_handled": len(report_rows),
                "strategy":       self._strategy,
            },
            metadata={"output_data": df},
            elapsed=self._stop_timer(t0),
        )
        self._result = result
        self.logger.info(
            f"缺失值处理完成 | {total_before} → {total_after} | "
            f"耗时 {result.elapsed_time:.3f}s"
        )
        return result

    # ── 内部工具 ───────────────────────────────────────────

    def _fill_column(
        self,
        df: pd.DataFrame,
        col: str,
        strategy: str,
    ) -> tuple[str, pd.DataFrame]:
        """对单列执行填充，返回 (描述, 更新后的df)"""

        if strategy == "mean":
            val = df[col].mean()
            df[col] = df[col].fillna(val)
            return f"均值={val:.4f}", df

        elif strategy == "median":
            val = df[col].median()
            df[col] = df[col].fillna(val)
            return f"中位数={val:.4f}", df

        elif strategy == "mode":
            mode_result = df[col].mode()
            val = mode_result.iloc[0] if len(mode_result) > 0 else df[col].mean()
            df[col] = df[col].fillna(val)
            return f"众数={val:.4f}", df

        elif strategy == "constant":
            df[col] = df[col].fillna(self._fill_value)
            return f"常数={self._fill_value}", df

        elif strategy == "interpolate":
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            return "线性插值", df

        elif strategy == "ffill":
            df[col] = df[col].ffill()
            # 边界处可能仍有 NaN，用 bfill 补救
            df[col] = df[col].bfill()
            return "前向填充", df

        elif strategy == "bfill":
            df[col] = df[col].bfill()
            df[col] = df[col].ffill()
            return "后向填充", df

        return "未处理", df

    def _knn_fill(
        self,
        df: pd.DataFrame,
        cols: list[str],
    ) -> pd.DataFrame:
        """使用 sklearn KNNImputer 进行 KNN 填充"""
        try:
            from sklearn.impute import KNNImputer
        except ImportError:
            raise ImportError(
                "KNN 填充需要 scikit-learn，请安装: pip install scikit-learn"
            )
        imputer = KNNImputer(n_neighbors=self._knn_neighbors)
        df = df.copy()
        df[cols] = imputer.fit_transform(df[cols])
        return df

    # ── summary / tex ──────────────────────────────────────

    def summary(self) -> str:
        r = self.get_result()
        lines = [
            f"\n{'='*55}",
            f"  缺失值处理结果摘要 ({self.METHOD_NAME_EN})",
            f"{'='*55}",
            f"  策略        : {r.scalars['strategy']}",
            f"  处理前缺失  : {r.scalars['missing_before']} 个",
            f"  处理后缺失  : {r.scalars['missing_after']} 个",
            f"  处理列数    : {r.scalars['n_cols_handled']} 列",
            f"  耗时        : {r.elapsed_time:.3f}s",
        ]
        if "missing_report" in r.tables and not r.tables["missing_report"].empty:
            lines.append("\n  处理明细：")
            lines.append(r.tables["missing_report"].to_string(index=False))
        lines.append("="*55)
        text = "\n".join(lines)
        print(text)
        return text

    def tex_description(self) -> str:
        strategy_desc = {
            "mean":       "均值填充法",
            "median":     "中位数填充法",
            "mode":       "众数填充法",
            "constant":   f"常数填充法（填充值 = {self._fill_value}）",
            "interpolate":"线性插值法",
            "ffill":      "前向填充法",
            "bfill":      "后向填充法",
            "knn":        f"K近邻插补法（$k={self._knn_neighbors}$）",
            "drop_rows":  "删除含缺失值的行",
            "drop_cols":  "删除缺失率超阈值的列",
        }
        desc = strategy_desc.get(self._strategy, self._strategy)

        result_info = ""
        if self._result is not None:
            before = self._result.scalars.get("missing_before", "N/A")
            after  = self._result.scalars.get("missing_after", "N/A")
            result_info = (
                f"\n经处理，数据集中缺失值数量由 {before} 个"
                f"降至 {after} 个，数据完整性得到保障。"
            )

        return rf"""
\subsubsection{{缺失值处理}}
对原始数据集进行缺失值分析，采用\textbf{{{desc}}}对缺失数据进行处理。
{result_info}
"""


# ============================================================
#  2. 异常值处理
# ============================================================

class OutlierHandler(PreprocessBase):
    """
    异常值检测与处理算法。

    检测方法
    --------
    zscore  : 标准分数法，|z_i| = |(x_i - μ) / σ| > threshold 视为异常
    iqr     : 四分位距法，x < Q1 - k·IQR 或 x > Q3 + k·IQR 视为异常
    mad     : 中位数绝对偏差法，|x_i - median| / MAD > threshold

    处理动作
    --------
    clip    : 截断至边界值（推荐，保留数据量）
    remove  : 将异常值置为 NaN（需配合缺失值处理）
    flag    : 仅标记，不修改数据（添加 _outlier 标记列）
    """

    METHOD_NAME_ZH = "异常值处理"
    METHOD_NAME_EN = "OutlierHandler"
    METHOD_ABBR    = "OLH"

    def __init__(self, language: str = "zh"):
        super().__init__(language)
        self._method: str    = "zscore"
        self._action: str    = "clip"
        self._threshold: float = 3.0
        self._iqr_k: float   = 1.5
        self._cols: Optional[list[str]] = None

    # ── fit ────────────────────────────────────────────────

    def fit(
        self,
        data: pd.DataFrame,
        method: str = "zscore",
        action: str = "clip",
        threshold: float = 3.0,
        iqr_k: float = 1.5,
        cols: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "OutlierHandler":
        """
        Parameters
        ----------
        data      : 输入 DataFrame
        method    : 检测方法 ('zscore', 'iqr', 'mad')
        action    : 处理动作 ('clip', 'remove', 'flag')
        threshold : 判断阈值（zscore/mad 的倍数）
        iqr_k     : IQR 倍数（iqr 方法专用，常用 1.5 或 3.0）
        cols      : 指定处理的列，None 表示所有数值列
        """
        self.validate_input(data)

        if method not in _OUTLIER_METHODS:
            raise ValueError(f"method='{method}' 无效，合法值: {sorted(_OUTLIER_METHODS)}")
        if action not in _OUTLIER_ACTIONS:
            raise ValueError(f"action='{action}' 无效，合法值: {sorted(_OUTLIER_ACTIONS)}")
        if threshold <= 0:
            raise ValueError(f"threshold={threshold} 必须 > 0")

        self._raw_data   = data.copy()
        self._indicator_names = list(data.columns)
        self._object_names    = list(data.index)
        self._method    = method
        self._action    = action
        self._threshold = threshold
        self._iqr_k     = iqr_k
        self._cols      = cols
        self._is_fitted = True

        self.logger.info(
            f"OutlierHandler.fit | method='{method}' | "
            f"action='{action}' | threshold={threshold}"
        )
        return self

    # ── compute ────────────────────────────────────────────

    def compute(self) -> MethodResult:
        """执行异常值处理。"""
        self._check_fitted("compute")
        t0 = self._start_timer()

        df = self._raw_data.copy()
        _, num_cols = self._extract_numeric(df)
        target_cols = (
            [c for c in self._cols if c in df.columns]
            if self._cols else num_cols
        )

        report_rows = []
        total_outliers = 0

        for col in target_cols:
            series = df[col].dropna()
            if len(series) < 4:
                self.logger.debug(f"列 '{col}' 样本过少（{len(series)}），跳过异常值检测")
                continue

            # 检测异常值
            mask = self._detect_outliers(df[col])
            n_outliers = int(mask.sum())
            total_outliers += n_outliers

            if n_outliers == 0:
                continue

            # 执行动作
            lower, upper = self._compute_bounds(df[col])
            action_desc = self._apply_action(df, col, mask, lower, upper)

            report_rows.append({
                "列":       col,
                "检测方法": self._method,
                "处理动作": self._action,
                "异常值数": n_outliers,
                "下界":     f"{lower:.4f}",
                "上界":     f"{upper:.4f}",
                "操作描述": action_desc,
            })

        report_df = (
            pd.DataFrame(report_rows)
            if report_rows
            else pd.DataFrame(columns=["列", "检测方法", "处理动作", "异常值数",
                                       "下界", "上界", "操作描述"])
        )

        result = self._build_result(
            tables={
                "output":         df,
                "outlier_report": report_df,
            },
            scalars={
                "total_outliers": total_outliers,
                "n_cols_handled": len(report_rows),
                "method":         self._method,
                "action":         self._action,
            },
            metadata={"output_data": df},
            elapsed=self._stop_timer(t0),
        )
        self._result = result
        self.logger.info(
            f"异常值处理完成 | 共 {total_outliers} 个异常 | "
            f"耗时 {result.elapsed_time:.3f}s"
        )
        return result

    # ── 内部工具 ───────────────────────────────────────────

    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """返回布尔 Series，True 表示异常值位置"""
        clean = series.dropna()

        if self._method == "zscore":
            mu, sigma = clean.mean(), clean.std()
            if sigma == 0:
                return pd.Series(False, index=series.index)
            z = (series - mu).abs() / sigma
            return z > self._threshold

        elif self._method == "iqr":
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self._iqr_k * iqr
            upper = q3 + self._iqr_k * iqr
            return (series < lower) | (series > upper)

        elif self._method == "mad":
            median = clean.median()
            mad = (clean - median).abs().median()
            if mad == 0:
                return pd.Series(False, index=series.index)
            modified_z = (series - median).abs() / (1.4826 * mad)
            return modified_z > self._threshold

        return pd.Series(False, index=series.index)

    def _compute_bounds(self, series: pd.Series) -> tuple[float, float]:
        """计算正常范围边界 [lower, upper]"""
        clean = series.dropna()

        if self._method == "zscore":
            mu, sigma = clean.mean(), clean.std()
            return mu - self._threshold * sigma, mu + self._threshold * sigma

        elif self._method == "iqr":
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            return q1 - self._iqr_k * iqr, q3 + self._iqr_k * iqr

        elif self._method == "mad":
            median = clean.median()
            mad = (clean - median).abs().median()
            delta = self._threshold * 1.4826 * mad
            return median - delta, median + delta

        return float(clean.min()), float(clean.max())

    def _apply_action(
        self,
        df: pd.DataFrame,
        col: str,
        mask: pd.Series,
        lower: float,
        upper: float,
    ) -> str:
        """执行异常值处理动作，就地修改 df"""

        if self._action == "clip":
            df.loc[mask & (df[col] < lower), col] = lower
            df.loc[mask & (df[col] > upper), col] = upper
            return f"截断至 [{lower:.4f}, {upper:.4f}]"

        elif self._action == "remove":
            df.loc[mask, col] = np.nan
            return "置为 NaN（待缺失值处理）"

        elif self._action == "flag":
            flag_col = f"{col}_outlier"
            df[flag_col] = mask.astype(int)
            return f"新增标记列 '{flag_col}'"

        return "无操作"

    # ── summary / tex ──────────────────────────────────────

    def summary(self) -> str:
        r = self.get_result()
        lines = [
            f"\n{'='*55}",
            f"  异常值处理结果摘要 ({self.METHOD_NAME_EN})",
            f"{'='*55}",
            f"  检测方法    : {r.scalars['method']}",
            f"  处理动作    : {r.scalars['action']}",
            f"  总异常值数  : {r.scalars['total_outliers']} 个",
            f"  处理列数    : {r.scalars['n_cols_handled']} 列",
            f"  耗时        : {r.elapsed_time:.3f}s",
        ]
        rpt = r.tables.get("outlier_report", pd.DataFrame())
        if not rpt.empty:
            lines.append("\n  处理明细：")
            lines.append(rpt.to_string(index=False))
        lines.append("="*55)
        text = "\n".join(lines)
        print(text)
        return text

    def tex_description(self) -> str:
        method_map = {
            "zscore": (
                "Z-score法（标准分数法）",
                r"$|z_i| = \left|\dfrac{x_i - \bar{x}}{\sigma}\right| > "
                + str(self._threshold) + r"$",
            ),
            "iqr": (
                "IQR法（四分位距法）",
                r"$x < Q_1 - " + str(self._iqr_k)
                + r"\cdot\mathrm{IQR}$ 或 $x > Q_3 + "
                + str(self._iqr_k) + r"\cdot\mathrm{IQR}$",
            ),
            "mad": (
                "MAD法（中位数绝对偏差法）",
                r"$\dfrac{|x_i - \tilde{x}|}{1.4826\cdot\mathrm{MAD}} > "
                + str(self._threshold) + r"$",
            ),
        }
        name, formula = method_map.get(self._method, (self._method, ""))

        action_map = {
            "clip":   "将异常值截断至检测边界",
            "remove": "将异常值替换为缺失值",
            "flag":   "标记异常值位置（不修改数值）",
        }
        action_desc = action_map.get(self._action, self._action)

        return rf"""
\subsubsection{{异常值处理}}
采用\textbf{{{name}}}识别数据中的异常值，判断准则为：{formula}。
对检测到的异常值执行如下操作：\textbf{{{action_desc}}}。
"""


# ============================================================
#  3. 组合清洗流水线
# ============================================================

class DataCleaner(PreprocessBase):
    """
    数据清洗完整流水线。
    内部依次执行：MissingValueHandler → OutlierHandler。

    Parameters（fit 传入，均为可选）
    ---------------------------------
    missing_strategy  : 缺失值填充策略，同 MissingValueHandler
    missing_fill_value: 常数填充值
    missing_threshold : 列删除缺失率阈值
    outlier_method    : 异常值检测方法
    outlier_action    : 异常值处理动作
    outlier_threshold : 检测阈值
    iqr_k             : IQR 倍数
    handle_missing    : bool，是否执行缺失值处理（默认 True）
    handle_outlier    : bool，是否执行异常值处理（默认 True）
    """

    METHOD_NAME_ZH = "数据清洗流水线"
    METHOD_NAME_EN = "DataCleaner"
    METHOD_ABBR    = "DC"

    def __init__(self, language: str = "zh"):
        super().__init__(language)
        self._missing_handler: Optional[MissingValueHandler] = None
        self._outlier_handler: Optional[OutlierHandler]      = None
        self._handle_missing: bool = True
        self._handle_outlier: bool = True

    def fit(
        self,
        data: pd.DataFrame,
        missing_strategy: str = "mean",
        missing_fill_value: float = 0.0,
        missing_threshold: float = 0.5,
        outlier_method: str = "zscore",
        outlier_action: str = "clip",
        outlier_threshold: float = 3.0,
        iqr_k: float = 1.5,
        handle_missing: bool = True,
        handle_outlier: bool = True,
        cols: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "DataCleaner":
        self.validate_input(data)

        self._raw_data        = data.copy()
        self._indicator_names = list(data.columns)
        self._object_names    = list(data.index)
        self._handle_missing  = handle_missing
        self._handle_outlier  = handle_outlier

        if handle_missing:
            self._missing_handler = MissingValueHandler(self._language)
            self._missing_handler.fit(
                data,
                strategy=missing_strategy,
                fill_value=missing_fill_value,
                drop_threshold=missing_threshold,
                cols=cols,
            )

        if handle_outlier:
            self._outlier_handler = OutlierHandler(self._language)
            # outlier handler 在 compute 时才知道缺失值处理后的数据，
            # 暂存参数，compute 阶段再调用 fit
            self._outlier_params = dict(
                method=outlier_method,
                action=outlier_action,
                threshold=outlier_threshold,
                iqr_k=iqr_k,
                cols=cols,
            )

        self._is_fitted = True
        self.logger.info(
            f"DataCleaner.fit | handle_missing={handle_missing} | "
            f"handle_outlier={handle_outlier}"
        )
        return self

    def compute(self) -> MethodResult:
        self._check_fitted("compute")
        t0 = self._start_timer()

        current_df = self._raw_data.copy()
        missing_result = None
        outlier_result = None

        # ── Step 1: 缺失值处理 ─────────────────────────────
        if self._handle_missing and self._missing_handler:
            self._missing_handler._raw_data = current_df
            missing_result = self._missing_handler.compute()
            current_df     = missing_result.metadata["output_data"]
            self.logger.debug("缺失值处理步骤完成")

        # ── Step 2: 异常值处理 ─────────────────────────────
        if self._handle_outlier and self._outlier_handler:
            self._outlier_handler.fit(current_df, **self._outlier_params)
            outlier_result = self._outlier_handler.compute()
            current_df     = outlier_result.metadata["output_data"]
            self.logger.debug("异常值处理步骤完成")

        # ── 合并报告 ───────────────────────────────────────
        tables: dict[str, pd.DataFrame] = {"output": current_df}
        scalars: dict[str, Any] = {}

        if missing_result:
            tables["missing_report"] = missing_result.tables.get(
                "missing_report", pd.DataFrame()
            )
            scalars.update({
                "missing_before": missing_result.scalars.get("missing_before", 0),
                "missing_after":  missing_result.scalars.get("missing_after", 0),
            })

        if outlier_result:
            tables["outlier_report"] = outlier_result.tables.get(
                "outlier_report", pd.DataFrame()
            )
            scalars["total_outliers"] = outlier_result.scalars.get("total_outliers", 0)

        scalars["output_shape"] = current_df.shape

        result = self._build_result(
            tables=tables,
            scalars=scalars,
            metadata={"output_data": current_df},
            elapsed=self._stop_timer(t0),
        )
        self._result = result
        self.logger.info(
            f"DataCleaner 完成 | "
            f"输出形状: {current_df.shape} | "
            f"耗时: {result.elapsed_time:.3f}s"
        )
        return result

    def summary(self) -> str:
        r = self.get_result()
        lines = [
            f"\n{'='*60}",
            f"  数据清洗流水线摘要 (DataCleaner)",
            f"{'='*60}",
            f"  输出数据形状  : {r.scalars.get('output_shape', 'N/A')}",
            f"  缺失值: {r.scalars.get('missing_before', 0)} → {r.scalars.get('missing_after', 0)}",
            f"  总异常值数    : {r.scalars.get('total_outliers', 0)} 个",
            f"  总耗时        : {r.elapsed_time:.3f}s",
        ]
        lines.append("="*60)
        text = "\n".join(lines)
        print(text)
        return text

    def tex_description(self) -> str:
        parts = []
        if self._handle_missing and self._missing_handler:
            parts.append(self._missing_handler.tex_description())
        if self._handle_outlier and self._outlier_handler:
            parts.append(self._outlier_handler.tex_description())
        return "\n".join(parts) if parts else r"\subsubsection{数据清洗}\n数据清洗步骤已禁用。"