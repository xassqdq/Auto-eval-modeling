# src/algorithms/base.py
"""
算法抽象基类模块

设计原则：
1. 统一接口：所有评价算法继承 BaseMethod，实现相同的 fit/compute/summary 接口
2. 丰富元数据：每个算法自带描述文字、LaTeX 片段生成、结果结构化输出
3. 可组合：算法间通过标准化的 MethodResult 传递数据
4. 可扩展：新算法只需继承 BaseMethod 并实现抽象方法
"""

from __future__ import annotations

import time
import abc
import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..utils.logging_config import LoggingMixin


# ============================================================
#  枚举定义
# ============================================================

class IndicatorDirection(str, Enum):
    """指标方向枚举"""
    POSITIVE = "positive"    # 正向指标（越大越好）
    NEGATIVE = "negative"    # 负向指标（越小越好）
    MODERATE = "moderate"    # 适度型指标（有最优值）


class MethodCategory(str, Enum):
    """算法类别枚举"""
    PREPROCESS = "preprocess"           # 数据预处理
    WEIGHT_SUBJECTIVE = "weight_sub"    # 主观赋权
    WEIGHT_OBJECTIVE = "weight_obj"     # 客观赋权
    WEIGHT_COMBINATION = "weight_comb"  # 组合赋权
    EVALUATION = "evaluation"           # 综合评价
    SENSITIVITY = "sensitivity"         # 灵敏度分析
    REDUCTION = "reduction"             # 降维
    CLUSTERING = "clustering"           # 聚类


# ============================================================
#  结果数据类
# ============================================================

@dataclass
class DataProfile:
    """
    数据概况描述，供算法推荐器使用

    Attributes
    ----------
    n_samples : int
        样本数量（评价对象数）
    n_indicators : int
        指标数量
    missing_rate : float
        缺失值比例（0~1）
    has_correlation : bool
        是否存在高相关性指标对
    max_correlation : float
        最大相关系数绝对值
    has_time_dim : bool
        是否包含时间维度（动态评价）
    numeric_cols : list[str]
        数值列名列表
    all_positive : bool
        所有数值是否为正数
    notes : list[str]
        自动生成的数据说明列表
    """
    n_samples: int = 0
    n_indicators: int = 0
    missing_rate: float = 0.0
    has_correlation: bool = False
    max_correlation: float = 0.0
    has_time_dim: bool = False
    numeric_cols: list[str] = field(default_factory=list)
    all_positive: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_indicators": self.n_indicators,
            "missing_rate": self.missing_rate,
            "has_correlation": self.has_correlation,
            "max_correlation": self.max_correlation,
            "has_time_dim": self.has_time_dim,
            "all_positive": self.all_positive,
            "notes": self.notes,
        }


@dataclass
class MethodResult:
    """
    算法运行结果标准容器

    所有算法的 compute() 方法返回此类型，
    确保下游节点和报告生成器可以统一处理结果。

    Attributes
    ----------
    method_name : str
        算法名称（如 'TOPSIS', 'AHP'）
    category : MethodCategory
        算法类别
    scores : pd.Series | None
        综合得分序列（index 为评价对象名）
    rankings : pd.Series | None
        排名序列（1 为最优，index 为评价对象名）
    weights : pd.Series | None
        权重向量（index 为指标名）
    tables : dict[str, pd.DataFrame]
        中间过程表格（key 为表名，如 'entropy_table', 'weight_table'）
    scalars : dict[str, float | str]
        标量结果（如 CR值, 一致性比例）
    metadata : dict[str, Any]
        附加元数据（算法专属信息）
    warnings : list[str]
        运行中产生的警告信息
    elapsed_time : float
        运行耗时（秒）
    """
    method_name: str = ""
    category: MethodCategory = MethodCategory.EVALUATION
    scores: Optional[pd.Series] = None
    rankings: Optional[pd.Series] = None
    weights: Optional[pd.Series] = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    elapsed_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典"""
        result = {
            "method_name": self.method_name,
            "category": self.category.value,
            "scalars": self.scalars,
            "warnings": self.warnings,
            "elapsed_time": self.elapsed_time,
        }
        if self.scores is not None:
            result["scores"] = self.scores.to_dict()
        if self.rankings is not None:
            result["rankings"] = self.rankings.to_dict()
        if self.weights is not None:
            result["weights"] = self.weights.to_dict()
        result["tables"] = {
            k: v.to_dict() for k, v in self.tables.items()
        }
        return result

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        生成综合摘要 DataFrame（对象 × 得分/排名）

        Returns
        -------
        pd.DataFrame
            列：得分、排名（若存在）
        """
        dfs = {}
        if self.scores is not None:
            dfs[f"{self.method_name}_得分"] = self.scores
        if self.rankings is not None:
            dfs[f"{self.method_name}_排名"] = self.rankings
        if not dfs:
            return pd.DataFrame()
        return pd.DataFrame(dfs)

    def add_warning(self, msg: str) -> None:
        """添加警告信息并记录日志"""
        self.warnings.append(msg)

    def __repr__(self) -> str:
        scores_info = (
            f"scores_range=[{self.scores.min():.4f}, {self.scores.max():.4f}]"
            if self.scores is not None else "no_scores"
        )
        return (
            f"MethodResult(method='{self.method_name}', "
            f"n_objects={len(self.scores) if self.scores is not None else 0}, "
            f"{scores_info}, "
            f"elapsed={self.elapsed_time:.3f}s)"
        )


# ============================================================
#  抽象基类
# ============================================================

class BaseMethod(LoggingMixin, abc.ABC):
    """
    所有评价算法的抽象基类。

    子类必须实现:
    - fit(data, **kwargs)     数据适配与参数验证
    - compute()               核心计算逻辑，返回 MethodResult
    - summary()               打印/返回结果摘要
    - tex_description()       返回该算法的 LaTeX 描述片段

    可选实现:
    - validate_input()        输入数据特定校验
    - get_default_params()    返回默认参数字典

    Example
    -------
    >>> class MyMethod(BaseMethod):
    ...     CATEGORY = MethodCategory.EVALUATION
    ...     METHOD_NAME_ZH = "我的方法"
    ...     METHOD_NAME_EN = "MyMethod"
    ...
    ...     def fit(self, data, weights=None, **kwargs):
    ...         self._data = data
    ...         self._weights = weights
    ...         return self
    ...
    ...     def compute(self):
    ...         # 核心算法逻辑
    ...         ...
    ...
    ...     def summary(self):
    ...         print(self._result)
    ...
    ...     def tex_description(self):
    ...         return r"\\subsection{我的方法}..."
    """

    # 子类需要覆盖的类属性
    CATEGORY: MethodCategory = MethodCategory.EVALUATION
    METHOD_NAME_ZH: str = "未命名算法"
    METHOD_NAME_EN: str = "UnnamedMethod"
    METHOD_ABBR: str = "UNNAMED"         # 缩写，用于文件命名
    REQUIRES_WEIGHTS: bool = False        # 是否需要外部权重输入
    REQUIRES_EXPERT_INPUT: bool = False   # 是否需要专家判断输入

    def __init__(self, language: str = "zh"):
        """
        Parameters
        ----------
        language : str
            输出语言，'zh'（中文）或 'en'（英文）
        """
        self._language = language
        self._is_fitted: bool = False
        self._result: Optional[MethodResult] = None
        self._fit_time: float = 0.0
        self._compute_time: float = 0.0
        # 原始数据备份（用于灵敏度分析）
        self._raw_data: Optional[pd.DataFrame] = None
        self._raw_weights: Optional[np.ndarray] = None
        self._indicator_names: list[str] = []
        self._object_names: list[str] = []

    # --------------------------------------------------------
    #  抽象方法（子类必须实现）
    # --------------------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> "BaseMethod":
        """
        数据适配与参数设置。

        Parameters
        ----------
        data : pd.DataFrame
            预处理后的数据，行为评价对象，列为指标
        **kwargs :
            算法专属参数（如 weights, judgment_matrix 等）

        Returns
        -------
        self : BaseMethod
            支持链式调用
        """
        ...

    @abc.abstractmethod
    def compute(self) -> MethodResult:
        """
        执行核心计算，返回标准化结果。

        必须在 fit() 之后调用。

        Returns
        -------
        MethodResult
            标准化计算结果容器
        """
        ...

    @abc.abstractmethod
    def summary(self) -> str:
        """
        返回计算结果的文字摘要（同时打印到控制台）。

        Returns
        -------
        str
            摘要文本
        """
        ...

    @abc.abstractmethod
    def tex_description(self) -> str:
        """
        返回该算法在论文中的 LaTeX 描述片段。
        包含：方法介绍、数学公式、参数说明。

        Returns
        -------
        str
            LaTeX 源码字符串
        """
        ...

    # --------------------------------------------------------
    #  通用实现方法（子类可覆盖）
    # --------------------------------------------------------

    def fit_compute(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> MethodResult:
        """
        便捷方法：一次性执行 fit + compute。

        Parameters
        ----------
        data : pd.DataFrame
        **kwargs :
            传递给 fit() 的参数

        Returns
        -------
        MethodResult
        """
        return self.fit(data, **kwargs).compute()

    def validate_input(self, data: pd.DataFrame) -> None:
        """
        通用输入数据校验（子类可在此基础上扩展）。

        Raises
        ------
        ValueError
            数据格式不符合要求时抛出
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data 必须为 pd.DataFrame，实际类型: {type(data)}")

        if data.empty:
            raise ValueError("数据不能为空")

        if data.shape[0] < 2:
            raise ValueError(f"评价对象数量至少需要 2 个，当前: {data.shape[0]}")

        if data.shape[1] < 1:
            raise ValueError(f"指标数量至少需要 1 个，当前: {data.shape[1]}")

        # 检查是否存在非数值列
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"数据中存在非数值列: {non_numeric}。"
                f"请先将这些列转换为数值或在加载时排除。"
            )

        # 检查缺失值
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            missing_rate = missing_count / data.size
            if missing_rate > 0.3:
                raise ValueError(
                    f"缺失值比例过高: {missing_rate:.1%}。"
                    f"请先使用预处理模块处理缺失值。"
                )
            self.logger.warning(
                f"数据中存在 {missing_count} 个缺失值 "
                f"（比例: {missing_rate:.1%}），"
                f"建议先进行预处理。"
            )

    def get_default_params(self) -> dict[str, Any]:
        """
        返回算法的默认参数字典（可用于构建配置模板）。

        Returns
        -------
        dict
        """
        return {}

    def get_result(self) -> MethodResult:
        """
        获取最近一次计算结果。

        Returns
        -------
        MethodResult

        Raises
        ------
        RuntimeError
            未执行 compute() 时抛出
        """
        if self._result is None:
            raise RuntimeError(
                f"{self.METHOD_NAME_EN} 尚未执行计算，请先调用 compute() 方法。"
            )
        return self._result

    def reset(self) -> "BaseMethod":
        """重置算法状态（清除 fit 和 compute 的结果）"""
        self._is_fitted = False
        self._result = None
        self._raw_data = None
        self._raw_weights = None
        self._indicator_names = []
        self._object_names = []
        self.logger.debug(f"{self.METHOD_NAME_EN} 状态已重置")
        return self

    # --------------------------------------------------------
    #  内部辅助方法
    # --------------------------------------------------------

    def _start_timer(self) -> float:
        """返回当前时间戳（用于计时）"""
        return time.perf_counter()

    def _stop_timer(self, start: float) -> float:
        """计算耗时（秒）"""
        return time.perf_counter() - start

    def _check_fitted(self, method_name: str = "compute") -> None:
        """检查是否已执行 fit()，未执行则抛出异常"""
        if not self._is_fitted:
            raise RuntimeError(
                f"请先调用 {type(self).__name__}.fit() 方法，"
                f"再调用 {method_name}()。"
            )

    def _normalize_weights(
        self,
        weights: np.ndarray | list[float],
    ) -> np.ndarray:
        """
        归一化权重向量（确保和为 1，元素 ≥ 0）。

        Parameters
        ----------
        weights : array-like
            原始权重

        Returns
        -------
        np.ndarray
            归一化权重

        Raises
        ------
        ValueError
            权重包含负值或全为零时抛出
        """
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0):
            raise ValueError(f"权重不能包含负值: {w}")
        total = w.sum()
        if total == 0:
            raise ValueError("权重之和不能为零")
        return w / total

    def _build_result(
        self,
        scores: Optional[pd.Series] = None,
        rankings: Optional[pd.Series] = None,
        weights: Optional[pd.Series] = None,
        tables: Optional[dict[str, pd.DataFrame]] = None,
        scalars: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        elapsed: float = 0.0,
    ) -> MethodResult:
        """
        构建标准化 MethodResult 对象的便捷方法。
        """
        result = MethodResult(
            method_name=self.METHOD_NAME_ZH if self._language == "zh"
                        else self.METHOD_NAME_EN,
            category=self.CATEGORY,
            scores=scores,
            rankings=rankings,
            weights=weights,
            tables=tables or {},
            scalars=scalars or {},
            metadata=metadata or {},
            elapsed_time=elapsed,
        )
        return result

    @staticmethod
    def _rank_scores(
        scores: pd.Series,
        ascending: bool = False,
    ) -> pd.Series:
        """
        将得分序列转换为排名（1 为最优）。

        Parameters
        ----------
        scores : pd.Series
        ascending : bool
            True 表示得分越小越优（如 VIKOR 的 Q 值）

        Returns
        -------
        pd.Series
            排名序列，类型为 int
        """
        return scores.rank(
            ascending=ascending,
            method="min",   # 并列时取最小排名
        ).astype(int)

    @staticmethod
    def _safe_log(
        arr: np.ndarray,
        epsilon: float = 1e-12,
    ) -> np.ndarray:
        """
        安全对数计算，避免 log(0)。

        Parameters
        ----------
        arr : np.ndarray
            输入数组（需非负）
        epsilon : float
            防零分母

        Returns
        -------
        np.ndarray
        """
        return np.log(np.where(arr <= 0, epsilon, arr))

    @staticmethod
    def _safe_divide(
        numerator: np.ndarray,
        denominator: np.ndarray,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """
        安全除法，分母为零时返回 fill_value。
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(
                denominator == 0,
                fill_value,
                numerator / denominator,
            )
        return result

    # --------------------------------------------------------
    #  属性访问器
    # --------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """是否已完成 fit()"""
        return self._is_fitted

    @property
    def language(self) -> str:
        """输出语言"""
        return self._language

    @language.setter
    def language(self, lang: str) -> None:
        if lang not in ("zh", "en"):
            raise ValueError(f"language 必须为 'zh' 或 'en'，实际: {lang}")
        self._language = lang

    @property
    def method_info(self) -> dict[str, str]:
        """返回算法基本信息字典"""
        return {
            "name_zh": self.METHOD_NAME_ZH,
            "name_en": self.METHOD_NAME_EN,
            "abbr": self.METHOD_ABBR,
            "category": self.CATEGORY.value,
            "requires_weights": str(self.REQUIRES_WEIGHTS),
            "requires_expert": str(self.REQUIRES_EXPERT_INPUT),
        }

    # --------------------------------------------------------
    #  深拷贝支持
    # --------------------------------------------------------

    def clone(self) -> "BaseMethod":
        """
        返回当前算法对象的深拷贝（保留配置，清除计算状态）。

        Returns
        -------
        BaseMethod
        """
        obj = copy.deepcopy(self)
        obj.reset()
        return obj

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not_fitted"
        return (
            f"{type(self).__name__}("
            f"name='{self.METHOD_NAME_EN}', "
            f"status='{status}', "
            f"lang='{self._language}')"
        )

    def __str__(self) -> str:
        lines = [
            f"{'='*50}",
            f"算法: {self.METHOD_NAME_ZH} ({self.METHOD_NAME_EN})",
            f"类别: {self.CATEGORY.value}",
            f"状态: {'已适配' if self._is_fitted else '未适配'}",
        ]
        if self._is_fitted:
            lines.append(f"指标数: {len(self._indicator_names)}")
            lines.append(f"对象数: {len(self._object_names)}")
        if self._result is not None:
            lines.append(f"计算耗时: {self._result.elapsed_time:.4f}s")
        lines.append(f"{'='*50}")
        return "\n".join(lines)