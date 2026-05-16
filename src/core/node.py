"""
工作流节点基类与具体节点实现

节点类型层级：
    BaseNode
    ├── DataLoadNode       - 数据加载
    ├── PreprocessNode     - 数据预处理
    ├── WeightNode         - 权重计算
    ├── EvaluationNode     - 综合评价
    ├── SensitivityNode    - 灵敏度分析
    └── ConsolidationNode  - 结果汇总

每个节点遵循统一接口：
    execute(bus: DataBus) → NodeResult

Author: AutoEval-Modeling
"""

from __future__ import annotations

import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from .data_bus import DataBus

logger = logging.getLogger(__name__)


# ======================================================================
# 枚举与数据类
# ======================================================================

class NodeStatus(Enum):
    """节点执行状态"""
    PENDING   = "pending"    # 等待执行
    RUNNING   = "running"    # 执行中
    SUCCESS   = "success"    # 成功完成
    SKIPPED   = "skipped"    # 被跳过（条件不满足）
    FAILED    = "failed"     # 执行失败


@dataclass
class NodeResult:
    """节点执行结果"""
    node_id:       str
    status:        NodeStatus
    outputs:       Dict[str, Any]  = field(default_factory=dict)
    error_msg:     Optional[str]   = None
    elapsed_sec:   float           = 0.0
    metadata:      Dict[str, Any]  = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == NodeStatus.SUCCESS


# ======================================================================
# 基类
# ======================================================================

class BaseNode(ABC):
    """
    工作流节点抽象基类

    Parameters
    ----------
    node_id : str
        节点唯一标识（工作流内不重复）
    config : dict
        节点配置参数
    dependencies : list of str
        依赖的上游节点 ID 列表

    Subclass Must Implement
    -----------------------
    _execute(bus: DataBus) → dict
        执行节点逻辑，从 bus 读取输入，将输出 put 回 bus，
        同时返回输出键值字典（用于记录）
    """

    def __init__(
        self,
        node_id:      str,
        config:       Optional[Dict]      = None,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        self.node_id:      str           = node_id
        self.config:       Dict          = config or {}
        self.dependencies: List[str]     = dependencies or []
        self.status:       NodeStatus    = NodeStatus.PENDING
        self._result:      Optional[NodeResult] = None

    # ------------------------------------------------------------------
    # 公开执行入口
    # ------------------------------------------------------------------

    def execute(self, bus: DataBus) -> NodeResult:
        """
        执行节点（带状态管理与异常捕获）

        Parameters
        ----------
        bus : DataBus
            共享数据总线
        """
        self.status = NodeStatus.RUNNING
        start_time  = time.time()
        logger.info("▶ 节点 [%s] 开始执行", self.node_id)

        try:
            # 前置条件检查
            self._pre_check(bus)
            # 执行核心逻辑
            outputs = self._execute(bus)
            elapsed = time.time() - start_time
            self.status = NodeStatus.SUCCESS

            result = NodeResult(
                node_id     = self.node_id,
                status      = NodeStatus.SUCCESS,
                outputs     = outputs or {},
                elapsed_sec = elapsed,
                metadata    = self._get_node_metadata(),
            )
            logger.info(
                "✔ 节点 [%s] 完成 (%.2fs)", self.node_id, elapsed
            )

        except Exception as exc:
            elapsed = time.time() - start_time
            self.status = NodeStatus.FAILED
            err_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            result = NodeResult(
                node_id     = self.node_id,
                status      = NodeStatus.FAILED,
                error_msg   = err_msg,
                elapsed_sec = elapsed,
            )
            logger.error("✘ 节点 [%s] 失败: %s", self.node_id, exc)
            logger.debug(traceback.format_exc())

        self._result = result
        return result

    # ------------------------------------------------------------------
    # 子类实现
    # ------------------------------------------------------------------

    @abstractmethod
    def _execute(self, bus: DataBus) -> Dict[str, Any]:
        """节点核心逻辑，子类必须实现"""
        raise NotImplementedError

    def _pre_check(self, bus: DataBus) -> None:
        """前置条件检查（可选 override）"""
        pass

    def _get_node_metadata(self) -> Dict:
        """返回节点元数据（可 override 添加算法特有信息）"""
        return {"node_type": type(self).__name__, "config": self.config}

    # ------------------------------------------------------------------
    # 属性与工具
    # ------------------------------------------------------------------

    @property
    def result(self) -> Optional[NodeResult]:
        return self._result

    @property
    def is_ready(self) -> bool:
        """依赖列表是否均已成功执行（由 Workflow 调用）"""
        return self.status == NodeStatus.PENDING

    def skip(self, reason: str = "") -> NodeResult:
        """将节点标记为跳过"""
        self.status = NodeStatus.SKIPPED
        self._result = NodeResult(
            node_id   = self.node_id,
            status    = NodeStatus.SKIPPED,
            metadata  = {"skip_reason": reason},
        )
        logger.info("⊙ 节点 [%s] 已跳过: %s", self.node_id, reason)
        return self._result

    def reset(self) -> None:
        """重置节点状态（用于重试）"""
        self.status  = NodeStatus.PENDING
        self._result = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(id='{self.node_id}', "
            f"status={self.status.value}, "
            f"deps={self.dependencies})"
        )


# ======================================================================
# 具体节点实现
# ======================================================================

class DataLoadNode(BaseNode):
    """
    数据加载节点

    Config Keys
    -----------
    file_path      : str   - 数据文件路径
    object_col     : str   - 评价对象所在列名（设为 index）
    indicator_cols : list  - 指标列名列表（None = 除 object_col 外全部）
    direction      : list  - 指标方向（1=正向, -1=负向，None=全正向）
    sheet_name     : str   - Excel Sheet 名称（csv 文件忽略）

    Outputs to Bus
    --------------
    raw_data       : DataFrame  (index=对象名, columns=指标名)
    indicator_direction : list  ([1,-1,1,...])
    object_names   : list
    indicator_names : list
    """

    def _execute(self, bus: DataBus) -> Dict[str, Any]:
        file_path      = self.config["file_path"]
        object_col     = self.config.get("object_col")
        indicator_cols = self.config.get("indicator_cols")
        direction      = self.config.get("direction")
        sheet_name     = self.config.get("sheet_name", 0)

        # ---- 读取文件 ----
        if file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            encoding = self.config.get("encoding", "utf-8")
            df = pd.read_csv(file_path, encoding=encoding)

        logger.info(
            "数据加载完成: %d 行 × %d 列", df.shape[0], df.shape[1]
        )

        # ---- 设置索引 ----
        if object_col and object_col in df.columns:
            df = df.set_index(object_col)
        elif object_col:
            logger.warning("列 '%s' 不存在，使用默认整数索引", object_col)

        # ---- 选取指标列 ----
        if indicator_cols:
            df = df[indicator_cols]
        else:
            df = df.select_dtypes(include=[np.number])

        df = df.astype(float)
        object_names    = list(df.index)
        indicator_names = list(df.columns)

        # ---- 处理方向 ----
        if direction is None:
            direction = [1] * len(indicator_names)
        if len(direction) != len(indicator_names):
            raise ValueError(
                f"direction 长度 ({len(direction)}) 与指标数 "
                f"({len(indicator_names)}) 不一致"
            )

        # ---- 写入总线 ----
        bus.put("raw_data",             df,             source_node=self.node_id)
        bus.put("indicator_direction",  direction,      source_node=self.node_id)
        bus.put("object_names",         object_names,   source_node=self.node_id)
        bus.put("indicator_names",      indicator_names, source_node=self.node_id)

        return {
            "raw_data":            df,
            "indicator_direction": direction,
            "n_objects":           len(object_names),
            "n_indicators":        len(indicator_names),
        }


class PreprocessNode(BaseNode):
    """
    数据预处理节点

    Config Keys
    -----------
    method     : "minmax" | "zscore" | "maxabs" (默认 minmax)
    handle_missing : "mean" | "median" | "drop" | "zero" (默认 mean)
    handle_outliers : bool (默认 False)
    outlier_std : float (默认 3.0)

    Inputs from Bus
    ---------------
    raw_data, indicator_direction

    Outputs to Bus
    --------------
    normalized_data   : DataFrame  (已正向化 + 标准化)
    """

    def _execute(self, bus: DataBus) -> Dict[str, Any]:
        df        = bus.get_required("raw_data",            self.node_id)
        direction = bus.get("indicator_direction", [1]*df.shape[1])

        method          = self.config.get("method",          "minmax")
        handle_missing  = self.config.get("handle_missing",  "mean")
        handle_outliers = self.config.get("handle_outliers", False)
        outlier_std     = self.config.get("outlier_std",     3.0)

        # 使用算法库中的预处理模块
        from src.algorithms.preprocess.cleaner   import MissingValueHandler
        from src.algorithms.preprocess.normalizer import Normalizer

        # Step 1: 缺失值处理
        mv_handler = MissingValueHandler(strategy=handle_missing)
        df_clean   = mv_handler.fit_transform(df)

        # Step 2: 异常值处理
        if handle_outliers:
            df_clean = self._clip_outliers(df_clean, n_std=outlier_std)

        # Step 3: 指标正向化（负向指标取倒数或极差变换）
        df_pos = self._apply_direction(df_clean, direction)

        # Step 4: 标准化
        normalizer  = Normalizer(method=method)
        df_norm     = normalizer.fit_transform(df_pos)

        bus.put("normalized_data", df_norm, source_node=self.node_id)
        bus.put("clean_data",      df_clean, source_node=self.node_id)

        logger.info(
            "预处理完成: method=%s, missing=%s", method, handle_missing
        )
        return {"normalized_data": df_norm}

    @staticmethod
    def _apply_direction(df: pd.DataFrame, direction: List[int]) -> pd.DataFrame:
        """负向指标转正向（极差取负+平移）"""
        df_pos = df.copy()
        for j, col in enumerate(df.columns):
            if direction[j] == -1:
                col_max = df[col].max()
                col_min = df[col].min()
                if abs(col_max - col_min) < 1e-12:
                    df_pos[col] = 0.0
                else:
                    df_pos[col] = col_max - df[col]
        return df_pos

    @staticmethod
    def _clip_outliers(df: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
        """基于标准差的异常值截断"""
        df_clip = df.copy()
        for col in df.columns:
            mean = df[col].mean()
            std  = df[col].std()
            if std < 1e-12:
                continue
            lower = mean - n_std * std
            upper = mean + n_std * std
            df_clip[col] = df[col].clip(lower, upper)
        return df_clip


class WeightNode(BaseNode):
    """
    权重计算节点

    Config Keys
    -----------
    method      : "entropy" | "critic" | "ahp" | "std_dev" | "combination"
    ahp_matrix  : list of list (仅 AHP 需要)
    combination_strategy : "multiplicative" | "linear" (仅 combination 需要)
    subjective_weights : list (仅 combination 需要)

    Inputs from Bus
    ---------------
    normalized_data

    Outputs to Bus
    --------------
    weights         : np.ndarray
    weight_method   : str
    weight_metadata : dict  (权重计算的详细中间结果)
    """

    def _execute(self, bus: DataBus) -> Dict[str, Any]:
        data   = bus.get_required("normalized_data", self.node_id)
        method = self.config.get("method", "entropy")

        from src.algorithms.weights.objective   import EntropyWeighter, CriticWeighter, StdDevWeighter
        from src.algorithms.weights.subjective  import AHPWeighter
        from src.algorithms.weights.combination import CombinationWeighter

        if method == "entropy":
            weighter = EntropyWeighter()
        elif method == "critic":
            weighter = CriticWeighter()
        elif method == "std_dev":
            weighter = StdDevWeighter()
        elif method == "ahp":
            ahp_matrix = self.config.get("ahp_matrix")
            if ahp_matrix is None:
                raise ValueError("AHP 方法需要提供 ahp_matrix 配置")
            weighter = AHPWeighter(judgment_matrix=ahp_matrix)
        elif method == "combination":
            subjective_w = self.config.get("subjective_weights")
            strategy     = self.config.get("combination_strategy", "multiplicative")
            weighter     = CombinationWeighter(
                strategy=strategy,
                subjective_weights=np.array(subjective_w) if subjective_w else None,
            )
        else:
            raise ValueError(f"未知赋权方法: {method}")

        weighter.fit(data)
        weights  = weighter.compute()
        metadata = weighter.get_metadata()

        bus.put("weights",         weights,  source_node=self.node_id)
        bus.put("weight_method",   method,   source_node=self.node_id)
        bus.put("weight_metadata", metadata, source_node=self.node_id)
        bus.put("weighter_obj",    weighter, source_node=self.node_id)

        indicator_names = bus.get("indicator_names", [])
        logger.info(
            "权重计算完成: method=%s | weights=%s",
            method,
            dict(zip(indicator_names, [round(w, 4) for w in weights])),
        )
        return {"weights": weights, "method": method}


class EvaluationNode(BaseNode):
    """
    综合评价节点

    Config Keys
    -----------
    method   : "topsis" | "vikor" | "gra" | "fuzzy" | "rsr" | "dea"
    v        : float  (VIKOR 偏好参数，默认 0.5)
    rho      : float  (GRA 分辨系数，默认 0.5)

    Inputs from Bus
    ---------------
    normalized_data, weights

    Outputs to Bus
    --------------
    scores        : pd.Series  (综合得分)
    ranks         : pd.Series  (综合排名)
    eval_metadata : dict
    evaluator_obj : BaseMethod
    """

    def _execute(self, bus: DataBus) -> Dict[str, Any]:
        data    = bus.get_required("normalized_data",  self.node_id)
        weights = bus.get_required("weights",          self.node_id)
        method  = self.config.get("method", "topsis")
        obj_names = bus.get("object_names", list(data.index))

        from src.algorithms.evaluation.topsis    import TOPSISEvaluator
        from src.algorithms.evaluation.vikor     import VIKOREvaluator
        from src.algorithms.evaluation.gra       import GRAEvaluator
        from src.algorithms.evaluation.fuzzy_eval import FuzzyEvaluator
        from src.algorithms.evaluation.rsr       import RSREvaluator
        from src.algorithms.evaluation.dea       import DEAEvaluator

        method_map = {
            "topsis": lambda: TOPSISEvaluator(),
            "vikor":  lambda: VIKOREvaluator(v=self.config.get("v", 0.5)),
            "gra":    lambda: GRAEvaluator(rho=self.config.get("rho", 0.5)),
            "fuzzy":  lambda: FuzzyEvaluator(),
            "rsr":    lambda: RSREvaluator(),
            "dea":    lambda: DEAEvaluator(),
        }
        if method not in method_map:
            raise ValueError(f"未知评价方法: {method}")

        evaluator = method_map[method]()
        evaluator.fit(data, weights)
        result    = evaluator.compute()

        scores   = result.get("scores")
        ranks    = result.get("ranks")
        metadata = evaluator.get_metadata()

        # 转为 Series
        if not isinstance(scores, pd.Series):
            scores = pd.Series(scores, index=obj_names, name="综合得分")
        if not isinstance(ranks, pd.Series):
            ranks  = pd.Series(ranks,  index=obj_names, name="排名")

        bus.put("scores",        scores,    source_node=self.node_id)
        bus.put("ranks",         ranks,     source_node=self.node_id)
        bus.put("eval_metadata", metadata,  source_node=self.node_id)
        bus.put("evaluator_obj", evaluator, source_node=self.node_id)

        logger.info(
            "评价完成: method=%s | 第1名=%s",
            method, scores.idxmax() if not scores.empty else "N/A"
        )
        return {"scores": scores, "ranks": ranks, "method": method}


class SensitivityNode(BaseNode):
    """
    灵敏度分析节点

    Config Keys
    -----------
    perturbation_range : float (默认 0.3)
    n_steps            : int   (默认 21)
    run_rank_consistency : bool (默认 True，多方法时)
    comparison_methods : list  (用于一致性检验的额外方法名称)

    Inputs from Bus
    ---------------
    normalized_data, weights, scores, ranks, evaluator_obj

    Outputs to Bus
    --------------
    sensitivity_result   : dict
    consistency_result   : dict (如果执行一致性检验)
    """

    def _execute(self, bus: DataBus) -> Dict[str, Any]:
        data      = bus.get_required("normalized_data",  self.node_id)
        weights   = bus.get_required("weights",          self.node_id)
        evaluator = bus.get("evaluator_obj")
        base_ranks = bus.get("ranks")

        perturb_range = self.config.get("perturbation_range", 0.3)
        n_steps       = self.config.get("n_steps",            21)

        from src.algorithms.sensitivity.weight_sensitivity import WeightSensitivityAnalyzer

        # 构建评价函数包装
        def eval_fn(data_mat: np.ndarray, w: np.ndarray):
            tmp_data = pd.DataFrame(
                data_mat,
                index=data.index,
                columns=data.columns,
            )
            evaluator.fit(tmp_data, w)
            result = evaluator.compute()
            scores = np.array(result.get("scores", np.zeros(len(data))))
            ranks  = np.array(result.get("ranks",  np.arange(1, len(data)+1)))
            return scores, ranks

        analyzer = WeightSensitivityAnalyzer(
            eval_function     = eval_fn,
            perturbation_range = perturb_range,
            n_steps           = n_steps,
        )
        analyzer.fit(
            data,
            weights,
            indicator_names = bus.get("indicator_names"),
            object_names    = bus.get("object_names"),
        )
        sens_result = analyzer.compute()

        bus.put("sensitivity_result", sens_result, source_node=self.node_id)
        bus.put("sensitivity_analyzer", analyzer,  source_node=self.node_id)

        logger.info("灵敏度分析完成")
        return {"sensitivity_result": sens_result}


class ConsolidationNode(BaseNode):
    """
    结果汇总节点

    将所有中间结果整合为统一的输出字典，
    供生成器（LaTeX / Code / Plot）使用。

    Outputs to Bus
    --------------
    final_report_data : dict  (所有结果的结构化汇总)
    """

    def _execute(self, bus: DataBus) -> Dict[str, Any]:
        report_data = {
            "metadata": {
                "object_names":    bus.get("object_names",    []),
                "indicator_names": bus.get("indicator_names", []),
                "direction":       bus.get("indicator_direction", []),
            },
            "data": {
                "raw":        bus.get("raw_data"),
                "clean":      bus.get("clean_data"),
                "normalized": bus.get("normalized_data"),
            },
            "weights": {
                "method":   bus.get("weight_method"),
                "values":   bus.get("weights"),
                "metadata": bus.get("weight_metadata"),
            },
            "evaluation": {
                "scores":   bus.get("scores"),
                "ranks":    bus.get("ranks"),
                "metadata": bus.get("eval_metadata"),
            },
            "sensitivity": {
                "result":   bus.get("sensitivity_result"),
            },
            "consistency": {
                "result":   bus.get("consistency_result"),
            },
        }

        bus.put("final_report_data", report_data, source_node=self.node_id)
        logger.info("结果汇总完成，共 %d 项输出", len(report_data))
        return {"final_report_data": report_data}