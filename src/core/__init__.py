"""
工作流引擎核心包

提供：
    DataBus          - 节点间数据传递总线
    BaseNode         - 工作流节点基类
    各具体节点类      - PreprocessNode, WeightNode, EvalNode 等
    Workflow         - DAG 工作流引擎
    CodeSandbox      - 受控代码执行沙箱
    AlgorithmRecommender - 算法知识图谱推荐器
"""

from .data_bus import DataBus
from .node import (
    BaseNode,
    DataLoadNode,
    PreprocessNode,
    WeightNode,
    EvaluationNode,
    SensitivityNode,
    ConsolidationNode,
)
from .workflow import Workflow, WorkflowStatus
from .sandbox import CodeSandbox
from .recommendation import AlgorithmRecommender

__all__ = [
    "DataBus",
    "BaseNode",
    "DataLoadNode",
    "PreprocessNode",
    "WeightNode",
    "EvaluationNode",
    "SensitivityNode",
    "ConsolidationNode",
    "Workflow",
    "WorkflowStatus",
    "CodeSandbox",
    "AlgorithmRecommender",
]