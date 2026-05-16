"""
AutoEval-Modeling: 问题情境解析模块 (Parser)

本模块负责：
1. 解析用户自然语言描述的评价问题
2. 识别评价情境类型并生成情境标签
3. 自动分析上传数据的统计特征（缺失率、相关性、分布等）
4. 为下游算法推荐器提供结构化的问题特征向量
"""

from .context_type import (
    ContextType,
    ContextFeature,
    CONTEXT_REGISTRY,
    get_context_by_tag,
)
from .nlp_parser import NLPParser, ProblemDescription
from history.data_profiler import DataProfiler, DataProfile

__all__ = [
    "ContextType",
    "ContextFeature",
    "CONTEXT_REGISTRY",
    "get_context_by_tag",
    "NLPParser",
    "ProblemDescription",
    "DataProfiler",
    "DataProfile",
]
