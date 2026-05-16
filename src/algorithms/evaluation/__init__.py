"""
algorithms/evaluation/__init__.py
综合评价模型模块统一导出
"""

from .topsis import TOPSIS
from .vikor import VIKOR
from .gra import GRA
from .fuzzy_eval import FuzzyComprehensiveEvaluation
from .electre import ELECTRE
from .rsr import RSR
from .dea import DEA
from .dynamic import DynamicTOPSIS, DynamicGRA, DynamicEvaluation

__all__ = [
    "TOPSIS",
    "VIKOR",
    "GRA",
    "FuzzyComprehensiveEvaluation",
    "ELECTRE",
    "RSR",
    "DEA",
    "DynamicTOPSIS",
    "DynamicGRA",
    "DynamicEvaluation",
]

# 方法名到类的映射表，供工作流引擎动态实例化
METHOD_REGISTRY: dict = {
    "topsis": TOPSIS,
    "vikor": VIKOR,
    "gra": GRA,
    "fuzzy": FuzzyComprehensiveEvaluation,
    "electre": ELECTRE,
    "rsr": RSR,
    "dea": DEA,
    "dynamic_topsis": DynamicTOPSIS,
    "dynamic_gra": DynamicGRA,
}


def get_method(name: str):
    """根据方法名获取对应的评价类。"""
    key = name.lower().strip()
    if key not in METHOD_REGISTRY:
        raise KeyError(
            f"未知评价方法: '{name}'。可用方法: {list(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[key]