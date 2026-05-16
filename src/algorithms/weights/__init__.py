# -*- coding: utf-8 -*-
"""
algorithms/weights/__init__.py
赋权方法模块统一入口

主观赋权 (Subjective):
    AHPMethod              — 层次分析法
    BinomialCoefficientMethod — 二项系数法
    RingRatioScoringMethod — 环比评分法

客观赋权 (Objective):
    EntropyWeightMethod    — 熵权法
    CRITICMethod           — CRITIC法
    StdDeviationMethod     — 标准离差法
    PCAWeightMethod        — 主成分分析权重

组合赋权 (Combination):
    MultiplicativeCombination  — 乘法合成归一化
    LinearCombination          — 线性加权组合
    GameTheoryCombination      — 博弈论组合赋权
    MinDeviationCombination    — 离差最小化组合赋权
"""

from .subjective import (
    AHPMethod,
    BinomialCoefficientMethod,
    RingRatioScoringMethod,
)
from .objective import (
    EntropyWeightMethod,
    CRITICMethod,
    StdDeviationMethod,
    PCAWeightMethod,
)
from .combination import (
    MultiplicativeCombination,
    LinearCombination,
    GameTheoryCombination,
    MinDeviationCombination,
)

__all__ = [
    # Subjective
    "AHPMethod",
    "BinomialCoefficientMethod",
    "RingRatioScoringMethod",
    # Objective
    "EntropyWeightMethod",
    "CRITICMethod",
    "StdDeviationMethod",
    "PCAWeightMethod",
    # Combination
    "MultiplicativeCombination",
    "LinearCombination",
    "GameTheoryCombination",
    "MinDeviationCombination",
]

# ── 快捷工厂字典（供推荐器动态实例化使用）──
WEIGHT_METHOD_REGISTRY: dict = {
    "ahp":             AHPMethod,
    "binomial":        BinomialCoefficientMethod,
    "ring_ratio":      RingRatioScoringMethod,
    "entropy":         EntropyWeightMethod,
    "critic":          CRITICMethod,
    "std_deviation":   StdDeviationMethod,
    "pca":             PCAWeightMethod,
    "multiplicative":  MultiplicativeCombination,
    "linear":          LinearCombination,
    "game_theory":     GameTheoryCombination,
    "min_deviation":   MinDeviationCombination,
}


def get_weight_method(name: str, **kwargs):
    """
    按名称获取赋权方法实例。

    Parameters
    ----------
    name : str
        方法标识键（见 WEIGHT_METHOD_REGISTRY）
    **kwargs :
        传递给构造函数的参数

    Returns
    -------
    BaseMethod 实例

    Examples
    --------
    >>> method = get_weight_method("entropy")
    >>> method = get_weight_method("ahp", cr_threshold=0.1)
    """
    name_lower = name.strip().lower()
    if name_lower not in WEIGHT_METHOD_REGISTRY:
        available = list(WEIGHT_METHOD_REGISTRY.keys())
        raise KeyError(
            f"未知赋权方法: '{name}'。\n"
            f"可用方法: {available}"
        )
    return WEIGHT_METHOD_REGISTRY[name_lower](**kwargs)