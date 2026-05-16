# src/algorithms/__init__.py
"""
评价算法库
所有算法统一实现 BaseMethod 接口
"""

from .base import (
    BaseMethod,
    MethodResult,
    DataProfile,
    IndicatorDirection,
    MethodCategory,
)

__all__ = [
    "BaseMethod",
    "MethodResult",
    "DataProfile",
    "IndicatorDirection",
    "MethodCategory",
]