# src/__init__.py
"""
AutoEval-Modeling 核心包
面向评价类数学建模的自动化工作流引擎与代码生成系统
"""

__version__ = "0.1.0"
__author__ = "AutoEval Team"
__description__ = "面向评价类数学建模的自动化工作流引擎与代码生成系统"

# 包级别可用常量
SUPPORTED_WEIGHT_METHODS = [
    "ahp", "entropy", "critic", "std_deviation",
    "combination_multiply", "combination_game"
]

SUPPORTED_EVAL_METHODS = [
    "topsis", "vikor", "gra", "fuzzy",
    "electre", "rsr", "dea", "dynamic"
]

SUPPORTED_PREPROCESS_METHODS = [
    "minmax", "zscore", "vector", "pca", "fa"
]