# tests/conftest.py

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ============================================================
# 1. 标准数据集 Fixtures
# ============================================================

@pytest.fixture
def sample_3x4():
    """最小规模标准数据：3个评价对象, 4个指标"""
    return pd.DataFrame({
        "对象": ["A", "B", "C"],
        "指标1": [80, 90, 70],
        "指标2": [0.7, 0.5, 0.9],
        "指标3": [300, 500, 200],
        "指标4": [15, 10, 20],
    }).set_index("对象")


@pytest.fixture
def sample_10x8():
    """中等规模数据：10个对象, 8个指标"""
    return pd.read_csv(FIXTURES_DIR / "sample_10x8.csv", index_col=0)


@pytest.fixture
def sample_with_missing():
    """含缺失值的数据"""
    df = pd.DataFrame({
        "对象": ["A", "B", "C", "D", "E"],
        "X1": [1.0, np.nan, 3.0, 4.0, 5.0],
        "X2": [2.0, 3.0, np.nan, 5.0, 6.0],
        "X3": [3.0, 4.0, 5.0, np.nan, 7.0],
    }).set_index("对象")
    return df


@pytest.fixture
def sample_with_outliers():
    """含异常值的数据"""
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(20, 5),
        columns=[f"X{i}" for i in range(1, 6)]
    )
    df.iloc[0, 0] = 100   # 极端异常值
    df.iloc[5, 3] = -80
    return df


@pytest.fixture
def sample_timeseries():
    """多时段动态评价数据"""
    return pd.read_csv(FIXTURES_DIR / "sample_timeseries.csv")


# ============================================================
# 2. 指标方向配置 Fixtures
# ============================================================

@pytest.fixture
def directions_all_positive():
    """所有指标均为正向"""
    return {"指标1": "positive", "指标2": "positive",
            "指标3": "positive", "指标4": "positive"}


@pytest.fixture
def directions_mixed():
    """混合方向指标"""
    return {"指标1": "positive", "指标2": "negative",
            "指标3": "positive", "指标4": "moderate"}


# ============================================================
# 3. AHP 判断矩阵 Fixtures
# ============================================================

@pytest.fixture
def ahp_matrix_consistent():
    """一致性良好的 3x3 判断矩阵 (CR < 0.1)"""
    return np.array([
        [1,   2,   5],
        [1/2, 1,   3],
        [1/5, 1/3, 1]
    ])


@pytest.fixture
def ahp_matrix_inconsistent():
    """一致性不合格的判断矩阵 (CR > 0.1)"""
    return np.array([
        [1,   9,   1/3],
        [1/9, 1,   7],
        [3,   1/7, 1]
    ])


# ============================================================
# 4. 预期结果基线 Fixtures
# ============================================================

@pytest.fixture
def expected_entropy_weights():
    """手算验证的熵权法权重基线"""
    with open(FIXTURES_DIR / "expected_results/entropy_weights.json") as f:
        return json.load(f)


@pytest.fixture
def expected_topsis_scores():
    with open(FIXTURES_DIR / "expected_results/topsis_scores.json") as f:
        return json.load(f)


# ============================================================
# 5. 工作流配置 Fixtures
# ============================================================

@pytest.fixture
def minimal_config():
    """最小有效工作流配置"""
    return {
        "data_source": "sample_3x4.csv",
        "index_col": "对象",
        "directions": {"指标1": "positive", "指标2": "positive",
                       "指标3": "positive", "指标4": "negative"},
        "pipeline": [
            {"step": "normalize", "method": "minmax"},
            {"step": "weight", "method": "entropy"},
            {"step": "evaluate", "method": "topsis"},
        ],
        "output": {"format": ["csv", "latex"], "lang": "zh"}
    }


@pytest.fixture
def full_pipeline_config():
    """完整管道配置"""
    return {
        "data_source": "sample_10x8.csv",
        "index_col": 0,
        "directions": "auto_detect",
        "pipeline": [
            {"step": "clean", "method": "interpolate"},
            {"step": "normalize", "method": "zscore"},
            {"step": "reduce", "method": "pca", "params": {"variance_threshold": 0.85}},
            {"step": "weight", "method": "critic"},
            {"step": "evaluate", "method": "topsis"},
            {"step": "sensitivity", "method": "oat", "params": {"delta": 0.1}},
        ],
        "output": {"format": ["csv", "latex", "png"], "lang": "zh"}
    }


# ============================================================
# 6. 临时输出目录
# ============================================================

@pytest.fixture
def tmp_output(tmp_path):
    """创建标准化的临时输出目录结构"""
    (tmp_path / "reports").mkdir()
    (tmp_path / "figures").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "logs").mkdir()
    return tmp_path