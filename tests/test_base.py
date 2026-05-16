# tests/test_base.py
"""基础模块单元测试"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from algorithms.base import (
    BaseMethod,
    MethodResult,
    DataProfile,
    IndicatorDirection,
    MethodCategory,
)
from utils.logging_config import setup_logger, get_logger


# ============================================================
#  具体算法子类（用于测试 BaseMethod 接口）
# ============================================================

class MockEvalMethod(BaseMethod):
    """最简测试算法：直接对行求和作为得分"""
    CATEGORY = MethodCategory.EVALUATION
    METHOD_NAME_ZH = "测试算法"
    METHOD_NAME_EN = "MockMethod"
    METHOD_ABBR = "MOCK"

    def fit(self, data: pd.DataFrame, weights=None, **kwargs):
        self.validate_input(data)
        self._raw_data = data.copy()
        self._indicator_names = list(data.columns)
        self._object_names = list(data.index)
        self._weights = weights
        self._is_fitted = True
        return self

    def compute(self) -> MethodResult:
        self._check_fitted()
        t0 = self._start_timer()

        scores = self._raw_data.sum(axis=1)
        scores.name = "mock_score"
        rankings = self._rank_scores(scores, ascending=False)

        result = self._build_result(
            scores=scores,
            rankings=rankings,
            elapsed=self._stop_timer(t0),
        )
        self._result = result
        return result

    def summary(self) -> str:
        r = self.get_result()
        text = f"MockMethod 结果摘要:\n{r.get_summary_dataframe()}"
        print(text)
        return text

    def tex_description(self) -> str:
        return r"\subsection{测试算法}\n本算法直接对指标求和。"


# ============================================================
#  测试用例
# ============================================================

class TestBaseMethod:
    """测试 BaseMethod 基类通用功能"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """创建标准测试数据集（5个对象，3个指标）"""
        return pd.DataFrame(
            {
                "指标1": [0.8, 0.6, 0.9, 0.4, 0.7],
                "指标2": [0.7, 0.8, 0.6, 0.9, 0.5],
                "指标3": [0.6, 0.7, 0.8, 0.5, 0.9],
            },
            index=["城市A", "城市B", "城市C", "城市D", "城市E"],
        )

    @pytest.fixture
    def method(self) -> MockEvalMethod:
        return MockEvalMethod(language="zh")

    def test_init(self, method):
        """测试初始化状态"""
        assert not method.is_fitted
        assert method.language == "zh"
        assert method._result is None

    def test_fit_returns_self(self, method, sample_data):
        """fit() 返回 self，支持链式调用"""
        result = method.fit(sample_data)
        assert result is method

    def test_fit_state_change(self, method, sample_data):
        """fit() 后状态变更"""
        method.fit(sample_data)
        assert method.is_fitted
        assert len(method._indicator_names) == 3
        assert len(method._object_names) == 5

    def test_compute_returns_result(self, method, sample_data):
        """compute() 返回 MethodResult"""
        result = method.fit(sample_data).compute()
        assert isinstance(result, MethodResult)
        assert result.scores is not None
        assert len(result.scores) == 5

    def test_rankings_are_integers(self, method, sample_data):
        """排名必须为整数"""
        result = method.fit_compute(sample_data)
        assert result.rankings is not None
        assert result.rankings.dtype in [np.int32, np.int64]

    def test_rankings_range(self, method, sample_data):
        """排名范围应为 1~n"""
        result = method.fit_compute(sample_data)
        assert result.rankings.min() == 1
        assert result.rankings.max() == 5

    def test_compute_before_fit_raises(self, method):
        """未 fit 直接 compute 应抛出 RuntimeError"""
        with pytest.raises(RuntimeError, match="fit"):
            method.compute()

    def test_get_result_before_compute_raises(self, method, sample_data):
        """未 compute 调用 get_result 应抛出 RuntimeError"""
        method.fit(sample_data)
        with pytest.raises(RuntimeError):
            method.get_result()

    def test_validate_empty_dataframe(self, method):
        """空 DataFrame 应触发 ValueError"""
        with pytest.raises(ValueError, match="空"):
            method.fit(pd.DataFrame())

    def test_validate_non_dataframe(self, method):
        """非 DataFrame 输入应触发 TypeError"""
        with pytest.raises(TypeError):
            method.fit([[1, 2], [3, 4]])

    def test_validate_non_numeric_columns(self, method):
        """含非数值列应触发 ValueError"""
        df = pd.DataFrame({
            "col1": [1.0, 2.0],
            "text_col": ["a", "b"],
        })
        with pytest.raises(ValueError, match="非数值"):
            method.fit(df)

    def test_reset(self, method, sample_data):
        """reset() 后状态归零"""
        method.fit_compute(sample_data)
        method.reset()
        assert not method.is_fitted
        assert method._result is None

    def test_clone(self, method, sample_data):
        """clone() 创建独立副本"""
        method.fit(sample_data)
        cloned = method.clone()
        assert not cloned.is_fitted          # 克隆后状态重置
        assert cloned is not method           # 不同对象

    def test_normalize_weights(self, method):
        """权重归一化"""
        w = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = method._normalize_weights(w)
        assert abs(normalized.sum() - 1.0) < 1e-10
        assert np.all(normalized >= 0)

    def test_normalize_weights_negative_raises(self, method):
        """负权重应抛出 ValueError"""
        with pytest.raises(ValueError, match="负值"):
            method._normalize_weights([-1.0, 2.0, 3.0])

    def test_safe_log(self, method):
        """安全对数不产生 -inf"""
        arr = np.array([0.0, 0.5, 1.0])
        result = method._safe_log(arr)
        assert np.all(np.isfinite(result))

    def test_safe_divide(self, method):
        """安全除法处理零分母"""
        num = np.array([1.0, 2.0, 3.0])
        den = np.array([2.0, 0.0, 1.5])
        result = method._safe_divide(num, den, fill_value=0.0)
        assert result[1] == 0.0              # 零分母位置填充 0
        assert abs(result[0] - 0.5) < 1e-10
        assert abs(result[2] - 2.0) < 1e-10

    def test_method_info(self, method):
        """method_info 包含必要字段"""
        info = method.method_info
        assert "name_zh" in info
        assert "name_en" in info
        assert "category" in info

    def test_tex_description(self, method, sample_data):
        """tex_description 返回非空字符串"""
        method.fit_compute(sample_data)
        tex = method.tex_description()
        assert isinstance(tex, str)
        assert len(tex) > 0

    def test_result_to_dict(self, method, sample_data):
        """MethodResult 可序列化为字典"""
        result = method.fit_compute(sample_data)
        d = result.to_dict()
        assert "method_name" in d
        assert "scores" in d
        assert "rankings" in d

    def test_elapsed_time_positive(self, method, sample_data):
        """计算耗时应为正数"""
        result = method.fit_compute(sample_data)
        assert result.elapsed_time >= 0


class TestMethodResult:
    """测试 MethodResult 数据类"""

    def test_get_summary_dataframe(self):
        """get_summary_dataframe 返回正确 DataFrame"""
        scores = pd.Series([0.8, 0.6, 0.9], index=["A", "B", "C"], name="s")
        rankings = pd.Series([2, 3, 1], index=["A", "B", "C"], name="r")
        result = MethodResult(
            method_name="Test",
            scores=scores,
            rankings=rankings,
        )
        df = result.get_summary_dataframe()
        assert df.shape == (3, 2)
        assert "A" in df.index

    def test_add_warning(self):
        """add_warning 正常追加"""
        result = MethodResult(method_name="Test")
        result.add_warning("测试警告")
        assert len(result.warnings) == 1
        assert "测试警告" in result.warnings[0]


class TestDataProfile:
    """测试 DataProfile 数据类"""

    def test_to_dict(self):
        profile = DataProfile(n_samples=5, n_indicators=3)
        d = profile.to_dict()
        assert d["n_samples"] == 5
        assert d["n_indicators"] == 3


class TestLogging:
    """测试日志模块"""

    def test_get_logger_returns_logger(self):
        import logging
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_same_name_same_instance(self):
        l1 = get_logger("singleton_test")
        l2 = get_logger("singleton_test")
        assert l1 is l2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])