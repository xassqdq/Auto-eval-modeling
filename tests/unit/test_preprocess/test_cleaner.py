# tests/unit/test_preprocess/test_cleaner.py

import pytest
import numpy as np
import pandas as pd
from src.algorithms.preprocess.cleaner import DataCleaner


class TestMissingValueHandler:
    """缺失值处理测试"""

    def test_detect_missing_rate(self, sample_with_missing):
        cleaner = DataCleaner(sample_with_missing)
        rates = cleaner.missing_rate()
        assert isinstance(rates, pd.Series)
        assert all(0 <= r <= 1 for r in rates)
        # X1有1个缺失, 共5行
        assert abs(rates["X1"] - 0.2) < 1e-10

    def test_drop_high_missing_columns(self, sample_with_missing):
        """缺失率超过阈值的列应被删除"""
        cleaner = DataCleaner(sample_with_missing)
        result = cleaner.drop_high_missing(threshold=0.15)
        # 所有列缺失率=0.2 > 0.15, 应全部删除 → 空DataFrame
        assert result.shape[1] == 0

    def test_interpolate_fills_all(self, sample_with_missing):
        """线性插值后不应有缺失值"""
        cleaner = DataCleaner(sample_with_missing)
        result = cleaner.fill_missing(method="interpolate")
        assert result.isnull().sum().sum() == 0

    def test_mean_fill(self, sample_with_missing):
        """均值填充后, 填充值应等于该列非缺失均值"""
        cleaner = DataCleaner(sample_with_missing)
        result = cleaner.fill_missing(method="mean")
        original_mean_x1 = sample_with_missing["X1"].mean()  # 跳过nan
        assert abs(result.loc["B", "X1"] - original_mean_x1) < 1e-10

    def test_output_shape_preserved(self, sample_with_missing):
        """填充后行列数不变"""
        cleaner = DataCleaner(sample_with_missing)
        result = cleaner.fill_missing(method="median")
        assert result.shape == sample_with_missing.shape

    def test_no_missing_passthrough(self, sample_3x4):
        """无缺失数据应原样通过"""
        cleaner = DataCleaner(sample_3x4)
        result = cleaner.fill_missing(method="mean")
        pd.testing.assert_frame_equal(result, sample_3x4)


class TestOutlierHandler:
    """异常值处理测试"""

    def test_zscore_detection(self, sample_with_outliers):
        cleaner = DataCleaner(sample_with_outliers)
        outliers = cleaner.detect_outliers(method="zscore", threshold=3.0)
        # 应至少检测到我们手动插入的2个异常值
        assert outliers.sum().sum() >= 2

    def test_iqr_detection(self, sample_with_outliers):
        cleaner = DataCleaner(sample_with_outliers)
        outliers = cleaner.detect_outliers(method="iqr", factor=1.5)
        assert isinstance(outliers, pd.DataFrame)
        assert outliers.shape == sample_with_outliers.shape

    def test_clip_outliers_bounded(self, sample_with_outliers):
        """截断后所有值应在合理范围内"""
        cleaner = DataCleaner(sample_with_outliers)
        result = cleaner.handle_outliers(method="clip", factor=3.0)
        assert result.abs().max().max() < 100  # 原始异常值=100

    def test_winsorize_preserves_shape(self, sample_with_outliers):
        cleaner = DataCleaner(sample_with_outliers)
        result = cleaner.handle_outliers(method="winsorize", limits=(0.05, 0.05))
        assert result.shape == sample_with_outliers.shape


# tests/unit/test_preprocess/test_normalizer.py

import pytest
import numpy as np
import pandas as pd
from src.algorithms.preprocess.normalizer import Normalizer


class TestMinMaxNormalization:
    """MinMax 归一化测试"""

    def test_range_01(self, sample_3x4):
        norm = Normalizer(method="minmax")
        result = norm.fit_transform(sample_3x4)
        assert result.min().min() >= 0.0 - 1e-10
        assert result.max().max() <= 1.0 + 1e-10

    def test_min_maps_to_0(self, sample_3x4):
        norm = Normalizer(method="minmax")
        result = norm.fit_transform(sample_3x4)
        for col in result.columns:
            assert abs(result[col].min() - 0.0) < 1e-10

    def test_max_maps_to_1(self, sample_3x4):
        norm = Normalizer(method="minmax")
        result = norm.fit_transform(sample_3x4)
        for col in result.columns:
            assert abs(result[col].max() - 1.0) < 1e-10

    def test_constant_column_handling(self):
        """常数列应输出全0或抛出警告"""
        df = pd.DataFrame({"A": [5, 5, 5], "B": [1, 2, 3]})
        norm = Normalizer(method="minmax")
        result = norm.fit_transform(df)
        assert all(result["A"] == 0) or all(result["A"] == 0.5)


class TestZScoreNormalization:
    """Z-Score 标准化测试"""

    def test_mean_zero(self, sample_3x4):
        norm = Normalizer(method="zscore")
        result = norm.fit_transform(sample_3x4)
        for col in result.columns:
            assert abs(result[col].mean()) < 1e-10

    def test_std_one(self, sample_3x4):
        norm = Normalizer(method="zscore")
        result = norm.fit_transform(sample_3x4)
        for col in result.columns:
            assert abs(result[col].std(ddof=0) - 1.0) < 1e-6


class TestDirectionTransform:
    """指标正向化测试"""

    def test_negative_to_positive(self, sample_3x4, directions_mixed):
        norm = Normalizer(method="minmax")
        result = norm.fit_transform(sample_3x4, directions=directions_mixed)
        # 负向指标正向化后, 原来最大值变最小
        original_max_idx = sample_3x4["指标2"].idxmax()
        assert result.loc[original_max_idx, "指标2"] == pytest.approx(0.0, abs=1e-10)

    def test_moderate_indicator(self, directions_mixed):
        """适度指标: 越接近最优值越好"""
        df = pd.DataFrame({"指标4": [10, 15, 20, 25]})
        norm = Normalizer(method="minmax")
        result = norm.fit_transform(df, directions={"指标4": "moderate"},
                                     moderate_best={"指标4": 15})
        # 15应该得到最高分
        assert result["指标4"].idxmax() == 1


# tests/unit/test_preprocess/test_reduction.py

import pytest
import numpy as np
import pandas as pd
from src.algorithms.preprocess.reduction import PCAReducer


class TestPCAReducer:

    def test_variance_threshold(self, sample_10x8):
        reducer = PCAReducer(variance_threshold=0.85)
        result = reducer.fit_transform(sample_10x8)
        assert result.shape[1] <= sample_10x8.shape[1]
        assert reducer.explained_variance_ratio_.sum() >= 0.85

    def test_fixed_components(self, sample_10x8):
        reducer = PCAReducer(n_components=3)
        result = reducer.fit_transform(sample_10x8)
        assert result.shape == (10, 3)

    def test_loadings_shape(self, sample_10x8):
        reducer = PCAReducer(n_components=3)
        reducer.fit_transform(sample_10x8)
        loadings = reducer.get_loadings()
        assert loadings.shape == (3, 8)

    def test_single_component_sufficient(self):
        """高度相关数据, 1个主成分应捕获大部分方差"""
        np.random.seed(42)
        x = np.random.randn(50)
        df = pd.DataFrame({"A": x, "B": x * 2 + 0.01 * np.random.randn(50),
                           "C": x * 0.5 + 0.01 * np.random.randn(50)})
        reducer = PCAReducer(variance_threshold=0.95)
        result = reducer.fit_transform(df)
        assert result.shape[1] == 1