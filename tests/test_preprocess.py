# tests/test_preprocess.py
"""
预处理模块完整单元测试
覆盖: cleaner / normalizer / reduction / data_profiler
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from algorithms.preprocess import (
    MissingValueHandler,
    OutlierHandler,
    DataCleaner,
    PositivityTransformer,
    MinMaxNormalizer,
    ZScoreNormalizer,
    VectorNormalizer,
    SumNormalizer,
    DataNormalizer,
    CorrelationAnalyzer,
    PCAReducer,
)
from algorithms.base import IndicatorDirection
from parser.data_profiler import DataProfiler


# ============================================================
#  Fixtures（共用测试数据）
# ============================================================

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """标准 5×4 测试数据（无缺失，无异常）"""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "指标A": [80.0, 60.0, 90.0, 40.0, 70.0],
            "指标B": [0.7,  0.8,  0.6,  0.9,  0.5],
            "指标C": [120,  98,   145,  76,   110],
            "指标D": [3.2,  4.1,  2.8,  5.5,  3.8],
        },
        index=["城市A", "城市B", "城市C", "城市D", "城市E"],
    )


@pytest.fixture
def data_with_missing(sample_data: pd.DataFrame) -> pd.DataFrame:
    df = sample_data.copy()
    df.loc["城市B", "指标A"] = np.nan
    df.loc["城市D", "指标C"] = np.nan
    df.loc["城市E", "指标B"] = np.nan
    return df


@pytest.fixture
def data_with_outliers(sample_data: pd.DataFrame) -> pd.DataFrame:
    df = sample_data.copy()
    df.loc["城市A", "指标A"] = 9999.0   # 极端异常值
    df.loc["城市E", "指标D"] = -100.0   # 极端负异常值
    return df


@pytest.fixture
def data_with_negative() -> pd.DataFrame:
    """含负向指标和适度型指标的测试数据"""
    return pd.DataFrame(
        {
            "正向A": [80.0, 60.0, 90.0, 40.0, 70.0],   # positive
            "负向B": [10.0, 20.0, 5.0,  30.0, 15.0],   # negative（越小越好）
            "适度C": [50.0, 30.0, 60.0, 80.0, 45.0],   # moderate（最优=50）
        },
        index=["方案1", "方案2", "方案3", "方案4", "方案5"],
    )


@pytest.fixture
def high_corr_data() -> pd.DataFrame:
    """构造高相关指标对的数据"""
    np.random.seed(0)
    x = np.random.randn(20)
    return pd.DataFrame({
        "X1": x,
        "X2": x + np.random.randn(20) * 0.1,  # 与 X1 高度相关
        "X3": np.random.randn(20),             # 独立
        "X4": -x + np.random.randn(20) * 0.05,# 与 X1 高度负相关
    })


# ============================================================
#  MissingValueHandler 测试
# ============================================================

class TestMissingValueHandler:

    def test_mean_fill(self, data_with_missing):
        mvh = MissingValueHandler()
        result = mvh.fit(data_with_missing, strategy="mean").compute()
        output = result.metadata["output_data"]
        assert output.isnull().sum().sum() == 0

    def test_median_fill(self, data_with_missing):
        mvh = MissingValueHandler()
        result = mvh.fit(data_with_missing, strategy="median").compute()
        assert result.metadata["output_data"].isnull().sum().sum() == 0

    def test_constant_fill(self, data_with_missing):
        mvh = MissingValueHandler()
        result = mvh.fit(data_with_missing, strategy="constant", fill_value=-1.0).compute()
        output = result.metadata["output_data"]
        assert output.isnull().sum().sum() == 0

    def test_interpolate_fill(self, data_with_missing):
        mvh = MissingValueHandler()
        result = mvh.fit(data_with_missing, strategy="interpolate").compute()
        assert result.metadata["output_data"].isnull().sum().sum() == 0

    def test_drop_rows(self, data_with_missing):
        mvh = MissingValueHandler()
        result = mvh.fit(data_with_missing, strategy="drop_rows").compute()
        output = result.metadata["output_data"]
        assert output.isnull().sum().sum() == 0
        assert len(output) < len(data_with_missing)

    def test_drop_cols(self, data_with_missing):
        # 设置极低阈值，所有含缺失的列都被删除
        mvh = MissingValueHandler()
        result = mvh.fit(
            data_with_missing, strategy="drop_cols", drop_threshold=0.0
        ).compute()
        output = result.metadata["output_data"]
        assert output.isnull().sum().sum() == 0

    def test_no_missing_data(self, sample_data):
        """无缺失时应正常返回，missing_before=0"""
        mvh = MissingValueHandler()
        result = mvh.fit(sample_data, strategy="mean").compute()
        assert result.scalars["missing_before"] == 0

    def test_invalid_strategy_raises(self, sample_data):
        with pytest.raises(ValueError, match="strategy"):
            MissingValueHandler().fit(sample_data, strategy="magic_fill")

    def test_scalars_keys(self, data_with_missing):
        result = MissingValueHandler().fit(data_with_missing).compute()
        for key in ["missing_before", "missing_after", "strategy"]:
            assert key in result.scalars

    def test_summary_returns_string(self, data_with_missing):
        mvh = MissingValueHandler()
        mvh.fit(data_with_missing).compute()
        text = mvh.summary()
        assert isinstance(text, str) and len(text) > 0

    def test_get_output_data(self, data_with_missing):
        mvh = MissingValueHandler()
        mvh.fit(data_with_missing).compute()
        output = mvh.get_output_data()
        assert isinstance(output, pd.DataFrame)
        assert output.isnull().sum().sum() == 0

    def test_tex_description_returns_string(self, sample_data):
        mvh = MissingValueHandler()
        mvh.fit(sample_data).compute()
        tex = mvh.tex_description()
        assert isinstance(tex, str) and len(tex) > 0


# ============================================================
#  OutlierHandler 测试
# ============================================================

class TestOutlierHandler:

    def test_zscore_clip(self, data_with_outliers):
        olh = OutlierHandler()
        result = olh.fit(
            data_with_outliers, method="zscore", action="clip", threshold=2.5
        ).compute()
        output = result.metadata["output_data"]
        # 异常值应被截断（不应存在 9999）
        assert output["指标A"].max() < 9999.0

    def test_iqr_clip(self, data_with_outliers):
        olh = OutlierHandler()
        result = olh.fit(
            data_with_outliers, method="iqr", action="clip", iqr_k=1.5
        ).compute()
        output = result.metadata["output_data"]
        assert output["指标A"].max() < 9999.0

    def test_mad_remove(self, data_with_outliers):
        olh = OutlierHandler()
        result = olh.fit(
            data_with_outliers, method="mad", action="remove", threshold=2.5
        ).compute()
        output = result.metadata["output_data"]
        # 移除后出现 NaN
        assert output["指标A"].isnull().sum() >= 1

    def test_flag_action(self, data_with_outliers):
        olh = OutlierHandler()
        result = olh.fit(
            data_with_outliers, method="zscore", action="flag", threshold=2.0
        ).compute()
        output = result.metadata["output_data"]
        # flag 动作不修改原值，但添加了 _outlier 列
        flag_cols = [c for c in output.columns if "_outlier" in c]
        assert len(flag_cols) > 0

    def test_no_outliers(self, sample_data):
        olh = OutlierHandler()
        result = olh.fit(sample_data, method="zscore", threshold=10.0).compute()
        assert result.scalars["total_outliers"] == 0

    def test_invalid_method_raises(self, sample_data):
        with pytest.raises(ValueError, match="method"):
            OutlierHandler().fit(sample_data, method="bad_method")

    def test_invalid_action_raises(self, sample_data):
        with pytest.raises(ValueError, match="action"):
            OutlierHandler().fit(sample_data, action="bad_action")

    def test_scalars_total_outliers(self, data_with_outliers):
        olh = OutlierHandler()
        result = olh.fit(
            data_with_outliers, method="zscore", threshold=2.0
        ).compute()
        assert result.scalars["total_outliers"] >= 1


# ============================================================
#  DataCleaner 测试
# ============================================================

class TestDataCleaner:

    def test_full_pipeline(self, data_with_missing):
        # 先注入缺失值，再注入异常值
        df = data_with_missing.copy()
        df.loc["城市A", "指标B"] = 999.0

        cleaner = DataCleaner()
        result = cleaner.fit(
            df,
            missing_strategy="mean",
            outlier_method="zscore",
            outlier_action="clip",
        ).compute()
        output = cleaner.get_output_data()
        assert output.isnull().sum().sum() == 0
        assert output["指标B"].max() < 999.0

    def test_disable_missing(self, sample_data):
        cleaner = DataCleaner()
        result = cleaner.fit(
            sample_data,
            handle_missing=False,
            handle_outlier=False,
        ).compute()
        output = cleaner.get_output_data()
        # 无操作，数据应与原始相同
        pd.testing.assert_frame_equal(output, sample_data)

    def test_reports_in_tables(self, data_with_missing):
        cleaner = DataCleaner()
        result = cleaner.fit(data_with_missing).compute()
        assert "missing_report" in result.tables


# ============================================================
#  PositivityTransformer 测试
# ============================================================

class TestPositivityTransformer:

    def test_positive_unchanged(self, sample_data):
        pt = PositivityTransformer()
        result = pt.fit(
            sample_data,
            directions={c: "positive" for c in sample_data.columns}
        ).compute()
        pd.testing.assert_frame_equal(
            result.metadata["output_data"], sample_data
        )

    def test_negative_transform(self, data_with_negative):
        pt = PositivityTransformer()
        result = pt.fit(
            data_with_negative,
            directions={"正向A": "positive", "负向B": "negative", "适度C": "positive"},
        ).compute()
        output = result.metadata["output_data"]
        # 负向变换后，原始最大值对应的位置应变为0
        orig_max_idx = data_with_negative["负向B"].idxmax()
        assert output.loc[orig_max_idx, "负向B"] == pytest.approx(0.0)

    def test_moderate_transform(self, data_with_negative):
        pt = PositivityTransformer()
        result = pt.fit(
            data_with_negative,
            directions={"正向A": "positive", "负向B": "positive", "适度C": "moderate"},
            moderate_optimal={"适度C": 50.0},
        ).compute()
        output = result.metadata["output_data"]
        # 最优值处（≈50）应得到最高分（=1.0）
        closest_idx = (data_with_negative["适度C"] - 50.0).abs().idxmin()
        assert output.loc[closest_idx, "适度C"] == pytest.approx(1.0)

    def test_moderate_without_optimal_raises(self, data_with_negative):
        with pytest.raises(ValueError, match="适度型"):
            PositivityTransformer().fit(
                data_with_negative,
                directions={"正向A": "positive", "负向B": "positive", "适度C": "moderate"},
                # 故意不提供 moderate_optimal
            ).compute()

    def test_output_range_negative(self, data_with_negative):
        """负向变换后所有值应 ≥ 0"""
        pt = PositivityTransformer()
        result = pt.fit(
            data_with_negative,
            directions={"正向A": "positive", "负向B": "negative", "适度C": "positive"},
        ).compute()
        output = result.metadata["output_data"]
        assert (output["负向B"] >= -1e-10).all()

    def test_output_range_moderate(self, data_with_negative):
        """适度型变换后所有值应在 [0, 1]"""
        pt = PositivityTransformer()
        result = pt.fit(
            data_with_negative,
            directions={"正向A": "positive", "负向B": "positive", "适度C": "moderate"},
            moderate_optimal={"适度C": 50.0},
        ).compute()
        output = result.metadata["output_data"]
        assert (output["适度C"] >= -1e-10).all()
        assert (output["适度C"] <= 1.0 + 1e-10).all()


# ============================================================
#  归一化方法测试
# ============================================================

class TestNormalizers:

    def test_minmax_range(self, sample_data):
        result = MinMaxNormalizer().fit(sample_data).compute()
        output = result.metadata["output_data"]
        for col in output.columns:
            assert output[col].min() >= -1e-10
            assert output[col].max() <= 1.0 + 1e-10

    def test_minmax_custom_range(self, sample_data):
        result = MinMaxNormalizer().fit(
            sample_data, feature_range=(0.001, 1.0)
        ).compute()
        output = result.metadata["output_data"]
        for col in output.columns:
            assert output[col].min() >= 0.001 - 1e-10

    def test_minmax_constant_col():
        """常数列不应产生 NaN"""
        df = pd.DataFrame({"A": [5.0, 5.0, 5.0], "B": [1.0, 2.0, 3.0]})
        result = MinMaxNormalizer().fit(df).compute()
        assert result.metadata["output_data"].isnull().sum().sum() == 0

    def test_zscore_mean_std(self, sample_data):
        result = ZScoreNormalizer().fit(sample_data).compute()
        output = result.metadata["output_data"]
        for col in output.columns:
            assert abs(output[col].mean()) < 1e-10
            assert abs(output[col].std(ddof=1) - 1.0) < 1e-10

    def test_vector_l2_norm(self, sample_data):
        result = VectorNormalizer().fit(sample_data).compute()
        output = result.metadata["output_data"]
        for col in output.columns:
            col_norm = np.sqrt((output[col] ** 2).sum())
            assert abs(col_norm - 1.0) < 1e-10

    def test_sum_col_sums_to_one(self, sample_data):
        result = SumNormalizer().fit(sample_data).compute()
        output = result.metadata["output_data"]
        for col in output.columns:
            assert abs(output[col].sum() - 1.0) < 1e-10

    def test_sum_no_negative_output(self, sample_data):
        result = SumNormalizer().fit(sample_data).compute()
        output = result.metadata["output_data"]
        assert (output >= -1e-10).all().all()

    def test_zscore_constant_col_no_nan():
        """常数列 Z-score 应置为 0，不产生 NaN"""
        df = pd.DataFrame({"A": [3.0, 3.0, 3.0], "B": [1.0, 2.0, 3.0]})
        result = ZScoreNormalizer().fit(df).compute()
        assert result.metadata["output_data"].isnull().sum().sum() == 0
        assert (result.metadata["output_data"]["A"] == 0.0).all()


# ============================================================
#  DataNormalizer（完整流水线）测试
# ============================================================

class TestDataNormalizer:

    def test_full_pipeline_minmax(self, data_with_negative):
        dn = DataNormalizer()
        result = dn.fit(
            data_with_negative,
            directions={
                "正向A": "positive",
                "负向B": "negative",
                "适度C": "moderate",
            },
            moderate_optimal={"适度C": 50.0},
            method="minmax",
        ).compute()
        output = dn.get_output_data()

        assert isinstance(output, pd.DataFrame)
        assert output.shape == data_with_negative.shape
        # MinMax 输出应在 [0, 1]
        assert (output >= -1e-10).all().all()
        assert (output <= 1.0 + 1e-10).all().all()

    def test_skip_positivity(self, sample_data):
        dn = DataNormalizer()
        result = dn.fit(
            sample_data,
            method="vector",
            skip_positivity=True,
        ).compute()
        output = dn.get_output_data()
        # 向量归一化后每列 L2 范数 ≈ 1
        for col in output.columns:
            col_norm = np.sqrt((output[col] ** 2).sum())
            assert abs(col_norm - 1.0) < 1e-10

    def test_method_none_with_positivity(self, data_with_negative):
        """method='none' 只做正向化，不做归一化"""
        dn = DataNormalizer()
        result = dn.fit(
            data_with_negative,
            directions={"正向A": "positive", "负向B": "negative", "适度C": "positive"},
            method="none",
        ).compute()
        output = dn.get_output_data()
        # 负向变换：原始 max(负向B) 应变为 0
        orig_max_idx = data_with_negative["负向B"].idxmax()
        assert output.loc[orig_max_idx, "负向B"] == pytest.approx(0.0)

    def test_invalid_method_raises(self, sample_data):
        with pytest.raises(ValueError, match="method"):
            DataNormalizer().fit(sample_data, method="bad_method")

    def test_summary_returns_string(self, sample_data):
        dn = DataNormalizer()
        dn.fit(sample_data, method="minmax", skip_positivity=True).compute()
        text = dn.summary()
        assert isinstance(text, str) and len(text) > 0


# ============================================================
#  CorrelationAnalyzer 测试
# ============================================================

class TestCorrelationAnalyzer:

    def test_high_corr_detected(self, high_corr_data):
        analyzer = CorrelationAnalyzer(threshold=0.8)
        report = analyzer.analyze(high_corr_data)
        assert report.n_high_corr_pairs >= 2
        assert report.max_corr > 0.8

    def test_low_corr_no_pairs(self):
        np.random.seed(123)
        df = pd.DataFrame(np.random.randn(30, 4), columns=["A", "B", "C", "D"])
        # 随机独立数据通常低相关
        analyzer = CorrelationAnalyzer(threshold=0.99)
        report = analyzer.analyze(df)
        assert report.n_high_corr_pairs == 0

    def test_corr_matrix_shape(self, high_corr_data):
        analyzer = CorrelationAnalyzer()
        report = analyzer.analyze(high_corr_data)
        assert report.corr_matrix.shape == (4, 4)

    def test_corr_matrix_symmetric(self, high_corr_data):
        analyzer = CorrelationAnalyzer()
        report = analyzer.analyze(high_corr_data)
        cm = report.corr_matrix.values
        assert np.allclose(cm, cm.T, atol=1e-10)

    def test_get_high_corr_dataframe(self, high_corr_data):
        analyzer = CorrelationAnalyzer(threshold=0.8)
        analyzer.analyze(high_corr_data)
        df = analyzer.get_high_corr_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "相关系数" in df.columns

    def test_recommendation_not_empty(self, high_corr_data):
        analyzer = CorrelationAnalyzer(threshold=0.8)
        report = analyzer.analyze(high_corr_data)
        assert len(report.recommendation) > 0

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            CorrelationAnalyzer(threshold=1.5)


# ============================================================
#  PCAReducer 测试
# ============================================================

class TestPCAReducer:

    @pytest.fixture
    def pca_data(self) -> pd.DataFrame:
        """生成10个样本、5个指标的测试数据"""
        np.random.seed(7)
        X = np.random.randn(15, 5)
        # 故意引入相关性
        X[:, 1] = X[:, 0] * 0.9 + np.random.randn(15) * 0.1
        X[:, 3] = X[:, 2] * 0.8 + np.random.randn(15) * 0.2
        return pd.DataFrame(
            X,
            columns=["F1", "F2", "F3", "F4", "F5"],
            index=[f"S{i}" for i in range(15)],
        )

    def test_output_shape(self, pca_data):
        reducer = PCAReducer()
        result = reducer.fit(pca_data, variance_ratio=0.80).compute()
        output = reducer.get_output_data()
        # 主成分数应 ≤ 原始特征数
        assert output.shape[1] <= pca_data.shape[1]
        assert output.shape[0] == pca_data.shape[0]

    def test_variance_ratio_met(self, pca_data):
        target = 0.80
        reducer = PCAReducer()
        result = reducer.fit(pca_data, variance_ratio=target).compute()
        actual = result.scalars["variance_ratio_actual"]
        assert actual >= target