# -*- coding: utf-8 -*-
"""
tests/test_weights.py
=====================
赋权方法模块单元测试
覆盖所有 8 种赋权算法的基本功能与边界情况。
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from src.algorithms.weights import (
    AHPMethod,
    BinomialCoefficientMethod,
    RingRatioScoringMethod,
    EntropyWeightMethod,
    CRITICMethod,
    StdDeviationMethod,
    PCAWeightMethod,
    MultiplicativeCombination,
    LinearCombination,
    GameTheoryCombination,
    MinDeviationCombination,
    get_weight_method,
    WEIGHT_METHOD_REGISTRY,
)


# ──────────────────────────────────────────────────────────────
#  公共 Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def ahp_matrix_3x3() -> np.ndarray:
    """3×3 一致性好的 AHP 判断矩阵（CR ≈ 0.0）。"""
    return np.array([
        [1.0,   2.0,   4.0],
        [0.5,   1.0,   2.0],
        [0.25,  0.5,   1.0],
    ])


@pytest.fixture
def ahp_matrix_inconsistent() -> np.ndarray:
    """3×3 一致性差的 AHP 判断矩阵。"""
    return np.array([
        [1, 3, 7],
        [1/3, 1, 9],
        [1/7, 1/9, 1],
    ])


@pytest.fixture
def decision_matrix() -> np.ndarray:
    """5×4 正向化决策矩阵（m=5对象, n=4指标）。"""
    np.random.seed(42)
    X = np.abs(np.random.randn(5, 4)) + 0.5
    return X


@pytest.fixture
def decision_df(decision_matrix) -> pd.DataFrame:
    return pd.DataFrame(
        decision_matrix,
        columns=["C1", "C2", "C3", "C4"],
    )


@pytest.fixture
def weight_set(decision_df) -> dict:
    """预先计算好的熵权和CRITIC权重（用于组合赋权测试）。"""
    ew = EntropyWeightMethod()
    ew.fit(decision_df).compute()
    cr = CRITICMethod()
    cr.fit(decision_df).compute()
    return {
        "entropy": ew.weights,
        "critic":  cr.weights,
    }


# ──────────────────────────────────────────────────────────────
#  1. AHPMethod 测试
# ──────────────────────────────────────────────────────────────

class TestAHPMethod:

    def test_basic_fit_compute(self, ahp_matrix_3x3):
        """基本拟合与计算流程。"""
        ahp = AHPMethod(indicator_names=["A", "B", "C"])
        ahp.fit(ahp_matrix_3x3).compute()
        r = ahp.summary()

        assert "weights" in r
        weights = r["weights"]
        assert weights.shape == (3,)
        assert abs(weights.sum() - 1.0) < 1e-8
        assert np.all(weights >= 0)
        assert r["is_consistent"] is True

    def test_weights_monotonic(self, ahp_matrix_3x3):
        """验证权重单调性（矩阵设计为 C1 > C2 > C3）。"""
        ahp = AHPMethod()
        ahp.fit(ahp_matrix_3x3).compute()
        w = ahp.summary()["weights"]
        assert w[0] > w[1] > w[2]

    def test_inconsistent_matrix_warning(self, ahp_matrix_inconsistent):
        """不一致矩阵应发出警告。"""
        ahp = AHPMethod(cr_threshold=0.1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ahp.fit(ahp_matrix_inconsistent).compute()
        assert any(
            "一致性检验" in str(w.message) or "CR" in str(w.message)
            for w in caught
        )

    @pytest.mark.parametrize("method", [
        "eigenvector", "geometric_mean", "arithmetic_mean"
    ])
    def test_weight_methods(self, ahp_matrix_3x3, method):
        """三种权重计算方法均能正常执行。"""
        ahp = AHPMethod(weight_method=method)
        ahp.fit(ahp_matrix_3x3).compute()
        w = ahp.summary()["weights"]
        assert abs(w.sum() - 1.0) < 1e-6
        assert np.all(w > 0)

    def test_from_comparisons_factory(self):
        """from_comparisons 工厂方法测试。"""
        ahp = AHPMethod.from_comparisons(
            3,
            [(0, 1, 2.0), (0, 2, 4.0), (1, 2, 2.0)],
            indicator_names=["X1", "X2", "X3"],
        )
        ahp.compute()
        w = ahp.summary()["weights"]
        assert w.shape == (3,)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_dataframe_input(self, ahp_matrix_3x3):
        """DataFrame 输入测试。"""
        df = pd.DataFrame(
            ahp_matrix_3x3,
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        ahp = AHPMethod()
        ahp.fit(df).compute()
        assert ahp.indicator_names == ["A", "B", "C"]

    def test_non_square_matrix_raises(self):
        """非方阵输入应抛出 ValueError。"""
        with pytest.raises(ValueError, match="方阵"):
            AHPMethod().fit(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_plot_returns_figure(self, ahp_matrix_3x3):
        """plot 方法返回 Figure。"""
        import matplotlib.pyplot as plt
        ahp = AHPMethod()
        ahp.fit(ahp_matrix_3x3).compute()
        fig = ahp.plot()
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_tex_description_contains_cr(self, ahp_matrix_3x3):
        """tex_description 包含关键 LaTeX 字段。"""
        ahp = AHPMethod()
        ahp.fit(ahp_matrix_3x3).compute()
        tex = ahp.tex_description()
        assert r"\subsubsection" in tex
        assert "CR" in tex or r"\lambda" in tex

    def test_static_check_consistency(self, ahp_matrix_3x3):
        """静态方法 check_consistency。"""
        cr, is_ok = AHPMethod.check_consistency(ahp_matrix_3x3)
        assert isinstance(cr, float)
        assert isinstance(is_ok, bool)
        assert is_ok  # 该矩阵 CR 应通过

    def test_not_fitted_raises(self):
        """未 fit 直接 compute 应抛出异常。"""
        ahp = AHPMethod()
        with pytest.raises(RuntimeError, match="fit"):
            ahp.compute()


# ──────────────────────────────────────────────────────────────
#  2. BinomialCoefficientMethod 测试
# ──────────────────────────────────────────────────────────────

class TestBinomialCoefficientMethod:

    @pytest.fixture
    def rankings_multi(self) -> np.ndarray:
        """3 专家 × 4 指标排名。"""
        return np.array([
            [1, 2, 3, 4],
            [2, 1, 3, 4],
            [1, 3, 2, 4],
        ])

    def test_single_expert(self):
        """单专家 1D 排名输入。"""
        bcm = BinomialCoefficientMethod(["A", "B", "C"])
        bcm.fit([1, 2, 3]).compute()
        w = bcm.summary()["weights"]
        assert w.shape == (3,)
        assert abs(w.sum() - 1.0) < 1e-8
        assert w[0] > w[1] > w[2]

    def test_multi_expert(self, rankings_multi):
        """多专家排名聚合。"""
        bcm = BinomialCoefficientMethod(aggregation="mean")
        bcm.fit(rankings_multi).compute()
        r = bcm.summary()
        assert r["weights"].shape == (4,)
        assert abs(r["weights"].sum() - 1.0) < 1e-8

    @pytest.mark.parametrize("agg", ["mean", "median", "geometric_mean"])
    def test_aggregation_methods(self, rankings_multi, agg):
        """三种聚合方法均正常。"""
        bcm = BinomialCoefficientMethod(aggregation=agg)
        bcm.fit(rankings_multi).compute()
        w = bcm.weights
        assert abs(w.sum() - 1.0) < 1e-8

    def test_plot(self, rankings_multi):
        """绘图测试。"""
        import matplotlib.pyplot as plt
        bcm = BinomialCoefficientMethod()
        bcm.fit(rankings_multi).compute()
        fig = bcm.plot()
        assert fig is not None
        plt.close("all")


# ──────────────────────────────────────────────────────────────
#  3. RingRatioScoringMethod 测试
# ──────────────────────────────────────────────────────────────

class TestRingRatioScoringMethod:

    def test_single_expert(self):
        """单专家比值：4 个指标 → 3 个比值。"""
        rrs = RingRatioScoringMethod(
            indicator_names=["A", "B", "C", "D"],
            order=[0, 1, 2, 3],
        )
        rrs.fit([1.5, 2.0, 1.2]).compute()
        w = rrs.summary()["weights"]
        assert w.shape == (4,)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_multi_expert_geometric_mean(self):
        """多专家几何平均聚合。"""
        ratios = [[1.5, 2.0, 1.2], [1.8, 1.6, 1.4]]
        rrs = RingRatioScoringMethod(aggregation="geometric_mean")
        rrs.fit(ratios).compute()
        w = rrs.weights
        assert abs(w.sum() - 1.0) < 1e-8

    def test_weights_monotonic_without_order(self):
        """无 order 时权重与得分单调对应（第一个最重要）。"""
        rrs = RingRatioScoringMethod()
        rrs.fit([3.0, 2.0, 1.5]).compute()
        sw = rrs._results["sorted_weights"]
        # sorted_weights 应严格递减
        assert sw[0] > sw[1] > sw[2] > sw[3]

    def test_negative_ratio_raises(self):
        """负比值应抛出 ValueError。"""
        rrs = RingRatioScoringMethod()
        with pytest.raises(ValueError, match="大于 0|> 0"):
            rrs.fit([-1.0, 2.0])

    def test_tex_description(self):
        """tex_description 包含 LaTeX 方程。"""
        rrs = RingRatioScoringMethod()
        rrs.fit([2.0, 1.5, 1.2]).compute()
        tex = rrs.tex_description()
        assert r"\begin{equation}" in tex
        assert r"v_k" in tex or r"v_{k}" in tex


# ──────────────────────────────────────────────────────────────
#  4. EntropyWeightMethod 测试
# ──────────────────────────────────────────────────────────────

class TestEntropyWeightMethod:

    def test_basic(self, decision_df):
        """基本功能测试。"""
        ew = EntropyWeightMethod()
        ew.fit(decision_df).compute()
        r  = ew.summary()
        assert r["weights"].shape == (4,)
        assert abs(r["weights"].sum() - 1.0) < 1e-8
        assert np.all(r["entropy"] >= 0)
        assert np.all(r["entropy"] <= 1.0 + 1e-6)

    def test_uniform_column_gets_zero_diff(self):
        """常数列差异系数为 0 → 权重为 0（均等列额外处理）。"""
        X = np.array([
            [1.0, 2.0],
            [1.0, 4.0],
            [1.0, 6.0],
            [1.0, 8.0],
        ])
        ew = EntropyWeightMethod()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ew.fit(X).compute()
        r = ew.summary()
        # 第一列全相同，差异系数 ≈ 0，权重应接近 0
        assert r["weights"][0] < 1e-4

    def test_dataframe_column_names(self, decision_df):
        """DataFrame 列名正确读取为指标名。"""
        ew = EntropyWeightMethod()
        ew.fit(decision_df).compute()
        assert ew.indicator_names == ["C1", "C2", "C3", "C4"]

    def test_plot(self, decision_df):
        """绘图正常。"""
        import matplotlib.pyplot as plt
        ew = EntropyWeightMethod()
        ew.fit(decision_df).compute()
        fig = ew.plot()
        assert fig is not None
        plt.close("all")

    def test_tex_table(self, decision_df):
        """LaTeX 输出包含 booktabs 环境。"""
        ew = EntropyWeightMethod()
        ew.fit(decision_df).compute()
        tex = ew.tex_description()
        assert r"\toprule" in tex
        assert r"\midrule" in tex
        assert r"\bottomrule" in tex


# ──────────────────────────────────────────────────────────────
#  5. CRITICMethod 测试
# ──────────────────────────────────────────────────────────────

class TestCRITICMethod:

    def test_basic(self, decision_df):
        """基本功能。"""
        cr = CRITICMethod()
        cr.fit(decision_df).compute()
        r  = cr.summary()
        assert r["weights"].shape == (4,)
        assert abs(r["weights"].sum() - 1.0) < 1e-8
        assert np.all(r["sigma"] >= 0)

    @pytest.mark.parametrize("corr_method", ["pearson", "spearman", "kendall"])
    def test_correlation_methods(self, decision_df, corr_method):
        """不同相关系数方法均正常运行。"""
        cr = CRITICMethod(correlation_method=corr_method)
        cr.fit(decision_df).compute()
        w = cr.weights
        assert abs(w.sum() - 1.0) < 1e-8

    def test_highly_correlated_indicators(self):
        """高度相关指标的冲突性应较低 → 权重应较低。"""
        np.random.seed(0)
        x = np.random.randn(10)
        X = np.column_stack([x, x * 0.99 + 0.01 * np.random.randn(10),
                              np.random.randn(10)])
        cr = CRITICMethod(indicator_names=["C1", "C2", "C3"])
        cr.fit(X).compute()
        r = cr.summary()
        # C1 和 C2 高度相关，冲突性低，权重应比 C3 小
        assert r["weights"][2] > r["weights"][0]

    def test_plot_returns_figure(self, decision_df):
        import matplotlib.pyplot as plt
        cr = CRITICMethod()
        cr.fit(decision_df).compute()
        fig = cr.plot()
        assert fig is not None
        plt.close("all")


# ──────────────────────────────────────────────────────────────
#  6. StdDeviationMethod 测试
# ──────────────────────────────────────────────────────────────

class TestStdDeviationMethod:

    @pytest.mark.parametrize("norm", ["minmax", "zscore", "raw"])
    def test_normalization_options(self, decision_df, norm):
        """三种归一化方案均正常。"""
        sd = StdDeviationMethod(normalization=norm)
        sd.fit(decision_df).compute()
        w = sd.weights
        assert abs(w.sum() - 1.0) < 1e-8

    def test_high_variance_gets_more_weight(self):
        """高方差列权重更大。"""
        X = np.array([
            [1.0, 100.0],
            [2.0, 200.0],
            [1.5, 300.0],
            [1.2, 400.0],
            [1.8, 500.0],
        ])
        sd = StdDeviationMethod(normalization="minmax")
        sd.fit(X).compute()
        w = sd.weights
        # 第二列方差远大于第一列，权重应更大
        assert w[1] > w[0]

    def test_all_zero_std_returns_equal_weights(self):
        """全常数矩阵应返回均等权重（附带警告）。"""
        X = np.ones((5, 3))
        sd = StdDeviationMethod()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sd.fit(X).compute()
        w = sd.weights
        assert np.allclose(w, 1 / 3, atol=1e-6)


# ──────────────────────────────────────────────────────────────
#  7. PCAWeightMethod 测试
# ──────────────────────────────────────────────────────────────

class TestPCAWeightMethod:

    def test_basic_with_threshold(self):
        """n_components=0.85 累计方差阈值。"""
        np.random.seed(7)
        X = np.random.randn(30, 6)
        pca_w = PCAWeightMethod(n_components=0.85)
        pca_w.fit(X).compute()
        r = pca_w.summary()
        assert r["weights"].shape == (6,)
        assert abs(r["weights"].sum() - 1.0) < 1e-8
        cum_var = r["explained_var_ratio"][:r["K"]].sum()
        assert cum_var >= 0.84   # 至少接近 85%

    def test_n_components_int(self):
        """n_components=2 精确指定。"""
        np.random.seed(0)
        X = np.random.randn(20, 5)
        pca_w = PCAWeightMethod(n_components=2)
        pca_w.fit(X).compute()
        assert pca_w._results["K"] == 2

    def test_correlation_vs_covariance(self):
        """相关矩阵 vs 协方差矩阵模式。"""
        np.random.seed(1)
        X = np.random.randn(25, 4) * np.array([1, 10, 100, 1000])
        pca_corr = PCAWeightMethod(use_correlation=True)
        pca_cov  = PCAWeightMethod(use_correlation=False)
        pca_corr.fit(X).compute()
        pca_cov.fit(X).compute()
        # 使用协方差时，大方差指标可能主导；使用相关矩阵时更均衡
        w_corr = pca_corr.weights
        w_cov  = pca_cov.weights
        # 两种方法权重不同
        assert not np.allclose(w_corr, w_cov, atol=1e-4)

    def test_scree_plot(self):
        """碎石图 / 可视化。"""
        import matplotlib.pyplot as plt
        np.random.seed(3)
        X = np.random.randn(15, 4)
        pca_w = PCAWeightMethod()
        pca_w.fit(X).compute()
        fig = pca_w.plot()
        assert fig is not None
        plt.close("all")


# ──────────────────────────────────────────────────────────────
#  8. 组合赋权法测试
# ──────────────────────────────────────────────────────────────

class TestCombinationMethods:

    @pytest.fixture
    def two_weights(self) -> List:
        """两组 4 维权重向量。"""
        return [
            np.array([0.40, 0.30, 0.20, 0.10]),
            np.array([0.25, 0.35, 0.25, 0.15]),
        ]

    @pytest.fixture
    def three_weights(self) -> List:
        """三组 4 维权重向量。"""
        return [
            np.array([0.40, 0.30, 0.20, 0.10]),
            np.array([0.25, 0.35, 0.25, 0.15]),
            np.array([0.30, 0.25, 0.30, 0.15]),
        ]

    # ── MultiplicativeCombination ──
    def test_multiplicative_basic(self, two_weights):
        mc = MultiplicativeCombination(indicator_names=["A","B","C","D"])
        mc.fit(two_weights).compute()
        r = mc.summary()
        assert abs(r["weights"].sum() - 1.0) < 1e-8
        assert np.all(r["weights"] > 0)

    def test_multiplicative_custom_alpha(self, two_weights):
        """自定义元权重。"""
        mc = MultiplicativeCombination(meta_weights=[0.3, 0.7])
        mc.fit(two_weights).compute()
        assert abs(mc.weights.sum() - 1.0) < 1e-8

    def test_multiplicative_three_methods(self, three_weights):
        mc = MultiplicativeCombination()
        mc.fit(three_weights).compute()
        assert mc.weights.shape == (4,)

    # ── LinearCombination ──
    def test_linear_equal_weights(self, two_weights):
        lc = LinearCombination()
        lc.fit(two_weights).compute()
        r  = lc.summary()
        # 等权线性组合 = 两向量均值
        expected = np.mean(two_weights, axis=0)
        expected /= expected.sum()
        assert np.allclose(r["weights"], expected, atol=1e-8)

    def test_linear_custom_alpha(self, two_weights):
        lc = LinearCombination(meta_weights=[0.6, 0.4])
        lc.fit(two_weights).compute()
        r  = lc.summary()
        expected = 0.6 * two_weights[0] + 0.4 * two_weights[1]
        expected /= expected.sum()
        assert np.allclose(r["weights"], expected, atol=1e-8)

    def test_linear_auto_optimize(self, three_weights):
        """自动优化元权重。"""
        lc = LinearCombination(auto_optimize=True)
        lc.fit(three_weights).compute()
        assert abs(lc.weights.sum() - 1.0) < 1e-6

    def test_linear_alpha_sum_equals_one(self, three_weights):
        lc = LinearCombination()
        lc.fit(three_weights).compute()
        assert abs(lc._results["alpha"].sum() - 1.0) < 1e-8

    # ── GameTheoryCombination ──
    @pytest.mark.parametrize("solver", ["kkt", "scipy"])
    def test_game_theory_both_solvers(self, two_weights, solver):
        gtc = GameTheoryCombination(solver=solver)
        gtc.fit(two_weights).compute()
        assert abs(gtc.weights.sum() - 1.0) < 1e-6

    def test_game_theory_three_methods(self, three_weights):
        gtc = GameTheoryCombination()
        gtc.fit(three_weights).compute()
        r = gtc.summary()
        assert r["alpha"].shape == (3,)
        assert abs(r["alpha"].sum() - 1.0) < 1e-6

    def test_game_theory_alpha_non_negative(self, three_weights):
        gtc = GameTheoryCombination()
        gtc.fit(three_weights).compute()
        assert np.all(gtc._results["alpha"] >= -1e-8)

    # ── MinDeviationCombination ──
    def test_min_deviation_no_matrix(self, two_weights):
        """无决策矩阵退化为权重空间最小化。"""
        mdc = MinDeviationCombination()
        mdc.fit(two_weights).compute()
        assert abs(mdc.weights.sum() - 1.0) < 1e-6

    def test_min_deviation_with_matrix(self, decision_matrix, two_weights):
        """有决策矩阵时启用评价结果层面优化。"""
        mdc = MinDeviationCombination(decision_matrix=decision_matrix)
        mdc.fit(two_weights).compute()
        r = mdc.summary()
        assert r["score_info"] is not None
        assert abs(r["weights"].sum() - 1.0) < 1e-6

    def test_min_deviation_plot_with_matrix(self, decision_matrix, two_weights):
        """有决策矩阵时绘制得分对比图。"""
        import matplotlib.pyplot as plt
        mdc = MinDeviationCombination(decision_matrix=decision_matrix)
        mdc.fit(two_weights).compute()
        fig = mdc.plot()
        assert fig is not None
        plt.close("all")

    # ── 通用接口测试 ──
    def test_tex_description_all(self, two_weights):
        """所有组合方法均有 tex_description。"""
        classes = [
            MultiplicativeCombination,
            LinearCombination,
            GameTheoryCombination,
            MinDeviationCombination,
        ]
        for cls in classes:
            obj = cls()
            obj.fit(two_weights).compute()
            tex = obj.tex_description()
            assert r"\subsubsection" in tex
            assert r"\begin{equation}" in tex

    def test_insufficient_weights_raises(self):
        """只提供一组权重应报错。"""
        with pytest.raises(ValueError, match="至少需要 2"):
            MultiplicativeCombination().fit([np.array([0.5, 0.5])])

    def test_dimension_mismatch_raises(self):
        """权重维度不一致应报错。"""
        with pytest.raises(ValueError, match="维度不一致"):
            LinearCombination().fit([
                np.array([0.5, 0.5]),
                np.array([0.3, 0.3, 0.4]),
            ])


# ──────────────────────────────────────────────────────────────
#  9. 注册表与工厂函数测试
# ──────────────────────────────────────────────────────────────

class TestRegistry:

    def test_all_keys_in_registry(self):
        """注册表包含所有预定义方法。"""
        expected = [
            "ahp", "binomial", "ring_ratio",
            "entropy", "critic", "std_deviation", "pca",
            "multiplicative", "linear", "game_theory", "min_deviation",
        ]
        for key in expected:
            assert key in WEIGHT_METHOD_REGISTRY, f"缺少键: '{key}'"

    def test_get_weight_method_valid(self):
        """工厂函数正确实例化。"""
        method = get_weight_method("entropy")
        assert isinstance(method, EntropyWeightMethod)

    def test_get_weight_method_with_kwargs(self):
        """工厂函数传参测试。"""
        method = get_weight_method("ahp", cr_threshold=0.15)
        assert isinstance(method, AHPMethod)
        assert method.cr_threshold == 0.15

    def test_get_weight_method_invalid_raises(self):
        """无效方法名应抛出 KeyError。"""
        with pytest.raises(KeyError, match="未知赋权方法"):
            get_weight_method("unknown_method_xyz")

    def test_get_weight_method_case_insensitive(self):
        """方法名大小写不敏感。"""
        m1 = get_weight_method("ENTROPY")
        m2 = get_weight_method("entropy")
        assert type(m1) is type(m2)


# ──────────────────────────────────────────────────────────────
#  10. 完整工作流集成测试
# ──────────────────────────────────────────────────────────────

class TestIntegration:

    def test_ahp_to_entropy_to_game_theory(self):
        """
        完整流程：AHP 主观权重 + 熵权法客观权重 → 博弈论组合
        """
        np.random.seed(99)
        # 决策矩阵：6 对象 × 4 指标
        X = np.abs(np.random.randn(6, 4)) + 1.0
        ind_names = ["经济", "社会", "环境", "技术"]

        # (1) AHP 主观权重
        matrix = np.array([
            [1.0, 2.0, 3.0, 5.0],
            [0.5, 1.0, 2.0, 3.0],
            [1/3, 0.5, 1.0, 2.0],
            [0.2, 1/3, 0.5, 1.0],
        ])
        ahp = AHPMethod(indicator_names=ind_names)
        ahp.fit(matrix).compute()
        w_ahp = ahp.weights

        # (2) 熵权法客观权重
        ew = EntropyWeightMethod(indicator_names=ind_names)
        ew.fit(X).compute()
        w_entropy = ew.weights

        # (3) 博弈论组合
        gtc = GameTheoryCombination(
            method_names=["AHP", "熵权法"],
            indicator_names=ind_names,
        )
        gtc.fit([w_ahp, w_entropy]).compute()
        r = gtc.summary()

        # 断言
        assert r["weights"].shape == (4,)
        assert abs(r["weights"].sum() - 1.0) < 1e-6
        assert all(r["weights"] > 0)

        # LaTeX 可生成
        tex = gtc.tex_description()
        assert len(tex) > 100

        # 绘图
        import matplotlib.pyplot as plt
        fig = gtc.plot()
        assert fig is not None
        plt.close("all")

    def test_critic_pca_linear_pipeline(self):
        """
        CRITIC + PCA → 线性组合（自动优化元权重）
        """
        np.random.seed(55)
        X = np.abs(np.random.randn(15, 5)) + 0.5

        cr   = CRITICMethod()
        cr.fit(X).compute()

        pca_w = PCAWeightMethod(n_components=0.9)
        pca_w.fit(X).compute()

        lc = LinearCombination(
            method_names=["CRITIC", "PCA"],
            auto_optimize=True,
        )
        lc.fit([cr.weights, pca_w.weights]).compute()

        assert abs(lc.weights.sum() - 1.0) < 1e-6
        assert lc._results["auto_optimize"] is True