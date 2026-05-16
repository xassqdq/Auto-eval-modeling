"""
tests/test_evaluation.py
综合评价模块全量单元测试
"""

import numpy as np
import pandas as pd
import pytest

from src.algorithms.evaluation import (
    TOPSIS, VIKOR, GRA, FuzzyComprehensiveEvaluation,
    ELECTRE, RSR, DEA, DynamicTOPSIS, DynamicGRA, DynamicEvaluation,
    get_method, METHOD_REGISTRY,
)


# ──────────────────────────────────────────────────────────────────────────────
# 公共测试夹具
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """4 个城市、3 个指标的决策矩阵。"""
    return pd.DataFrame(
        {
            "GDP增长": [8.1, 7.5, 9.2, 6.8],
            "失业率":   [4.2, 3.8, 5.1, 4.0],
            "创新指数": [72.0, 68.0, 85.0, 61.0],
        },
        index=["城市A", "城市B", "城市C", "城市D"],
    )


@pytest.fixture
def sample_weights() -> list:
    return [0.3, 0.3, 0.4]


@pytest.fixture
def sample_types() -> list:
    return ["positive", "negative", "positive"]


@pytest.fixture
def dea_data() -> pd.DataFrame:
    """5 个医院的投入-产出数据。"""
    return pd.DataFrame(
        {
            "医生数":   [10, 15, 12, 8,  20],
            "床位数":   [50, 80, 60, 40, 100],
            "出院人数": [200, 280, 230, 160, 380],
            "手术量":   [80,  120, 95,  60,  150],
        },
        index=["医院1", "医院2", "医院3", "医院4", "医院5"],
    )


@pytest.fixture
def period_data(sample_data) -> dict:
    """3 个时期的面板数据（基于 sample_data 加随机扰动）。"""
    rng = np.random.default_rng(42)
    return {
        2021: sample_data,
        2022: sample_data + rng.normal(0, 0.3, sample_data.shape),
        2023: sample_data + rng.normal(0, 0.5, sample_data.shape),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 注册表测试
# ──────────────────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_get_method_valid(self):
        cls = get_method("topsis")
        assert cls is TOPSIS

    def test_get_method_case_insensitive(self):
        assert get_method("TOPSIS") is TOPSIS
        assert get_method("Gra") is GRA

    def test_get_method_invalid(self):
        with pytest.raises(KeyError, match="未知评价方法"):
            get_method("unknown_method")

    def test_registry_completeness(self):
        expected = {"topsis", "vikor", "gra", "fuzzy", "electre",
                    "rsr", "dea", "dynamic_topsis", "dynamic_gra"}
        assert expected.issubset(set(METHOD_REGISTRY.keys()))


# ──────────────────────────────────────────────────────────────────────────────
# TOPSIS 测试
# ──────────────────────────────────────────────────────────────────────────────

class TestTOPSIS:
    def test_basic_flow(self, sample_data, sample_weights, sample_types):
        model = TOPSIS(indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"Score (Ci)", "D+", "D-", "Rank"}
        assert len(result) == 4

    def test_score_range(self, sample_data, sample_weights, sample_types):
        model = TOPSIS(indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        scores = result["Score (Ci)"].values
        assert np.all(scores >= 0) and np.all(scores <= 1), \
            "TOPSIS 得分须在 [0,1] 范围内"

    def test_rank_is_permutation(self, sample_data, sample_weights, sample_types):
        model = TOPSIS(indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        ranks = sorted(result["Rank"].values)
        assert ranks == list(range(1, 5)), "排名须为 1,2,3,4 的排列"

    def test_equal_weights_default(self, sample_data, sample_types):
        model = TOPSIS(indicator_types=sample_types)
        model.fit(sample_data)
        result = model.compute()
        assert result is not None

    def test_weight_normalization(self, sample_data, sample_types):
        """未归一化权重应自动归一化。"""
        model1 = TOPSIS(indicator_types=sample_types)
        model1.fit(sample_data, weights=[3, 3, 4])
        r1 = model1.compute()

        model2 = TOPSIS(indicator_types=sample_types)
        model2.fit(sample_data, weights=[0.3, 0.3, 0.4])
        r2 = model2.compute()

        np.testing.assert_allclose(
            r1["Score (Ci)"].values, r2["Score (Ci)"].values, atol=1e-6
        )

    def test_invalid_weights_dim(self, sample_data, sample_types):
        model = TOPSIS(indicator_types=sample_types)
        with pytest.raises(ValueError, match="权重维度"):
            model.fit(sample_data, weights=[0.5, 0.5])

    def test_invalid_indicator_type(self, sample_data):
        model = TOPSIS(indicator_types=["positive", "neutral", "positive"])
        with pytest.raises(ValueError, match="无效指标类型"):
            model.fit(sample_data)

    def test_missing_values_handled(self, sample_types):
        data_with_nan = pd.DataFrame(
            {"A": [1.0, np.nan, 3.0, 4.0],
             "B": [4.0, 3.0, np.nan, 1.0],
             "C": [2.0, 3.0, 1.0, 4.0]},
            index=["a", "b", "c", "d"],
        )
        model = TOPSIS(indicator_types=sample_types)
        model.fit(data_with_nan)
        result = model.compute()
        assert result.isnull().sum().sum() == 0, "结果不应含缺失值"

    def test_summary_structure(self, sample_data, sample_weights, sample_types):
        model = TOPSIS(indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        model.compute()
        s = model.summary()

        required_keys = {"method", "criteria", "weights", "scores",
                         "ranking", "best", "worst", "ideal_positive",
                         "ideal_negative"}
        assert required_keys.issubset(s.keys())

    def test_plot_scores_returns_figure(self, sample_data, sample_weights, sample_types):
        import matplotlib.pyplot as plt
        model = TOPSIS(indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        model.compute()
        fig = model.plot_scores()
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_tex_description_not_empty(self, sample_data, sample_weights, sample_types):
        model = TOPSIS(indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        model.compute()
        tex = model.tex_description()
        assert "\\subsection" in tex
        assert "C_i" in tex

    def test_compute_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="请先调用 fit"):
            TOPSIS().compute()


# ──────────────────────────────────────────────────────────────────────────────
# VIKOR 测试
# ──────────────────────────────────────────────────────────────────────────────

class TestVIKOR:
    def test_basic_flow(self, sample_data, sample_weights, sample_types):
        model = VIKOR(v=0.5, indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        assert isinstance(result, pd.DataFrame)
        assert "Q (妥协值)" in result.columns
        assert "Rank_Q" in result.columns

    def test_q_value_range(self, sample_data, sample_weights, sample_types):
        model = VIKOR(v=0.5, indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        q_vals = result["Q (妥协值)"].values
        assert np.all(q_vals >= -1e-6) and np.all(q_vals <= 1 + 1e-6)

    def test_v_parameter_bounds(self):
        with pytest.raises(ValueError, match="v 必须在"):
            VIKOR(v=1.5)
        with pytest.raises(ValueError, match="v 必须在"):
            VIKOR(v=-0.1)

    def test_compromise_check_in_metadata(self, sample_data, sample_weights, sample_types):
        model = VIKOR(v=0.5, indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        model.compute()

        assert "condition1_advantage" in model.metadata
        assert "condition2_stability" in model.metadata
        assert "compromise_valid" in model.metadata

    def test_summary(self, sample_data, sample_weights, sample_types):
        model = VIKOR(v=0.5, indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        model.compute()
        s = model.summary()
        assert s["method"] == "VIKOR"
        assert s["v_parameter"] == 0.5


# ──────────────────────────────────────────────────────────────────────────────
# GRA 测试
# ──────────────────────────────────────────────────────────────────────────────

class TestGRA:
    def test_basic_flow(self, sample_data, sample_weights, sample_types):
        model = GRA(rho=0.5, indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        assert isinstance(result, pd.DataFrame)
        assert "Grey Relational Grade" in result.columns
        assert "Rank" in result.columns

    def test_grade_range(self, sample_data, sample_weights, sample_types):
        model = GRA(rho=0.5, indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        grades = result["Grey Relational Grade"].values
        assert np.all(grades >= 0) and np.all(grades <= 1)

    def test_rho_bounds(self):
        with pytest.raises(ValueError, match="分辨系数 rho"):
            GRA(rho=0.0)
        with pytest.raises(ValueError, match="分辨系数 rho"):
            GRA(rho=1.0)

    def test_coefficient_matrix_shape(self, sample_data, sample_weights, sample_types):
        model = GRA(rho=0.5, indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        model.compute()
        coeff_df = model.get_coefficient_matrix()

        assert coeff_df.shape == (4, 3)
        assert np.all(coeff_df.values >= 0) and np.all(coeff_df.values <= 1)

    def test_custom_reference(self, sample_data, sample_types):
        custom_ref = np.array([9.0, 3.5, 80.0])
        model = GRA(rho=0.5, indicator_types=sample_types,
                    reference_mode="custom")
        model.fit(sample_data, reference=custom_ref)
        result = model.compute()
        assert isinstance(result, pd.DataFrame)

    def test_alias(self):
        from src.algorithms.evaluation.gra import GreyRelationalAnalysis
        assert GreyRelationalAnalysis is GRA

    def test_heatmap_figure(self, sample_data, sample_weights, sample_types):
        import matplotlib.pyplot as plt
        model = GRA(rho=0.5, indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        model.compute()
        fig = model.plot_coefficient_heatmap()
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ──────────────────────────────────────────────────────────────────────────────
# Fuzzy 测试
# ──────────────────────────────────────────────────────────────────────────────

class TestFuzzyComprehensiveEvaluation:
    @pytest.fixture
    def level_boundaries_4(self):
        """4 级等级，3 个指标。"""
        return [
            [(6.0, 7.5), (6.5, 7.5, 8.5), (7.5, 8.5, 9.5), (8.5, 10.0)],
            [(5.0, 3.5), (4.5, 3.5, 4.2), (3.8, 3.2, 4.0), (3.0, 4.0)],
            [(55.0, 65.0), (60.0, 65.0, 75.0), (70.0, 75.0, 85.0), (80.0, 90.0)],
        ]

    def test_auto_mode(self, sample_data, sample_weights):
        model = FuzzyComprehensiveEvaluation(
            level_names=["差", "一般", "良", "优"],
        )
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        assert isinstance(result, pd.DataFrame)
        assert "Score" in result.columns
        assert "Dominant Level" in result.columns

    def test_membership_tensor_mode(self, sample_data, sample_weights):
        n, m = sample_data.shape
        p = 4
        tensor = np.random.dirichlet(np.ones(p), size=(n, m))
        model = FuzzyComprehensiveEvaluation(
            level_names=["差", "一般", "良", "优"],
        )
        model.fit(sample_data, weights=sample_weights, membership_tensor=tensor)
        result = model.compute()

        assert len(result) == n

    def test_b_matrix_rows_sum_to_1(self, sample_data, sample_weights):
        model = FuzzyComprehensiveEvaluation(
            level_names=["差", "一般", "良", "优"],
        )
        model.fit(sample_data, weights=sample_weights)
        model.compute()

        row_sums = model._B.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_invalid_tensor_shape(self, sample_data):
        wrong_tensor = np.ones((4, 3, 3))  # 等级数不匹配
        model = FuzzyComprehensiveEvaluation(level_names=["差", "一般", "良", "优"])
        with pytest.raises(ValueError, match="membership_tensor 形状"):
            model.fit(sample_data, membership_tensor=wrong_tensor)

    def test_score_rank_consistent(self, sample_data, sample_weights):
        """得分排名须与 Rank 列一致。"""
        model = FuzzyComprehensiveEvaluation(
            level_names=["差", "一般", "良", "优"],
        )
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        scores = result["Score"]
        ranks = result["Rank"]
        # 最高分对应 Rank=1
        assert scores.idxmax() == ranks.idxmin()

    def test_get_membership_dataframe(self, sample_data, sample_weights):
        model = FuzzyComprehensiveEvaluation(
            level_names=["差", "一般", "良", "优"],
        )
        model.fit(sample_data, weights=sample_weights)
        model.compute()
        df = model.get_membership_dataframe("城市A")
        assert df.shape == (3, 4)

    def test_operator_min_max(self, sample_data, sample_weights):
        model = FuzzyComprehensiveEvaluation(
            level_names=["差", "一般", "良", "优"],
            operator="min_max",
        )
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()
        assert result is not None


# ──────────────────────────────────────────────────────────────────────────────
# ELECTRE 测试
# ──────────────────────────────────────────────────────────────────────────────

class TestELECTRE:
    def test_basic_flow(self, sample_data, sample_weights, sample_types):
        model = ELECTRE(
            concordance_threshold=0.6,
            discordance_threshold=0.4,
            indicator_types=sample_types,
        )
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        assert isinstance(result, pd.DataFrame)
        assert "Net Flow φ" in result.columns
        assert "Rank" in result.columns

    def test_matrix_shapes(self, sample_data, sample_weights, sample_types):
        model = ELECTRE(indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        model.compute()

        n = len(sample_data)
        assert model.get_concordance_matrix().shape == (n, n)
        assert model.get_discordance_matrix().shape == (n, n)
        assert model.get_outranking_matrix().shape == (n, n)

    def test_concordance_range(self, sample_data, sample_weights, sample_types):
        model = ELECTRE(indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        model.compute()

        c_mat = model._concordance_matrix
        assert np.all(c_mat >= -1e-6) and np.all(c_mat <= 1 + 1e-6)

    def test_discordance_range(self, sample_data, sample_weights, sample_types):
        model = ELECTRE(indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        model.compute()

        d_mat = model._discordance_matrix
        assert np.all(d_mat >= -1e-6) and np.all(d_mat <= 1 + 1e-6)

    def test_outranking_is_binary(self, sample_data, sample_weights, sample_types):
        model = ELECTRE(indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        model.compute()

        E = model._outranking_matrix
        assert set(np.unique(E)).issubset({0, 1})
        assert E.diagonal().sum() == 0


# ──────────────────────────────────────────────────────────────────────────────
# RSR 测试
# ──────────────────────────────────────────────────────────────────────────────

class TestRSR:
    def test_basic_flow(self, sample_data, sample_weights, sample_types):
        model = RSR(n_grades=4, indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        assert isinstance(result, pd.DataFrame)
        assert "RSR" in result.columns
        assert "Probit" in result.columns
        assert "Grade" in result.columns
        assert "Rank" in result.columns

    def test_rsr_range(self, sample_data, sample_weights, sample_types):
        model = RSR(n_grades=4, indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        rsr_vals = result["RSR"].values
        assert np.all(rsr_vals > 0) and np.all(rsr_vals <= 1 + 1e-6)

    def test_grade_labels_valid(self, sample_data, sample_weights, sample_types):
        model = RSR(n_grades=4, indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        valid_grades = {"差", "中下", "中上", "优"}
        assert set(result["Grade"].values).issubset(valid_grades)

    def test_regression_r2_reasonable(self, sample_data, sample_weights, sample_types):
        model = RSR(n_grades=4, indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        model.compute()

        r2 = model.metadata["regression_r2"]
        assert 0 <= r2 <= 1

    def test_invalid_n_grades(self):
        with pytest.raises(ValueError, match="n_grades 须在"):
            RSR(n_grades=6)

    def test_summary_regression_keys(self, sample_data, sample_weights, sample_types):
        model = RSR(n_grades=4, indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        model.compute()
        s = model.summary()
        assert "regression" in s
        assert set(s["regression"].keys()) == {"slope", "intercept", "r_squared"}


# ──────────────────────────────────────────────────────────────────────────────
# DEA 测试
# ──────────────────────────────────────────────────────────────────────────────

class TestDEA:
    def test_basic_ccr(self, dea_data):
        model = DEA(model="ccr")
        model.fit(dea_data, input_cols=["医生数", "床位数"],
                  output_cols=["出院人数", "手术量"])
        result = model.compute()

        assert "CCR_Efficiency" in result.columns
        assert "BCC_Efficiency" not in result.columns

    def test_basic_bcc(self, dea_data):
        model = DEA(model="bcc")
        model.fit(dea_data, input_cols=["医生数", "床位数"],
                  output_cols=["出院人数", "手术量"])
        result = model.compute()

        assert "BCC_Efficiency" in result.columns
        assert "CCR_Efficiency" not in result.columns

    def test_both_models(self, dea_data):
        model = DEA(model="both")
        model.fit(dea_data, input_cols=["医生数", "床位数"],
                  output_cols=["出院人数", "手术量"])
        result = model.compute()

        assert "CCR_Efficiency" in result.columns
        assert "BCC_Efficiency" in result.columns
        assert "Scale_Efficiency" in result.columns
        assert "Returns_to_Scale" in result.columns

    def test_efficiency_range(self, dea_data):
        model = DEA(model="both")
        model.fit(dea_data, input_cols=["医生数", "床位数"],
                  output_cols=["出院人数", "手术量"])
        result = model.compute()

        for col in ["CCR_Efficiency", "BCC_Efficiency", "Scale_Efficiency"]:
            vals = result[col].values
            assert np.all(vals >= -1e-6) and np.all(vals <= 1 + 1e-6), \
                f"{col} 超出 [0,1] 范围"

    def test_bcc_ge_ccr(self, dea_data):
        """BCC 效率 ≥ CCR 效率（VRS 前沿不劣于 CRS 前沿）。"""
        model = DEA(model="both")
        model.fit(dea_data, input_cols=["医生数", "床位数"],
                  output_cols=["出院人数", "手术量"])
        model.compute()

        ccr = model._ccr_efficiency
        bcc = model._bcc_efficiency
        assert np.all(bcc >= ccr - 1e-4), "BCC 效率须 ≥ CCR 效率"

    def test_rts_valid_labels(self, dea_data):
        model = DEA(model="both")
        model.fit(dea_data, input_cols=["医生数", "床位数"],
                  output_cols=["出院人数", "手术量"])
        model.compute()

        valid_rts = {"CRS", "IRS", "DRS"}
        assert set(model._rts).issubset(valid_rts)

    def test_missing_input_output_cols(self, dea_data):
        model = DEA(model="ccr")
        with pytest.raises(ValueError, match="必须指定"):
            model.fit(dea_data)

    def test_nonexistent_cols(self, dea_data):
        model = DEA(model="ccr")
        with pytest.raises(ValueError, match="数据中缺少"):
            model.fit(dea_data, input_cols=["不存在列"], output_cols=["出院人数"])

    def test_get_reference_set(self, dea_data):
        model = DEA(model="both")
        model.fit(dea_data, input_cols=["医生数", "床位数"],
                  output_cols=["出院人数", "手术量"])
        model.compute()

        ref = model.get_reference_set("医院3")
        assert isinstance(ref, dict)
        assert all(v >= 0 for v in ref.values())

    def test_slack_analysis_shape(self, dea_data):
        model = DEA(model="both")
        model.fit(dea_data, input_cols=["医生数", "床位数"],
                  output_cols=["出院人数", "手术量"])
        model.compute()

        slacks = model.get_slack_analysis()
        assert slacks.shape == (5, 4)  # 5 DMU, 2 inputs + 2 outputs

    def test_efficiency_plot(self, dea_data):
        import matplotlib.pyplot as plt
        model = DEA(model="both")
        model.fit(dea_data, input_cols=["医生数", "床位数"],
                  output_cols=["出院人数", "手术量"])
        model.compute()
        fig = model.plot_efficiency()
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ──────────────────────────────────────────────────────────────────────────────
# 动态评价测试
# ──────────────────────────────────────────────────────────────────────────────

class TestDynamicTOPSIS:
    def test_basic_flow(self, period_data, sample_weights, sample_types):
        model = DynamicTOPSIS(
            weights=sample_weights,
            indicator_types=sample_types,
        )
        model.fit(period_data)
        result = model.compute()

        assert isinstance(result, pd.DataFrame)
        assert "Dynamic_Score" in result.columns
        assert "Dynamic_Rank" in result.columns

    def test_dynamic_score_is_weighted_avg(self, period_data,
                                            sample_weights, sample_types):
        """动态得分须等于时间加权均值。"""
        model = DynamicTOPSIS(
            weights=sample_weights,
            indicator_types=sample_types,
            time_weight_method="equal",
        )
        model.fit(period_data)
        result = model.compute()

        score_cols = [c for c in result.columns if c.startswith("Score_")]
        manual_avg = result[score_cols].mean(axis=1)
        np.testing.assert_allclose(
            result["Dynamic_Score"].values,
            manual_avg.loc[result.index].values,
            atol=1e-4,
        )

    def test_period_result_access(self, period_data, sample_weights, sample_types):
        model = DynamicTOPSIS(
            weights=sample_weights,
            indicator_types=sample_types,
        )
        model.fit(period_data)
        model.compute()

        period_res = model.get_period_result(2022)
        assert isinstance(period_res, pd.DataFrame)
        assert "Score (Ci)" in period_res.columns

    def test_inconsistent_index_raises(self, sample_data, sample_weights, sample_types):
        bad_data = {
            2021: sample_data,
            2022: sample_data.rename(index={"城市A": "城市X"}),
        }
        model = DynamicTOPSIS(weights=sample_weights, indicator_types=sample_types)
        with pytest.raises(ValueError, match="行索引与第一期不一致"):
            model.fit(bad_data)

    def test_custom_time_weights(self, period_data, sample_weights, sample_types):
        model = DynamicTOPSIS(
            weights=sample_weights,
            indicator_types=sample_types,
        )
        model.fit(period_data, time_weights=[0.2, 0.3, 0.5])
        result = model.compute()
        assert result is not None

    def test_trajectory_plot(self, period_data, sample_weights, sample_types):
        import matplotlib.pyplot as plt
        model = DynamicTOPSIS(weights=sample_weights, indicator_types=sample_types)
        model.fit(period_data)
        model.compute()
        fig = model.plot_score_trajectory()
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_heatmap_plot(self, period_data, sample_weights, sample_types):
        import matplotlib.pyplot as plt
        model = DynamicTOPSIS(weights=sample_weights, indicator_types=sample_types)
        model.fit(period_data)
        model.compute()
        fig = model.plot_dynamic_heatmap()
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestDynamicGRA:
    def test_basic_flow(self, period_data, sample_weights, sample_types):
        model = DynamicGRA(
            rho=0.5,
            weights=sample_weights,
            indicator_types=sample_types,
        )
        model.fit(period_data)
        result = model.compute()

        assert isinstance(result, pd.DataFrame)
        assert "Dynamic_Grade" in result.columns
        assert "Dynamic_Rank" in result.columns

    def test_dynamic_grade_range(self, period_data, sample_weights, sample_types):
        model = DynamicGRA(
            rho=0.5,
            weights=sample_weights,
            indicator_types=sample_types,
        )
        model.fit(period_data)
        result = model.compute()

        grades = result["Dynamic_Grade"].values
        assert np.all(grades >= 0) and np.all(grades <= 1)


class TestDynamicEvaluation:
    def test_borda_fusion(self, period_data, sample_weights, sample_types):
        model = DynamicEvaluation(
            methods=["topsis", "gra"],
            fusion="borda",
            topsis_kwargs={"weights": sample_weights,
                           "indicator_types": sample_types},
            gra_kwargs={"weights": sample_weights,
                        "indicator_types": sample_types},
        )
        model.fit(period_data)
        result = model.compute()

        assert isinstance(result, pd.DataFrame)
        assert "Fused_Rank" in result.columns
        assert "Borda_Score" in result.columns

    def test_average_score_fusion(self, period_data, sample_weights, sample_types):
        model = DynamicEvaluation(
            methods=["topsis", "gra"],
            fusion="average_score",
            topsis_kwargs={"weights": sample_weights,
                           "indicator_types": sample_types},
            gra_kwargs={"weights": sample_weights,
                        "indicator_types": sample_types},
        )
        model.fit(period_data)
        result = model.compute()

        assert "Average_Score" in result.columns
        scores = result["Average_Score"].values
        assert np.all(scores >= 0) and np.all(scores <= 1 + 1e-6)

    def test_fused_rank_permutation(self, period_data, sample_weights, sample_types):
        model = DynamicEvaluation(
            methods=["topsis", "gra"],
            fusion="borda",
            topsis_kwargs={"weights": sample_weights,
                           "indicator_types": sample_types},
            gra_kwargs={"weights": sample_weights,
                        "indicator_types": sample_types},
        )
        model.fit(period_data)
        result = model.compute()

        ranks = sorted(result["Fused_Rank"].values)
        assert ranks == list(range(1, 5))

    def test_single_method_topsis(self, period_data, sample_weights, sample_types):
        model = DynamicEvaluation(
            methods=["topsis"],
            fusion="borda",
            topsis_kwargs={"weights": sample_weights,
                           "indicator_types": sample_types},
        )
        model.fit(period_data)
        result = model.compute()
        assert "Fused_Rank" in result.columns

    def test_rank_comparison_plot(self, period_data, sample_weights, sample_types):
        import matplotlib.pyplot as plt
        model = DynamicEvaluation(
            methods=["topsis", "gra"],
            fusion="borda",
            topsis_kwargs={"weights": sample_weights,
                           "indicator_types": sample_types},
            gra_kwargs={"weights": sample_weights,
                        "indicator_types": sample_types},
        )
        model.fit(period_data)
        model.compute()
        fig = model.plot_rank_comparison()
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_summary_structure(self, period_data, sample_weights, sample_types):
        model = DynamicEvaluation(
            methods=["topsis", "gra"],
            fusion="borda",
            topsis_kwargs={"weights": sample_weights,
                           "indicator_types": sample_types},
            gra_kwargs={"weights": sample_weights,
                        "indicator_types": sample_types},
        )
        model.fit(period_data)
        model.compute()
        s = model.summary()

        assert s["method"] == "DynamicEvaluation"
        assert "topsis_summary" in s
        assert "gra_summary" in s
        assert "fused_ranking" in s

    def test_tex_description_not_empty(self, period_data, sample_weights, sample_types):
        model = DynamicEvaluation(
            methods=["topsis", "gra"],
            fusion="borda",
            topsis_kwargs={"weights": sample_weights,
                           "indicator_types": sample_types},
            gra_kwargs={"weights": sample_weights,
                        "indicator_types": sample_types},
        )
        model.fit(period_data)
        model.compute()
        tex = model.tex_description()
        assert "Borda" in tex
        assert "\\subsection" in tex


# ──────────────────────────────────────────────────────────────────────────────
# 时间权重工具函数测试
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeTimeWeights:
    from src.algorithms.evaluation.dynamic import compute_time_weights

    def test_equal_weights_sum_to_1(self):
        from src.algorithms.evaluation.dynamic import compute_time_weights
        w = compute_time_weights(5, method="equal")
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)
        np.testing.assert_allclose(w, np.full(5, 0.2), atol=1e-10)

    def test_linear_weights_sum_to_1(self):
        from src.algorithms.evaluation.dynamic import compute_time_weights
        w = compute_time_weights(4, method="linear")
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)
        # 越后期权重越大
        assert w[-1] > w[0]

    def test_exponential_monotone_increasing(self):
        from src.algorithms.evaluation.dynamic import compute_time_weights
        w = compute_time_weights(5, method="exponential", decay=0.9)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)
        for i in range(1, len(w)):
            assert w[i] >= w[i - 1] - 1e-10, "指数权重须单调不减"

    def test_time_degree_weights(self):
        from src.algorithms.evaluation.dynamic import compute_time_weights
        w = compute_time_weights(4, method="time_degree")
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)

    def test_custom_weights(self):
        from src.algorithms.evaluation.dynamic import compute_time_weights
        w = compute_time_weights(3, method="custom",
                                  custom_weights=[1.0, 2.0, 3.0])
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)
        np.testing.assert_allclose(w, [1/6, 2/6, 3/6], atol=1e-6)

    def test_invalid_method(self):
        from src.algorithms.evaluation.dynamic import compute_time_weights
        with pytest.raises(ValueError, match="未知时间权重方法"):
            compute_time_weights(3, method="invalid")

    def test_invalid_decay(self):
        from src.algorithms.evaluation.dynamic import compute_time_weights
        with pytest.raises(ValueError, match="decay 须在"):
            compute_time_weights(3, method="exponential", decay=1.5)


# ──────────────────────────────────────────────────────────────────────────────
# 跨方法一致性测试（健壮性）
# ──────────────────────────────────────────────────────────────────────────────

class TestCrossMethodConsistency:
    """验证在同一数据集上，不同方法的排名结果具有合理一致性。"""

    def test_ranking_correlation(self, sample_data, sample_weights, sample_types):
        """TOPSIS 与 GRA 的排名 Spearman 相关系数须 > 0.5。"""
        from scipy.stats import spearmanr

        t_model = TOPSIS(indicator_types=sample_types)
        t_model.fit(sample_data, weights=sample_weights)
        t_result = t_model.compute()

        g_model = GRA(rho=0.5, indicator_types=sample_types)
        g_model.fit(sample_data, weights=sample_weights)
        g_result = g_model.compute()

        alts = sample_data.index.tolist()
        t_ranks = [t_result.loc[a, "Rank"] for a in alts]
        g_ranks = [g_result.loc[a, "Rank"] for a in alts]

        rho, _ = spearmanr(t_ranks, g_ranks)
        assert rho > 0.5, \
            f"TOPSIS 与 GRA 排名相关性过低: ρ={rho:.3f}"

    def test_all_methods_return_same_alternatives(
            self, sample_data, sample_weights, sample_types):
        """所有方法的结果行索引须与输入数据行索引一致。"""
        methods_results = []

        for cls, kwargs in [
            (TOPSIS, {"indicator_types": sample_types}),
            (VIKOR, {"indicator_types": sample_types}),
            (GRA,   {"indicator_types": sample_types}),
            (ELECTRE, {"indicator_types": sample_types}),
            (RSR,   {"indicator_types": sample_types}),
        ]:
            m = cls(**kwargs)
            m.fit(sample_data, weights=sample_weights)
            result = m.compute()
            methods_results.append(set(result.index.tolist()))

        expected = set(sample_data.index.tolist())
        for r in methods_results:
            assert r == expected, f"评价对象集合不一致: {r} vs {expected}"

    def test_topsis_best_is_max_score(self, sample_data, sample_weights, sample_types):
        """TOPSIS 排名第一的对象须对应最高 Ci 得分。"""
        model = TOPSIS(indicator_types=sample_types)
        model.fit(sample_data, weights=sample_weights)
        result = model.compute()

        best_by_rank = result[result["Rank"] == 1].index[0]
        best_by_score = result["Score (Ci)"].idxmax()
        assert best_by_rank == best_by_score

    def test_determinism(self, sample_data, sample_weights, sample_types):
        """相同输入须产生完全相同的输出（确定性）。"""
        results = []
        for _ in range(3):
            model = TOPSIS(indicator_types=sample_types)
            model.fit(sample_data, weights=sample_weights)
            results.append(model.compute()["Score (Ci)"].values.copy())

        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[1], results[2])