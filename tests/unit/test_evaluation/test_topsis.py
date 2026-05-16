# tests/unit/test_evaluation/test_topsis.py

import pytest
import numpy as np
import pandas as pd
from src.algorithms.evaluation.topsis import TOPSIS


class TestTOPSIS:
    """TOPSIS 综合评价测试"""

    @pytest.fixture
    def topsis_setup(self, sample_3x4):
        """标准TOPSIS实例"""
        weights = np.array([0.3, 0.2, 0.35, 0.15])
        topsis = TOPSIS(
            data=sample_3x4,
            weights=weights,
            directions=["positive", "positive", "positive", "negative"]
        )
        topsis.compute()
        return topsis

    # ---------- 得分属性 ----------

    def test_scores_in_01(self, topsis_setup):
        """相对贴近度应在 [0, 1] 区间"""
        assert all(0 <= s <= 1 for s in topsis_setup.scores)

    def test_scores_length(self, topsis_setup, sample_3x4):
        assert len(topsis_setup.scores) == len(sample_3x4)

    # ---------- 排名属性 ----------

    def test_ranking_unique(self, topsis_setup, sample_3x4):
        """排名应无重复（此处假设无并列）"""
        ranking = topsis_setup.ranking
        assert len(set(ranking)) == len(sample_3x4)

    def test_ranking_covers_all(self, topsis_setup, sample_3x4):
        ranking = topsis_setup.ranking
        assert set(ranking) == set(range(1, len(sample_3x4) + 1))

    def test_highest_score_rank_1(self, topsis_setup):
        """最高分应排名第1"""
        scores = topsis_setup.scores
        ranking = topsis_setup.ranking
        max_idx = np.argmax(scores)
        assert ranking[max_idx] == 1

    # ---------- 距离计算 ----------

    def test_positive_ideal_distance_nonneg(self, topsis_setup):
        assert all(d >= 0 for d in topsis_setup.d_positive)

    def test_negative_ideal_distance_nonneg(self, topsis_setup):
        assert all(d >= 0 for d in topsis_setup.d_negative)

    def test_distance_sum_positive(self, topsis_setup):
        """d+ + d- > 0 对所有对象成立"""
        for dp, dn in zip(topsis_setup.d_positive, topsis_setup.d_negative):
            assert dp + dn > 0

    # ---------- 已知结果验证 ----------

    def test_known_result(self, sample_3x4, expected_topsis_scores):
        weights = np.array([0.3, 0.2, 0.35, 0.15])
        topsis = TOPSIS(sample_3x4, weights,
                        ["positive", "positive", "positive", "negative"])
        topsis.compute()
        for obj, expected_score in expected_topsis_scores.items():
            idx = list(sample_3x4.index).index(obj)
            assert abs(topsis.scores[idx] - expected_score) < 0.01

    # ---------- 特殊情况 ----------

    def test_identical_objects(self):
        """完全相同的对象应获得相同得分"""
        df = pd.DataFrame({
            "X1": [1, 1, 1], "X2": [2, 2, 2]
        }, index=["A", "B", "C"])
        weights = np.array([0.5, 0.5])
        topsis = TOPSIS(df, weights, ["positive", "positive"])
        topsis.compute()
        assert abs(topsis.scores[0] - topsis.scores[1]) < 1e-10

    def test_dominant_object_ranks_first(self):
        """在所有指标上都最优的对象应排名第1"""
        df = pd.DataFrame({
            "X1": [100, 50, 30],
            "X2": [200, 100, 80]
        }, index=["A", "B", "C"])
        weights = np.array([0.5, 0.5])
        topsis = TOPSIS(df, weights, ["positive", "positive"])
        topsis.compute()
        assert topsis.ranking[0] == 1

    def test_equal_weights(self, sample_3x4):
        """等权重应正常运行"""
        n = sample_3x4.shape[1]
        weights = np.ones(n) / n
        topsis = TOPSIS(sample_3x4, weights, ["positive"] * n)
        topsis.compute()
        assert abs(sum(topsis.scores) / len(topsis.scores)) > 0  # 非零

    # ---------- 输入校验 ----------

    def test_weight_dimension_mismatch(self, sample_3x4):
        with pytest.raises(ValueError, match="dimension|shape"):
            TOPSIS(sample_3x4, np.array([0.5, 0.5]), ["positive"] * 4)

    def test_direction_count_mismatch(self, sample_3x4):
        with pytest.raises(ValueError):
            TOPSIS(sample_3x4, np.array([0.25]*4), ["positive", "negative"])


# tests/unit/test_evaluation/test_vikor.py

import pytest
import numpy as np
import pandas as pd
from src.algorithms.evaluation.vikor import VIKOR


class TestVIKOR:

    def test_s_r_q_length(self, sample_3x4):
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        v = VIKOR(sample_3x4, weights, ["positive"]*4)
        v.compute()
        assert len(v.S) == len(sample_3x4)
        assert len(v.R) == len(sample_3x4)
        assert len(v.Q) == len(sample_3x4)

    def test_q_in_01(self, sample_3x4):
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        v = VIKOR(sample_3x4, weights, ["positive"]*4)
        v.compute()
        assert all(0 <= q <= 1 for q in v.Q)

    def test_compromise_condition(self, sample_3x4):
        """VIKOR需要验证折衷条件"""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        v = VIKOR(sample_3x4, weights, ["positive"]*4)
        v.compute()
        conditions = v.check_compromise_conditions()
        assert "acceptable_advantage" in conditions
        assert "acceptable_stability" in conditions

    def test_v_parameter_effect(self, sample_3x4):
        """v=0 最大群体效用, v=1 最小个体遗憾"""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        v0 = VIKOR(sample_3x4, weights, ["positive"]*4, v=0)
        v1 = VIKOR(sample_3x4, weights, ["positive"]*4, v=1)
        v0.compute()
        v1.compute()
        # Q值应不同
        assert not np.allclose(v0.Q, v1.Q)


# tests/unit/test_evaluation/test_gra.py

import pytest
import numpy as np
import pandas as pd
from src.algorithms.evaluation.gra import GrayRelationalAnalysis


class TestGRA:

    def test_relational_degree_range(self, sample_3x4):
        gra = GrayRelationalAnalysis(sample_3x4, directions=["positive"]*4)
        gra.compute()
        assert all(0 < d <= 1 for d in gra.relational_degrees)

    def test_rho_parameter_effect(self, sample_3x4):
        """分辨系数 ρ 的影响"""
        gra1 = GrayRelationalAnalysis(sample_3x4, ["positive"]*4, rho=0.1)
        gra2 = GrayRelationalAnalysis(sample_3x4, ["positive"]*4, rho=0.9)
        gra1.compute()
        gra2.compute()
        # 不同ρ应产生不同关联度
        assert not np.allclose(gra1.relational_degrees, gra2.relational_degrees)

    def test_ideal_reference_series(self, sample_3x4):
        """正向指标的理想参考序列应为各列最大值"""
        gra = GrayRelationalAnalysis(sample_3x4, ["positive"]*4)
        gra.compute()
        for i, col in enumerate(sample_3x4.columns):
            assert gra.reference_series[i] == sample_3x4[col].max()

    def test_relational_coefficient_matrix_shape(self, sample_3x4):
        gra = GrayRelationalAnalysis(sample_3x4, ["positive"]*4)
        gra.compute()
        assert gra.relational_coefficients.shape == sample_3x4.shape