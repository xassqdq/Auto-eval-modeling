# tests/unit/test_sensitivity/test_weight_sensitivity.py

import pytest
import numpy as np
import pandas as pd
from src.algorithms.sensitivity.weight_sensitivity import WeightSensitivityAnalyzer
from src.algorithms.evaluation.topsis import TOPSIS


class TestWeightSensitivity:

    @pytest.fixture
    def base_setup(self, sample_3x4):
        weights = np.array([0.3, 0.2, 0.35, 0.15])
        return sample_3x4, weights, ["positive"]*4

    def test_oat_produces_results(self, base_setup):
        data, weights, dirs = base_setup
        analyzer = WeightSensitivityAnalyzer(
            data=data, base_weights=weights, directions=dirs,
            eval_method="topsis", delta=0.1
        )
        results = analyzer.run_oat()
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

    def test_ranking_stability(self, base_setup):
        """小幅权重变动不应导致排名剧烈变化"""
        data, weights, dirs = base_setup
        analyzer = WeightSensitivityAnalyzer(
            data=data, base_weights=weights, directions=dirs,
            eval_method="topsis", delta=0.05
        )
        results = analyzer.run_oat()
        # Kendall τ 应该很高
        assert all(results["kendall_tau"] > 0.5)

    def test_perturbed_weights_valid(self, base_setup):
        """扰动后的权重仍应和为1, 全部非负"""
        data, weights, dirs = base_setup
        analyzer = WeightSensitivityAnalyzer(
            data=data, base_weights=weights, directions=dirs,
            eval_method="topsis", delta=0.1
        )
        for pw in analyzer.generate_perturbed_weights():
            assert abs(pw.sum() - 1.0) < 1e-10
            assert all(pw >= 0)


# tests/unit/test_sensitivity/test_rank_consistency.py

import pytest
import numpy as np
from src.algorithms.sensitivity.rank_consistency import RankConsistencyChecker


class TestRankConsistency:

    def test_identical_rankings_tau_1(self):
        r1 = [1, 2, 3, 4, 5]
        r2 = [1, 2, 3, 4, 5]
        checker = RankConsistencyChecker()
        assert checker.kendall_tau(r1, r2) == pytest.approx(1.0)

    def test_reversed_rankings_tau_neg1(self):
        r1 = [1, 2, 3, 4, 5]
        r2 = [5, 4, 3, 2, 1]
        checker = RankConsistencyChecker()
        assert checker.kendall_tau(r1, r2) == pytest.approx(-1.0)

    def test_spearman_rho_range(self):
        r1 = [1, 3, 2, 5, 4]
        r2 = [2, 1, 3, 4, 5]
        checker = RankConsistencyChecker()
        rho = checker.spearman_rho(r1, r2)
        assert -1 <= rho <= 1

    def test_multiple_methods_consistency(self):
        """比较TOPSIS和VIKOR的排名一致性"""
        rankings_topsis = [1, 2, 3, 4, 5]
        rankings_vikor = [1, 3, 2, 4, 5]
        checker = RankConsistencyChecker()
        tau = checker.kendall_tau(rankings_topsis, rankings_vikor)
        assert tau > 0  # 应正相关