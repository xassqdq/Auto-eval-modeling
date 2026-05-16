"""
generators/ 模块单元测试
================================================
涵盖 LatexBuilder / CodeBuilder / PlotBuilder 核心功能测试。
运行: pytest tests/test_generators.py -v
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.generators import CodeBuilder, LatexBuilder, PlotBuilder


# ══════════════════════════════════════════════════════════════════════════════
#  公共 Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def tmp_dir():
    """临时输出目录（测试结束后自动清理）。"""
    d = tempfile.mkdtemp(prefix="autoeval_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_indicators():
    return ["指标A", "指标B", "指标C", "指标D"]


@pytest.fixture
def sample_objects():
    return ["对象1", "对象2", "对象3", "对象4", "对象5"]


@pytest.fixture
def sample_weights():
    return np.array([0.30, 0.25, 0.28, 0.17])


@pytest.fixture
def sample_scores():
    return np.array([0.72, 0.85, 0.61, 0.78, 0.65])


@pytest.fixture
def entropy_result(sample_indicators, sample_weights):
    e = np.array([0.92, 0.88, 0.91, 0.96])
    return {
        "indicators": sample_indicators,
        "weights":    sample_weights,
        "entropy":    e,
        "diversity":  1 - e,
    }


@pytest.fixture
def topsis_result(sample_objects, sample_scores):
    ranking = np.argsort(-sample_scores) + 1
    return {
        "objects":    sample_objects,
        "scores":     sample_scores,
        "ranking":    ranking,
        "d_positive": np.array([0.15, 0.10, 0.25, 0.13, 0.22]),
        "d_negative": np.array([0.38, 0.45, 0.30, 0.40, 0.33]),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  LatexBuilder 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestLatexBuilder:

    def test_init_cn(self, tmp_dir):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir / "latex_cn")
        assert lb.language == "cn"
        assert lb.section_count == 0

    def test_init_en(self, tmp_dir):
        lb = LatexBuilder(language="en", output_dir=tmp_dir / "latex_en")
        assert lb.language == "en"

    def test_add_entropy_section(self, tmp_dir, entropy_result):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir)
        lb.add_entropy_section(entropy_result, label="test")
        assert lb.section_count == 1
        assert "熵权法" in lb.used_algorithms

    def test_add_topsis_section(self, tmp_dir, topsis_result):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir)
        lb.add_topsis_section(topsis_result, label="test")
        assert lb.section_count == 1
        assert "TOPSIS" in lb.used_algorithms

    def test_add_ahp_section(self, tmp_dir, sample_indicators):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir)
        n = len(sample_indicators)
        jm = np.eye(n)
        result = {
            "indicators":     sample_indicators,
            "weights":        np.ones(n) / n,
            "judgment_matrix": jm,
            "lambda_max":     float(n),
            "ci":             0.0,
            "cr":             0.0,
        }
        lb.add_ahp_section(result, label="test_ahp")
        assert lb.section_count == 1
        assert "AHP" in lb.used_algorithms

    def test_add_gra_section(self, tmp_dir, sample_objects, sample_scores, sample_indicators):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir)
        result = {
            "objects":    sample_objects,
            "scores":     sample_scores,
            "ranking":    np.argsort(-sample_scores) + 1,
            "rho":        0.5,
            "indicators": sample_indicators,
            "coeff_matrix": np.random.rand(5, 4),
        }
        lb.add_gra_section(result, label="test_gra")
        assert lb.section_count == 1

    def test_build_document_contains_title(self, tmp_dir, entropy_result, topsis_result):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir)
        lb.add_entropy_section(entropy_result)
        lb.add_topsis_section(topsis_result)
        doc = lb.build_document(title="测试报告")
        assert "测试报告" in doc
        assert r"\begin{document}" in doc
        assert r"\end{document}" in doc
        assert r"\begin{table}" in doc

    def test_build_document_en(self, tmp_dir, entropy_result):
        lb = LatexBuilder(language="en", output_dir=tmp_dir)
        lb.add_entropy_section(entropy_result)
        doc = lb.build_document(title="Test Report")
        assert "Test Report" in doc
        assert r"\documentclass" in doc

    def test_save_creates_file(self, tmp_dir, entropy_result):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir / "save_test")
        lb.add_entropy_section(entropy_result)
        path = lb.save("test_report.tex", title="测试")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_save_sections_only(self, tmp_dir, topsis_result):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir / "sections_test")
        lb.add_topsis_section(topsis_result)
        path = lb.save_sections_only("test_sections.tex")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert r"\documentclass" not in content    # 无文档骨架

    def test_add_figure(self, tmp_dir):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir)
        lb.add_figure("test.png", "测试图片", "test_fig")
        assert lb.section_count == 1
        assert r"\includegraphics" in lb._sections[-1]

    def test_clear(self, tmp_dir, entropy_result):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir)
        lb.add_entropy_section(entropy_result)
        assert lb.section_count == 1
        lb.clear()
        assert lb.section_count == 0
        assert len(lb.used_algorithms) == 0

    def test_render_no_placeholder_leak(self, tmp_dir, entropy_result):
        """确保所有 <<VAR>> 占位符均已被替换。"""
        lb = LatexBuilder(language="cn", output_dir=tmp_dir)
        lb.add_entropy_section(entropy_result)
        doc = lb.build_document()
        assert "<<" not in doc
        assert ">>" not in doc

    def test_pca_section(self, tmp_dir):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir)
        lb.add_pca_section({
            "n_components": 2, "threshold": 85.0,
            "eigenvalues":   [2.5, 1.3, 0.8, 0.4],
            "explained_var": [50.0, 26.0, 16.0, 8.0],
            "cumulative_var":[50.0, 76.0, 92.0, 100.0],
        }, label="pca_test")
        assert lb.section_count == 1

    def test_critic_section(self, tmp_dir, sample_indicators, sample_weights):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir)
        n = len(sample_indicators)
        lb.add_critic_section({
            "indicators":   sample_indicators,
            "weights":      sample_weights,
            "std_devs":     np.random.rand(n),
            "conflicts":    np.random.rand(n) * 5,
            "info_amounts": np.random.rand(n),
        })
        assert lb.section_count == 1
        assert "CRITIC" in lb.used_algorithms

    def test_vikor_section(self, tmp_dir, sample_objects, sample_scores):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir)
        n = len(sample_objects)
        S = np.random.rand(n)
        R = np.random.rand(n) * 0.3
        Q = 0.5 * S / S.max() + 0.5 * R / R.max()
        lb.add_vikor_section({
            "objects": sample_objects,
            "S": S, "R": R, "Q": Q,
            "ranking": Q.argsort() + 1,
            "v": 0.5,
        }, label="vikor_test")
        assert lb.section_count == 1

    def test_final_result_section(self, tmp_dir, sample_objects, sample_scores):
        lb = LatexBuilder(language="cn", output_dir=tmp_dir)
        lb.add_topsis_section({
            "objects": sample_objects, "scores": sample_scores,
            "ranking": np.argsort(-sample_scores) + 1,
            "d_positive": np.zeros(5), "d_negative": sample_scores,
        })
        lb.add_final_result_section({
            "objects": sample_objects,
            "scores":  sample_scores,
            "ranking": np.argsort(-sample_scores) + 1,
        })
        assert lb.section_count == 2


# ══════════════════════════════════════════════════════════════════════════════
#  CodeBuilder 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestCodeBuilder:

    def test_init(self, tmp_dir):
        cb = CodeBuilder(language="cn", output_dir=tmp_dir / "code_cn")
        assert cb.language == "cn"
        assert cb.block_count == 0

    def test_set_data_config(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(output_dir=tmp_dir)
        cb.set_data_config(
            data_file="test.csv",
            object_col="名称",
            indicator_cols=sample_indicators,
            indicator_types=[1, 1, -1, 0],
        )
        assert cb._data_config["object_col"] == "名称"
        assert len(cb._data_config["indicator_cols"]) == 4

    def test_add_weight_entropy(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(output_dir=tmp_dir)
        cb.set_data_config("test.csv", "obj", sample_indicators)
        cb.add_weight_method("entropy")
        assert cb.block_count == 1
        assert cb._weight_var == "weights_ew"

    def test_add_weight_critic(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(output_dir=tmp_dir)
        cb.set_data_config("test.csv", "obj", sample_indicators)
        cb.add_weight_method("critic")
        assert cb._weight_var == "weights_crit"

    def test_add_weight_ahp(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(output_dir=tmp_dir)
        cb.set_data_config("test.csv", "obj", sample_indicators)
        jm = np.eye(len(sample_indicators))
        cb.add_weight_method("ahp", judgment_matrix=jm)
        assert cb._weight_var == "weights_ahp"

    def test_add_weight_unknown_raises(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(output_dir=tmp_dir)
        cb.set_data_config("test.csv", "obj", sample_indicators)
        with pytest.raises(ValueError, match="未知赋权方法"):
            cb.add_weight_method("unknown_method")

    def test_add_eval_topsis(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(output_dir=tmp_dir)
        cb.set_data_config("test.csv", "obj", sample_indicators)
        cb.add_weight_method("entropy")
        cb.add_evaluation_method("topsis")
        assert cb.block_count == 2
        assert "topsis" in cb._eval_methods

    def test_add_eval_gra(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(output_dir=tmp_dir)
        cb.set_data_config("test.csv", "obj", sample_indicators)
        cb.add_weight_method("critic")
        cb.add_evaluation_method("gra")
        assert "gra" in cb._eval_methods

    def test_add_eval_unknown_raises(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(output_dir=tmp_dir)
        cb.set_data_config("test.csv", "obj", sample_indicators)
        with pytest.raises(ValueError, match="未知评价方法"):
            cb.add_evaluation_method("unknown_eval")

    def test_build_script_no_config_raises(self, tmp_dir):
        cb = CodeBuilder(output_dir=tmp_dir)
        with pytest.raises(RuntimeError):
            cb.build_script()

    def test_build_script_contains_imports(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(output_dir=tmp_dir)
        cb.set_data_config("test.csv", "obj", sample_indicators)
        cb.add_weight_method("entropy")
        script = cb.build_script()
        assert "import numpy as np" in script
        assert "import pandas as pd" in script

    def test_build_script_contains_data_path(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(output_dir=tmp_dir)
        cb.set_data_config("my_data.csv", "city", sample_indicators)
        cb.add_weight_method("entropy")
        script = cb.build_script()
        assert "my_data.csv" in script
        assert "city" in script

    def test_save_creates_py_file(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(output_dir=tmp_dir / "scripts")
        cb.set_data_config("test.csv", "obj", sample_indicators)
        cb.add_weight_method("entropy")
        cb.add_evaluation_method("topsis")
        path = cb.save("test_eval.py")
        assert path.exists()
        assert path.suffix == ".py"
        content = path.read_text(encoding="utf-8")
        assert "def entropy_weight" in content
        assert "def topsis" in content

    def test_chaining(self, tmp_dir, sample_indicators):
        """测试链式调用。"""
        cb = (
            CodeBuilder(output_dir=tmp_dir)
            .set_data_config("d.csv", "obj", sample_indicators)
            .add_weight_method("entropy")
            .add_evaluation_method("topsis")
            .add_sensitivity_analysis()
        )
        assert cb.block_count == 3

    def test_clear(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(output_dir=tmp_dir)
        cb.set_data_config("test.csv", "obj", sample_indicators)
        cb.add_weight_method("entropy")
        cb.clear()
        assert cb.block_count == 0

    def test_en_language(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(language="en", output_dir=tmp_dir)
        cb.set_data_config("test.csv", "obj", sample_indicators)
        cb.add_weight_method("entropy")
        script = cb.build_script()
        assert "Section" in script

    def test_custom_block(self, tmp_dir, sample_indicators):
        cb = CodeBuilder(output_dir=tmp_dir)
        cb.set_data_config("test.csv", "obj", sample_indicators)
        cb.add_custom_block("# 自定义代码块\nprint('hello')", "自定义步骤")
        assert cb.block_count == 1
        script = cb.build_script()
        assert "print('hello')" in script


# ══════════════════════════════════════════════════════════════════════════════
#  PlotBuilder 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotBuilder:

    def test_init(self, tmp_dir):
        pb = PlotBuilder(language="cn", output_dir=tmp_dir / "figs")
        assert pb.language == "cn"
        assert pb.figure_count == 0

    def test_plot_weights(self, tmp_dir, sample_weights, sample_indicators):
        pb = PlotBuilder(output_dir=tmp_dir / "figs_w")
        fig, ax = pb.plot_weights(sample_weights, sample_indicators,
                                   method_name="熵权法")
        assert fig is not None
        assert pb.figure_count == 1

    def test_plot_weight_comparison(self, tmp_dir, sample_indicators, sample_weights):
        pb = PlotBuilder(output_dir=tmp_dir / "figs_wc")
        w2 = np.array([0.28, 0.24, 0.30, 0.18])
        fig, ax = pb.plot_weight_comparison(
            {"方法A": sample_weights, "方法B": w2}, sample_indicators
        )
        assert fig is not None

    def test_plot_ranking(self, tmp_dir, sample_scores, sample_objects):
        pb = PlotBuilder(output_dir=tmp_dir / "figs_r")
        fig, ax = pb.plot_ranking(sample_scores, sample_objects)
        assert fig is not None
        assert pb.figure_count == 1

    def test_plot_radar(self, tmp_dir, sample_objects, sample_indicators):
        pb = PlotBuilder(output_dir=tmp_dir / "figs_radar")
        data = np.random.rand(5, 4)
        fig, ax = pb.plot_radar(data, sample_indicators, sample_objects)
        assert fig is not None

    def test_plot_correlation_heatmap(self, tmp_dir, sample_indicators):
        pb = PlotBuilder(output_dir=tmp_dir / "figs_corr")
        data = np.random.rand(20, 4)
        fig, ax = pb.plot_correlation_heatmap(data, sample_indicators)
        assert fig is not None

    def test_plot_sensitivity(self, tmp_dir, sample_objects, sample_indicators):
        pb = PlotBuilder(output_dir=tmp_dir / "figs_sens")
        ratios = np.linspace(-0.2, 0.2, 21)
        base_rank = np.array([1, 2, 3, 4, 5])
        rank_records = {}
        for ind in sample_indicators:
            rm = np.tile(base_rank, (21, 1))
            rank_records[ind] = rm
        sens = {"perturb_ratios": ratios, "rank_records": rank_records}
        fig, axes = pb.plot_sensitivity(sens, sample_objects)
        assert fig is not None

    def test_plot_pca_scree(self, tmp_dir):
        pb = PlotBuilder(output_dir=tmp_dir / "figs_pca")
        ev  = np.array([2.5, 1.3, 0.8, 0.4])
        var = np.array([50.0, 26.0, 16.0, 8.0])
        fig, axes = pb.plot_pca_scree(ev, var, n_selected=2)
        assert fig is not None

    def test_plot_entropy_analysis(self, tmp_dir, entropy_result, sample_indicators):
        pb = PlotBuilder(output_dir=tmp_dir / "figs_ew")
        fig, axes = pb.plot_entropy_analysis(entropy_result, sample_indicators)
        assert fig is not None
        assert axes.shape == (3,)

    def test_plot_ahp_heatmap(self, tmp_dir):
        pb = PlotBuilder(output_dir=tmp_dir / "figs_ahp")
        jm = np.array([[1, 2, 3], [0.5, 1, 2], [1/3, 0.5, 1]])
        fig, ax = pb.plot_ahp_heatmap(jm, ["A", "B", "C"])
        assert fig is not None

    def test_plot_dynamic_trend(self, tmp_dir, sample_objects):
        pb = PlotBuilder(output_dir=tmp_dir / "figs_dyn")
        scores_time = np.random.rand(5, 5)
        years = ["2019", "2020", "2021", "2022", "2023"]
        fig, ax = pb.plot_dynamic_trend(scores_time, sample_objects, years)
        assert fig is not None

    def test_save_all(self, tmp_dir, sample_weights, sample_indicators,
                      sample_scores, sample_objects):
        out = tmp_dir / "figs_save_all"
        pb  = PlotBuilder(output_dir=out, fig_format="png")
        pb.plot_weights(sample_weights, sample_indicators)
        pb.plot_ranking(sample_scores, sample_objects)
        saved = pb.save_all()
        assert len(saved) == 2
        for p in saved:
            assert p.exists()
            assert p.suffix == ".png"
        # 保存后应清空注册表
        assert pb.figure_count == 0

    def test_save_figure_single(self, tmp_dir, sample_weights, sample_indicators):
        out = tmp_dir / "figs_single"
        pb  = PlotBuilder(output_dir=out, fig_format="png")
        pb.plot_weights(sample_weights, sample_indicators,
                        filename="my_weights")
        path = pb.save_figure("my_weights")
        assert path is not None
        assert path.exists()

    def test_save_figure_not_exist(self, tmp_dir):
        pb  = PlotBuilder(output_dir=tmp_dir)
        result = pb.save_figure("nonexistent")
        assert result is None

    def test_clear(self, tmp_dir, sample_weights, sample_indicators):
        pb = PlotBuilder(output_dir=tmp_dir / "figs_clear")
        pb.plot_weights(sample_weights, sample_indicators)
        assert pb.figure_count == 1
        pb.clear()
        assert pb.figure_count == 0

    def test_language_en(self, tmp_dir, sample_weights, sample_indicators):
        pb = PlotBuilder(language="en", output_dir=tmp_dir / "figs_en")
        fig, ax = pb.plot_weights(sample_weights, sample_indicators,
                                   method_name="EWM")
        title = ax.get_title()
        assert "Weight" in title

    def test_repr(self, tmp_dir):
        pb = PlotBuilder(output_dir=tmp_dir)
        r  = repr(pb)
        assert "PlotBuilder" in r
        assert "figures=0" in r