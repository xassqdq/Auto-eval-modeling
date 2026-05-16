"""
AutoEval-Modeling 输出生成器包
================================================
包含三个核心生成器：
  - LatexBuilder : LaTeX 学术报告生成器
  - CodeBuilder  : 可独立运行的 Python 脚本生成器
  - PlotBuilder  : 数据可视化图表生成器

快速使用示例
-----------
>>> from src.generators import LatexBuilder, CodeBuilder, PlotBuilder
>>>
>>> # ── LaTeX 报告 ──────────────────────────────────────
>>> lb = LatexBuilder(language="cn", output_dir="output/reports")
>>> lb.add_entropy_section(entropy_result)
>>> lb.add_topsis_section(topsis_result)
>>> lb.save("report.tex")
>>>
>>> # ── 独立 Python 脚本 ─────────────────────────────────
>>> cb = CodeBuilder(output_dir="output/scripts")
>>> cb.set_data_config(data_file="data.csv", object_col="城市",
...                   indicator_cols=["GDP","专利数","R&D"],
...                   indicator_types=[1, 1, 1])
>>> cb.add_weight_method("entropy")
>>> cb.add_evaluation_method("topsis")
>>> cb.save("solve_city.py")
>>>
>>> # ── 可视化图表 ───────────────────────────────────────
>>> pb = PlotBuilder(language="cn", output_dir="output/figures")
>>> pb.plot_weights(weights, labels=indicators)
>>> pb.plot_ranking(scores, labels=objects)
>>> pb.save_all()
"""

from .latex_builder import LatexBuilder
from .code_builder import CodeBuilder
from .plot_builder import PlotBuilder

__all__ = ["LatexBuilder", "CodeBuilder", "PlotBuilder"]
__version__ = "1.0.0"
__author__ = "AutoEval-Modeling"