"""
generators/ 模块完整联调示例
================================================
演示 LatexBuilder + CodeBuilder + PlotBuilder 的协同使用。
运行方式: python examples/run_generators_demo.py
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path（适配从任意目录运行）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.generators import CodeBuilder, LatexBuilder, PlotBuilder

# ─────────────────────────────────────────────────────────────────────────────
# 构造模拟数据（城市创新能力评价，5城市×6指标）
# ─────────────────────────────────────────────────────────────────────────────
CITIES     = ["北京", "上海", "广州", "深圳", "杭州"]
INDICATORS = ["R&D投入(亿元)", "专利申请数(万件)", "高新企业数(家)",
              "GDP增长率(%)", "人才引进数(万人)", "创新孵化器(个)"]
IND_TYPES  = [1, 1, 1, 1, 1, 1]    # 全部正向

rng = np.random.default_rng(42)
raw_data = rng.uniform(low=20, high=100, size=(5, 6))

df = pd.DataFrame(raw_data, columns=INDICATORS)
df.insert(0, "城市", CITIES)

# ─────────────────────────────────────────────────────────────────────────────
# 模拟算法输出结果
# ─────────────────────────────────────────────────────────────────────────────
# 熵权结果
weights_ew   = np.array([0.18, 0.22, 0.15, 0.19, 0.14, 0.12])
entropy_vals = np.array([0.92, 0.88, 0.94, 0.91, 0.95, 0.96])
diversity    = 1 - entropy_vals

entropy_result = {
    "indicators": INDICATORS,
    "weights":    weights_ew,
    "entropy":    entropy_vals,
    "diversity":  diversity,
}

# CRITIC 权重结果（模拟）
weights_crit = np.array([0.20, 0.21, 0.14, 0.17, 0.15, 0.13])
critic_result = {
    "indicators":   INDICATORS,
    "weights":      weights_crit,
    "std_devs":     rng.uniform(0.1, 0.4, 6),
    "conflicts":    rng.uniform(3.0, 5.5, 6),
    "info_amounts": rng.uniform(0.3, 1.2, 6),
}

# TOPSIS 结果（模拟）
scores_topsis = np.array([0.78, 0.82, 0.61, 0.74, 0.69])
d_pos = np.array([0.12, 0.09, 0.23, 0.15, 0.18])
d_neg = np.array([0.43, 0.48, 0.32, 0.40, 0.38])
ranking_topsis = np.argsort(-scores_topsis) + 1

topsis_result = {
    "objects":     CITIES,
    "scores":      scores_topsis,
    "ranking":     ranking_topsis,
    "d_positive":  d_pos,
    "d_negative":  d_neg,
}

# GRA 结果（模拟）
scores_gra  = np.array([0.75, 0.80, 0.58, 0.71, 0.65])
ranking_gra = np.argsort(-scores_gra) + 1
gra_result  = {
    "objects":  CITIES,
    "scores":   scores_gra,
    "ranking":  ranking_gra,
    "rho":      0.5,
    "indicators": INDICATORS,
}

# AHP 结果（模拟，3×3 子集）
n_sub = 4
jm = np.array([
    [1,   2,   3,   4  ],
    [1/2, 1,   2,   3  ],
    [1/3, 1/2, 1,   2  ],
    [1/4, 1/3, 1/2, 1  ],
])
ahp_result = {
    "indicators":     INDICATORS[:n_sub],
    "weights":        np.array([0.46, 0.28, 0.17, 0.09]),
    "judgment_matrix": jm,
    "lambda_max":     4.031,
    "ci":             0.010,
    "cr":             0.011,
}

# 灵敏度分析结果（模拟）
n_steps = 21
ratios  = np.linspace(-0.2, 0.2, n_steps)
rank_records = {}
base_rank = ranking_topsis.copy()
for ind in INDICATORS:
    rm = []
    for ratio in ratios:
        noise = rng.integers(-1, 2, size=len(CITIES))
        rank_pert = np.clip(base_rank + noise, 1, len(CITIES))
        rm.append(rank_pert)
    rank_records[ind] = np.vstack(rm)

sens_result = {
    "perturb_ratios": ratios,
    "rank_records":   rank_records,
    "is_robust":      True,
}

# PCA 结果（模拟）
pca_result = {
    "n_components":   3,
    "threshold":      85.5,
    "eigenvalues":    np.array([2.8, 1.4, 0.9, 0.5, 0.2, 0.1]),
    "explained_var":  np.array([46.7, 23.3, 15.0, 8.3, 3.3, 1.7]),
    "cumulative_var": np.array([46.7, 70.0, 85.0, 93.3, 96.7, 98.4]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. 生成 Python 脚本
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print(">>> Step 1: 生成 Python 求解脚本")
print("=" * 60)

cb = CodeBuilder(
    language="cn",
    output_dir="output/scripts",
    title="城市创新能力综合评价自动求解脚本",
)
cb.set_data_config(
    data_file="data/city_innovation.csv",
    object_col="城市",
    indicator_cols=INDICATORS,
    indicator_types=IND_TYPES,
)
cb.add_weight_method("entropy")
cb.add_weight_method("critic")
cb.add_evaluation_method("topsis", active_weights="weights_ew")
cb.add_evaluation_method("gra",    active_weights="weights_crit")
cb.add_sensitivity_analysis()
cb.add_save_results()

script_path = cb.save("city_innovation_eval.py")
print(f"Python 脚本已生成: {script_path}")
print(f"代码块数量: {cb.block_count}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 生成可视化图表
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(">>> Step 2: 生成可视化图表")
print("=" * 60)

pb = PlotBuilder(
    language="cn",
    output_dir="output/figures",
    fig_format="png",
    dpi=300,
)

# 2-1 熵权法分析三联图
pb.plot_entropy_analysis(entropy_result, INDICATORS)
print("  ✓ 熵权法分析图")

# 2-2 权重对比图
pb.plot_weight_comparison(
    {"熵权法": weights_ew, "CRITIC法": weights_crit},
    INDICATORS,
)
print("  ✓ 权重对比图")

# 2-3 TOPSIS 排序图
pb.plot_ranking(scores_topsis, CITIES, method_name="TOPSIS")
print("  ✓ TOPSIS 排序图")

# 2-4 雷达图（使用归一化原始数据）
X_norm = (raw_data - raw_data.min(0)) / (raw_data.max(0) - raw_data.min(0) + 1e-10)
pb.plot_radar(X_norm, INDICATORS, CITIES)
print("  ✓ 雷达图")

# 2-5 相关性热力图
pb.plot_correlation_heatmap(raw_data, INDICATORS)
print("  ✓ 相关性热力图")

# 2-6 灵敏度分析
pb.plot_sensitivity(sens_result, CITIES)
print("  ✓ 灵敏度分析图")

# 2-7 PCA 碎石图
pb.plot_pca_scree(
    pca_result["eigenvalues"],
    pca_result["explained_var"],
    n_selected=pca_result["n_components"],
)
print("  ✓ PCA 碎石图")

# 2-8 两方法得分对比散点
pb.plot_score_scatter(
    scores_topsis, scores_gra, CITIES,
    method_x="TOPSIS", method_y="GRA",
)
print("  ✓ 得分散点对比图")

# 2-9 AHP 热力图
pb.plot_ahp_heatmap(jm, INDICATORS[:n_sub])
print("  ✓ AHP 判断矩阵热力图")

# 批量保存所有图表
saved = pb.save_all()
print(f"\n共保存 {len(saved)} 张图表到 output/figures/")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 生成 LaTeX 报告
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(">>> Step 3: 生成 LaTeX 报告")
print("=" * 60)

lb = LatexBuilder(
    language="cn",
    output_dir="output/reports",
    fig_dir="figures",
    decimal_places=4,
)

# 按工作流顺序添加各算法段落
lb.add_pca_section(pca_result, label="city")
lb.add_entropy_section(entropy_result, label="city")
lb.add_critic_section(critic_result, label="city")

lb.add_combination_weight_section(
    result={
        "indicators":         INDICATORS,
        "subjective_weights": ahp_result["weights"].tolist() + [0.10, 0.05, 0.05],
        "objective_weights":  weights_ew.tolist(),
        "combined_weights":   (weights_ew * 0.5 + 0.5 *
                               np.pad(ahp_result["weights"],
                                      (0, len(weights_ew) - len(ahp_result["weights"])))).tolist(),
        "method":             "乘法合成",
    },
    label="city",
)

lb.add_topsis_section(topsis_result, label="city")
lb.add_gra_section(gra_result, label="city")

lb.add_sensitivity_section(
    result={"delta": 20.0, "is_robust": True},
    label="city",
)

lb.add_figure("sensitivity_oat.png",
              "权重灵敏度分析（OAT）", "sensitivity_city")
lb.add_figure("radar_chart.png",
              "各城市创新能力综合雷达图", "radar_city")

lb.add_double_figure(
    "ranking_topsis.png", "TOPSIS 综合得分排序", "rank_topsis_city",
    "scatter_topsis_gra.png", "TOPSIS vs GRA 得分对比", "scatter_city",
    main_caption="综合评价结果可视化",
    main_label="result_vis_city",
)

lb.add_final_result_section(
    result={
        "objects": CITIES,
        "scores":  scores_topsis,
        "ranking": ranking_topsis,
    },
    label="city",
)

tex_path = lb.save(
    filename="city_innovation_report.tex",
    title="城市创新能力综合评价研究报告",
    author="AutoEval-Modeling 自动生成",
    abstract=(
        "本报告对五个城市的创新能力进行了系统综合评价。"
        "采用 PCA 消除指标共线性，结合熵权法与 CRITIC 法确定客观权重，"
        "以 TOPSIS 与灰色关联分析进行综合排序，"
        "并通过 OAT 灵敏度分析验证结果稳健性。"
        "评价结果表明上海综合创新能力最强，广州相对较弱。"
    ),
)
print(f"LaTeX 报告已生成: {tex_path}")
print(f"段落数量: {lb.section_count}")

# ─────────────────────────────────────────────────────────────────────────────
# 输出汇总
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(">>> 生成产物汇总")
print("=" * 60)
print(f"  Python 脚本 : output/scripts/city_innovation_eval.py")
print(f"  图表目录    : output/figures/ ({len(saved)} 张)")
print(f"  LaTeX 报告  : output/reports/city_innovation_report.tex")
print("  " + "-" * 40)
print(f"  {cb!r}")
print(f"  {pb!r}")
print(f"  {lb!r}")
print("=" * 60)
print("全部完成！")