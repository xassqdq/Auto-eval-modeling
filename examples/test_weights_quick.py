# -*- coding: utf-8 -*-
"""
examples/test_weights_quick.py
快速验证所有赋权方法是否正常运行（无需 pytest）
"""

import numpy as np
import pandas as pd
import warnings

# 抑制字体警告
warnings.filterwarnings("ignore", category=UserWarning)

from src.algorithms.weights import (
    AHPMethod, BinomialCoefficientMethod, RingRatioScoringMethod,
    EntropyWeightMethod, CRITICMethod, StdDeviationMethod, PCAWeightMethod,
    MultiplicativeCombination, LinearCombination,
    GameTheoryCombination, MinDeviationCombination,
)

SEPARATOR = "─" * 55

def print_section(title: str) -> None:
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print(f"{'═'*55}")


# ── 公共数据 ──────────────────────────────────────────────────
np.random.seed(42)
X = np.abs(np.random.randn(8, 4)) + 1.0
df = pd.DataFrame(X, columns=["C1", "C2", "C3", "C4"])

IND_NAMES = ["C1", "C2", "C3", "C4"]

# ── 1. AHP ────────────────────────────────────────────────────
print_section("1. AHP 层次分析法")
A = np.array([
    [1,    2,    4,   8  ],
    [0.5,  1,    2,   4  ],
    [0.25, 0.5,  1,   2  ],
    [0.125,0.25, 0.5, 1  ],
])
ahp = AHPMethod(indicator_names=IND_NAMES)
ahp.fit(A).compute()
r_ahp = ahp.summary()
w_ahp = r_ahp["weights"]

# ── 2. 二项系数法 ─────────────────────────────────────────────
print_section("2. 二项系数法")
rankings = np.array([
    [1, 2, 3, 4],
    [2, 1, 3, 4],
    [1, 3, 2, 4],
])
bcm = BinomialCoefficientMethod(indicator_names=IND_NAMES)
bcm.fit(rankings).compute()
bcm.summary()

# ── 3. 环比评分法 ─────────────────────────────────────────────
print_section("3. 环比评分法")
ratios = [[2.0, 1.5, 1.2], [1.8, 1.6, 1.3]]
rrs = RingRatioScoringMethod(
    indicator_names=IND_NAMES,
    order=[0, 1, 2, 3],
)
rrs.fit(ratios).compute()
rrs.summary()

# ── 4. 熵权法 ─────────────────────────────────────────────────
print_section("4. 熵权法")
ew = EntropyWeightMethod(indicator_names=IND_NAMES)
ew.fit(df).compute()
r_ew = ew.summary()
w_ew = r_ew["weights"]

# ── 5. CRITIC 法 ──────────────────────────────────────────────
print_section("5. CRITIC 法")
cr = CRITICMethod(indicator_names=IND_NAMES)
cr.fit(df).compute()
r_cr = cr.summary()
w_cr = r_cr["weights"]

# ── 6. 标准离差法 ─────────────────────────────────────────────
print_section("6. 标准离差法")
sd = StdDeviationMethod(indicator_names=IND_NAMES, normalization="minmax")
sd.fit(df).compute()
sd.summary()

# ── 7. PCA 权重法 ─────────────────────────────────────────────
print_section("7. PCA 权重法")
pca_w = PCAWeightMethod(indicator_names=IND_NAMES, n_components=0.85)
pca_w.fit(df).compute()
pca_w.summary()

# ── 8. 乘法合成 ───────────────────────────────────────────────
print_section("8. 乘法合成组合赋权")
mc = MultiplicativeCombination(
    method_names=["AHP", "熵权法"],
    indicator_names=IND_NAMES,
)
mc.fit([w_ahp, w_ew]).compute()
mc.summary()

# ── 9. 线性加权 ───────────────────────────────────────────────
print_section("9. 线性加权组合赋权（等权）")
lc = LinearCombination(
    method_names=["AHP", "熵权法", "CRITIC"],
    indicator_names=IND_NAMES,
)
lc.fit([w_ahp, w_ew, w_cr]).compute()
lc.summary()

# ── 10. 博弈论组合 ────────────────────────────────────────────
print_section("10. 博弈论组合赋权")
gtc = GameTheoryCombination(
    method_names=["AHP", "熵权法", "CRITIC"],
    indicator_names=IND_NAMES,
    solver="kkt",
)
gtc.fit([w_ahp, w_ew, w_cr]).compute()
r_gtc = gtc.summary()

# ── 11. 离差最小化 ────────────────────────────────────────────
print_section("11. 离差最小化组合赋权（含决策矩阵）")
mdc = MinDeviationCombination(
    method_names=["AHP", "熵权法"],
    indicator_names=IND_NAMES,
    decision_matrix=X,
)
mdc.fit([w_ahp, w_ew]).compute()
mdc.summary()

# ── 12. tex_description 生成 ─────────────────────────────────
print_section("12. LaTeX 片段生成测试")
for name, obj in [
    ("AHP",        ahp),
    ("熵权法",     ew),
    ("CRITIC",     cr),
    ("博弈论组合", gtc),
    ("离差最小化", mdc),
]:
    tex = obj.tex_description()
    print(f"  [{name}] LaTeX 字数: {len(tex)} 字符 ✓")

# ── 13. 可视化 ────────────────────────────────────────────────
print_section("13. 可视化测试（保存图片）")
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("output/figures")
out_dir.mkdir(parents=True, exist_ok=True)

for fname, method in [
    ("ahp_plot",   ahp),
    ("ew_plot",    ew),
    ("cr_plot",    cr),
    ("gtc_plot",   gtc),
    ("mdc_plot",   mdc),
]:
    try:
        fig = method.plot()
        fig.savefig(out_dir / f"{fname}.png", dpi=150,
                    bbox_inches="tight")
        plt.close("all")
        print(f"  [{fname}] 已保存 ✓")
    except Exception as e:
        print(f"  [{fname}] 绘图异常: {e}")

print(f"\n{'═'*55}")
print("  所有赋权方法验证完成！")
print(f"{'═'*55}\n")