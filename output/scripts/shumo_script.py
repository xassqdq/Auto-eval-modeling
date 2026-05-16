# -*- coding: utf-8 -*-
"""
shumo 综合评价模型求解脚本
================
自动生成时间 : 2026-05-16 09:32:38
生成工具     : AutoEval-Modeling (https://github.com/AutoEval-Modeling)

工作流摘要
----------
  1. 数据加载与基本信息显示
  2. 数据预处理（缺失值填充、正向化、Min-Max 归一化）
  3. CRITIC 法计算客观权重
  4. TOPSIS 综合评价
  5. 权重灵敏度分析（OAT）
  6. 保存结果

使用说明
--------
1. 安装依赖: pip install numpy pandas scipy matplotlib seaborn openpyxl
2. 将数据文件放置于脚本同目录（或修改 DATA_PATH 变量）
3. 直接运行: python shumo_script.py
4. 结果图片将保存在 ./output/figures/ 目录下

注意事项
--------
- 本脚本由 AutoEval-Modeling 自动生成，请在学术使用前核实计算逻辑。
- 如遇数据格式问题，请检查 DATA_PATH 指向文件的编码与分隔符设置。
"""

import os
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 字体设置（支持中文显示）─────────────────────────────────────────────────
def _setup_font():
    font_candidates = [
        "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
        "PingFang SC", "STHeiti", "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in font_candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

_setup_font()

# ── 全局输出目录 ─────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
FIG_DIR    = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 第1节：数据加载与基本信息
# =============================================================================
DATA_PATH        = "data/city_innovation.csv"         # 数据文件路径
OBJECT_COL       = "城市"        # 评价对象列名
INDICATOR_COLS   = ['RD投入_亿元', '专利授权数_件', '高新企业数_家', 'GDP增长率_百分比']      # 评价指标列名列表
# 指标方向: 1=正向（越大越好），-1=负向（越小越好），0=适度型
INDICATOR_TYPES  = [1, 1, 1, 1]


def load_data(path: str) -> pd.DataFrame:
    """加载数据文件（支持 CSV / Excel）。"""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        # 自动检测分隔符
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    logger.info("数据加载完成: shape=%s", df.shape)
    return df


df_raw = load_data(DATA_PATH)
print("=" * 60)
print("原始数据预览：")
print(df_raw.head())
print(f"\n数据规模: {df_raw.shape[0]} 个评价对象，{len(INDICATOR_COLS)} 个指标")
print("=" * 60)


# =============================================================================
# 第2节：数据预处理
# =============================================================================

def handle_missing(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """缺失值处理：数值列用均值填充，并记录填充情况。"""
    df = df.copy()
    miss_cnt = df[cols].isna().sum()
    if miss_cnt.any():
        logger.warning("发现缺失值：\n%s", miss_cnt[miss_cnt > 0].to_string())
        df[cols] = df[cols].fillna(df[cols].mean())
    return df


def forward_transform(matrix: np.ndarray, types: list) -> np.ndarray:
    """
    指标正向化：
      - 正向指标（type= 1）：保持不变
      - 负向指标（type=-1）：取负（max - x）
      - 适度指标（type= 0）：1 / (1 + |x - optimal|)，optimal 取均值
    """
    mat = matrix.copy().astype(float)
    for j, t in enumerate(types):
        col = mat[:, j]
        if t == -1:
            mat[:, j] = col.max() - col
        elif t == 0:
            optimal = col.mean()
            mat[:, j] = 1.0 / (1.0 + np.abs(col - optimal))
    return mat


def minmax_normalize(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Min-Max 归一化，将各指标缩放至 [0, 1]。"""
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    denom   = np.where((col_max - col_min) < eps, eps, col_max - col_min)
    return (matrix - col_min) / denom


def zscore_normalize(matrix: np.ndarray) -> np.ndarray:
    """Z-score 标准化。"""
    return (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0, ddof=1) + 1e-10)


# ── 执行预处理 ───────────────────────────────────────────────────────────────
df_clean = handle_missing(df_raw, INDICATOR_COLS)
objects  = df_clean[OBJECT_COL].tolist()
X_raw    = df_clean[INDICATOR_COLS].values.astype(float)

X_pos    = forward_transform(X_raw, INDICATOR_TYPES)   # 正向化
X_norm   = minmax_normalize(X_pos)                      # Min-Max 归一化

print(f"预处理后数据形状: {X_norm.shape}")
print("归一化矩阵（前3行）：")
print(pd.DataFrame(X_norm, columns=INDICATOR_COLS).head(3).to_string())


# =============================================================================
# 第3节：CRITIC 法计算指标权重
# =============================================================================

def critic_weight(matrix: np.ndarray, eps: float = 1e-10) -> dict:
    """
    CRITIC 法：综合对比强度（标准差）与冲突性（相关系数）确定权重。
    """
    m, n    = matrix.shape
    sigma   = matrix.std(axis=0, ddof=1)                    # 各指标标准差

    # Pearson 相关系数矩阵
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(matrix.T)
    corr = np.nan_to_num(corr, nan=0.0)

    conflict = np.array(
        [(1 - corr[j, :]).sum() for j in range(n)]
    )                                                        # 冲突性
    C_j  = sigma * conflict                                  # 信息量
    w_j  = C_j / (C_j.sum() + eps)

    return {
        "weights":      w_j,
        "std_devs":     sigma,
        "conflicts":    conflict,
        "info_amounts": C_j,
        "corr_matrix":  corr,
    }


critic_result = critic_weight(X_norm)
weights_crit  = critic_result["weights"]

print("\n【CRITIC 法权重】")
for name, w, s, c, ci in zip(
    INDICATOR_COLS,
    weights_crit,
    critic_result["std_devs"],
    critic_result["conflicts"],
    critic_result["info_amounts"],
):
    print(f"  {name:12s}: σ={s:.4f}, 冲突性={c:.4f}, Cj={ci:.4f}, 权重={w:.4f}")


# =============================================================================
# 第4节：TOPSIS 综合评价
# =============================================================================

def topsis(matrix: np.ndarray, weights: np.ndarray, eps: float = 1e-10) -> dict:
    """
    TOPSIS 综合评价。

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n)
        已归一化的决策矩阵（值域 [0,1]，正向化处理后）。
    weights : np.ndarray, shape (n,)
        指标权重向量（归一化）。

    Returns
    -------
    dict
        scores, ranking, d_positive, d_negative
    """
    # 加权规范化矩阵
    V    = matrix * weights

    # 正负理想解
    V_pos = V.max(axis=0)
    V_neg = V.min(axis=0)

    # 欧氏距离
    D_pos = np.sqrt(((V - V_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - V_neg) ** 2).sum(axis=1))

    # 相对贴近度
    scores  = D_neg / (D_pos + D_neg + eps)
    ranking = scores.argsort()[::-1] + 1          # 1-based 排名（降序）

    return {
        "scores":     scores,
        "ranking":    ranking,
        "d_positive": D_pos,
        "d_negative": D_neg,
    }


# 使用当前工作流选定的权重（修改 ACTIVE_WEIGHTS 以切换）
ACTIVE_WEIGHTS = weights_crit   # 例如 weights_ew / weights_ahp / weights_combined
topsis_result  = topsis(X_norm, np.array(ACTIVE_WEIGHTS))
scores_topsis  = topsis_result["scores"]
ranking_topsis = topsis_result["ranking"]

print("\n【TOPSIS 综合评价结果】")
result_df = pd.DataFrame({
    "评价对象":    objects,
    "D+":         topsis_result["d_positive"].round(4),
    "D-":         topsis_result["d_negative"].round(4),
    "综合得分 Ci": scores_topsis.round(4),
    "排名":        ranking_topsis,
}).sort_values("排名").reset_index(drop=True)
print(result_df.to_string(index=False))


# =============================================================================
# 第5节：权重灵敏度分析（OAT 单因素扰动法）
# =============================================================================

def weight_sensitivity(
    matrix: np.ndarray,
    base_weights: np.ndarray,
    eval_func,
    delta_range: float = 0.20,
    steps: int = 21,
    method_name: str = "TOPSIS",
) -> dict:
    """
    对每个指标的权重在 [-delta_range, +delta_range] 范围内逐步扰动，
    记录各评价对象的综合排名变化。

    Parameters
    ----------
    eval_func : callable
        接受 (matrix, weights) 返回包含 "ranking" 数组的字典。
    delta_range : float
        扰动范围比例（如 0.20 表示 ±20%）。

    Returns
    -------
    dict
        "perturb_ratios" : 扰动比例数组
        "rank_records"   : dict[indicator_name] -> (steps, n_objects) 排名矩阵
        "is_robust"      : 整体是否稳健（各对象排名始终不变）
    """
    n_ind   = len(base_weights)
    ratios  = np.linspace(-delta_range, delta_range, steps)
    rank_records = {}
    is_robust    = True

    base_result = eval_func(matrix, base_weights)
    base_ranking = base_result["ranking"]

    for j, ind_name in enumerate(INDICATOR_COLS):
        rank_matrix = []
        for ratio in ratios:
            w_new    = base_weights.copy()
            w_new[j] = max(0.0, w_new[j] * (1 + ratio))
            total    = w_new.sum()
            if total < 1e-10:
                rank_matrix.append(base_ranking.copy())
                continue
            w_new = w_new / total                        # 重新归一化
            res   = eval_func(matrix, w_new)
            rank_matrix.append(res["ranking"].copy())
        rank_matrix = np.vstack(rank_matrix)             # (steps, n_objects)
        rank_records[ind_name] = rank_matrix
        # 检查是否与基准排名一致
        if not np.all(rank_matrix == base_ranking):
            is_robust = False

    return {
        "perturb_ratios": ratios,
        "rank_records":   rank_records,
        "is_robust":      is_robust,
    }


# 以 TOPSIS 为基础进行灵敏度分析
sens_result = weight_sensitivity(X_norm, np.array(ACTIVE_WEIGHTS), topsis)
print(f"\n【灵敏度分析】整体稳健性: "
      f"{'✓ 稳健' if sens_result['is_robust'] else '✗ 部分不稳健'}")


# =============================================================================
# 第6节：保存结果
# =============================================================================

def save_results(result_df: pd.DataFrame, filename: str = "eval_results.csv"):
    """将综合评价结果保存为 CSV 文件。"""
    out_path = OUTPUT_DIR / filename
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("结果已保存: %s", out_path)
    return out_path


save_results(result_df, "topsis_results.csv")
print(f"\n所有结果文件已保存至 {OUTPUT_DIR} 目录。")
print("=" * 60)


# =============================================================================
# 脚本入口
# =============================================================================
if __name__ == "__main__":
    print("AutoEval-Modeling 自动评价脚本运行完成。")
    print(f"图片已保存至: {FIG_DIR}")
