"""
AutoEval-Modeling: Gradio Web 交互界面 (备选)
==============================================
基于 Gradio 的轻量级交互界面，适用于快速演示与 API 共享。

启动方式:
    python app/gradio_app.py
"""

import os
import sys
import json
import time
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

# ── 将项目根目录加入 sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
except ImportError:
    raise ImportError("请安装 Gradio: pip install gradio>=4.0")

# ── 延迟导入后端模块 ──
try:
    from src.core.workflow import WorkflowEngine
    from src.core.recommendation import AlgorithmRecommender
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


# ╔══════════════════════════════════════════════════════════════╗
# ║                      常量与配置                               ║
# ╚══════════════════════════════════════════════════════════════╝

APP_TITLE = "AutoEval-Modeling"
APP_DESCRIPTION = """
## 📊 面向评价类数学建模的自动化工作流引擎

上传数据 → 配置参数 → 一键生成评价结果、Python代码与LaTeX报告

**支持的评价方法：** TOPSIS、VIKOR、灰色关联分析、模糊综合评价、ELECTRE、RSR、DEA 等  
**支持的赋权方法：** AHP、熵权法、CRITIC、标准离差法、PCA、组合赋权等
"""

WEIGHT_METHODS = [
    "entropy (熵权法)",
    "critic (CRITIC法)",
    "stddev (标准离差法)",
    "pca_weight (PCA求权重)",
    "ahp (层次分析法)",
]

EVAL_METHODS = [
    "topsis (TOPSIS)",
    "vikor (VIKOR)",
    "gra (灰色关联分析)",
    "fuzzy_eval (模糊综合评价)",
    "rsr (秩和比法)",
    "dea (数据包络分析)",
]

NORM_METHODS = [
    "minmax (Min-Max归一化)",
    "zscore (Z-Score标准化)",
    "vector (向量归一化)",
]


# ╔══════════════════════════════════════════════════════════════╗
# ║                      核心处理函数                              ║
# ╚══════════════════════════════════════════════════════════════╝

def parse_method_code(method_str: str) -> str:
    """从显示字符串中提取方法代码"""
    return method_str.split("(")[0].strip()


def load_and_preview(file_obj) -> Tuple[Optional[pd.DataFrame], str, str, str]:
    """
    加载数据文件并返回预览信息。
    Returns: (dataframe, preview_html, column_info, auto_detect_info)
    """
    if file_obj is None:
        return None, "请上传数据文件", "", ""

    try:
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)

        if file_path.endswith(".csv"):
            for enc in ["utf-8", "gbk", "gb2312", "latin1"]:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            else:
                return None, "❌ 无法解析 CSV 编码", "", ""
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            return None, f"❌ 不支持的文件格式: {file_path}", "", ""

        # 预览
        preview = df.head(15).to_html(classes="dataframe", index=False)
        preview_html = f"""
        <div style="max-height:400px; overflow:auto;">
            <p><b>数据维度：</b>{df.shape[0]} 行 × {df.shape[1]} 列</p>
            {preview}
        </div>
        """

        # 列信息
        col_info_lines = []
        numeric_cols = []
        object_col_candidate = None
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            missing_pct = f"{missing/len(df)*100:.1f}%"
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            col_info_lines.append(f"| {col} | {dtype} | {missing} ({missing_pct}) | {'✅' if is_numeric else '❌'} |")
            if is_numeric:
                numeric_cols.append(col)
            elif object_col_candidate is None:
                object_col_candidate = col

        col_info = "| 列名 | 类型 | 缺失值 | 数值型 |\n|---|---|---|---|\n" + "\n".join(col_info_lines)

        # 自动检测
        auto_info = f"""
**自动检测结果：**
- 评价对象列（推测）：`{object_col_candidate or '未检测到'}`
- 数值型指标列：{', '.join([f'`{c}`' for c in numeric_cols]) or '无'}
- 总共 {len(numeric_cols)} 个可用指标
"""

        return df, preview_html, col_info, auto_info

    except Exception as e:
        return None, f"❌ 加载失败: {str(e)}", "", ""


def analyze_data_features(
    df: pd.DataFrame,
    indicator_cols_str: str,
) -> str:
    """分析数据特征并生成推荐建议"""
    if df is None:
        return "请先上传数据"

    # 解析指标列
    indicator_cols = [c.strip() for c in indicator_cols_str.split(",") if c.strip()]
    valid_cols = [c for c in indicator_cols if c in df.columns]

    if not valid_cols:
        return "⚠️ 未找到有效的指标列，请检查输入"

    report_lines = []
    report_lines.append("## 📊 数据特征分析报告\n")

    n_objects = len(df)
    n_indicators = len(valid_cols)
    report_lines.append(f"- **样本数：** {n_objects}")
    report_lines.append(f"- **指标数：** {n_indicators}")
    report_lines.append(f"- **样本/指标比：** {n_objects/max(n_indicators,1):.2f}\n")

    sub = df[valid_cols]

    # 缺失值分析
    missing = sub.isnull().sum()
    total_missing = missing.sum()
    report_lines.append(f"### 缺失值")
    if total_missing == 0:
        report_lines.append("✅ 无缺失值\n")
    else:
        report_lines.append(f"⚠️ 共 {total_missing} 个缺失值")
        for col, cnt in missing[missing > 0].items():
            report_lines.append(f"  - {col}: {cnt} ({cnt/n_objects*100:.1f}%)")
        report_lines.append("")

    # 相关性分析
    try:
        corr = sub.corr()
        high_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                r = abs(corr.iloc[i, j])
                if r > 0.8:
                    high_pairs.append((corr.columns[i], corr.columns[j], round(r, 3)))

        report_lines.append("### 指标相关性")
        if high_pairs:
            report_lines.append("⚠️ **发现高相关性指标对 (|r| > 0.8):**")
            for c1, c2, r in high_pairs:
                report_lines.append(f"  - {c1} ↔ {c2}: r = {r}")
            report_lines.append("\n💡 **建议启用 PCA 降维**\n")
        else:
            report_lines.append("✅ 未发现显著高相关性\n")
    except Exception:
        report_lines.append("### 指标相关性\n⚠️ 无法计算相关矩阵\n")

    # 算法推荐
    report_lines.append("### 🤖 推荐算法组合\n")

    recommendations = []
    if high_pairs:
        recommendations.append(("PCA降维 → CRITIC → TOPSIS", "消除多重共线性，客观反映差异"))
    if n_objects / max(n_indicators, 1) < 3:
        recommendations.append(("熵权法 → 灰色关联分析", "样本少，灰色关联对数据量要求低"))
    if not recommendations:
        recommendations.append(("熵权法 → TOPSIS", "通用性强，适合大多数评价场景"))
        recommendations.append(("CRITIC → VIKOR", "兼顾对比性与冲突性"))

    for i, (algo, reason) in enumerate(recommendations, 1):
        report_lines.append(f"**方案 {i}：** {algo}")
        report_lines.append(f"  _理由：{reason}_\n")

    return "\n".join(report_lines)


def run_evaluation(
    file_obj,
    object_col: str,
    indicator_cols_str: str,
    directions_str: str,
    norm_method: str,
    weight_method: str,
    eval_method: str,
    enable_pca: bool,
    enable_sensitivity: bool,
    output_language: str,
    progress=gr.Progress(),
) -> Tuple[str, str, str, str, Optional[str]]:
    """
    执行评价工作流。

    Returns:
        (result_summary, weights_table, ranking_table, generated_code, latex_content)
    """
    progress(0, desc="初始化...")

    # ── 1. 数据加载 ──
    if file_obj is None:
        return "❌ 请先上传数据文件", "", "", "", None

    try:
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        if file_path.endswith(".csv"):
            for enc in ["utf-8", "gbk", "gb2312"]:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        return f"❌ 数据加载失败: {e}", "", "", "", None

    progress(0.1, desc="数据加载完成")

    # ── 2. 解析配置 ──
    indicator_cols = [c.strip() for c in indicator_cols_str.split(",") if c.strip()]
    valid_cols = [c for c in indicator_cols if c in df.columns]

    if not valid_cols:
        return "❌ 未找到有效指标列", "", "", "", None

    # 解析方向
    directions = {}
    if directions_str.strip():
        for item in directions_str.split(","):
            parts = item.strip().split(":")
            if len(parts) == 2:
                col_name = parts[0].strip()
                direction = parts[1].strip().lower()
                if col_name in valid_cols:
                    directions[col_name] = direction
    # 默认正向
    for col in valid_cols:
        if col not in directions:
            directions[col] = "positive"

    object_col = object_col.strip() if object_col else None
    if object_col and object_col not in df.columns:
        object_col = None

    objects = (
        df[object_col].tolist()
        if object_col
        else [f"对象{i+1}" for i in range(len(df))]
    )

    wm = parse_method_code(weight_method)
    em = parse_method_code(eval_method)
    nm = parse_method_code(norm_method)

    progress(0.2, desc="配置解析完成")

    # ── 3. 数据预处理：标准化 ──
    X = df[valid_cols].values.astype(float)
    n, m = X.shape

    # 处理负向指标
    X_processed = X.copy()
    for j, col in enumerate(valid_cols):
        if directions.get(col) == "negative":
            X_processed[:, j] = -X_processed[:, j]

    # 标准化
    if nm == "minmax":
        col_min = X_processed.min(axis=0)
        col_max = X_processed.max(axis=0)
        denom = col_max - col_min
        denom[denom == 0] = 1
        X_norm = (X_processed - col_min) / denom
    elif nm == "zscore":
        col_mean = X_processed.mean(axis=0)
        col_std = X_processed.std(axis=0)
        col_std[col_std == 0] = 1
        X_norm = (X_processed - col_mean) / col_std
        # 转为正值域
        X_norm = X_norm - X_norm.min(axis=0) + 0.01
    else:
        # 向量归一化
        norms = np.sqrt(np.sum(X_processed ** 2, axis=0))
        norms[norms == 0] = 1
        X_norm = X_processed / norms

    progress(0.35, desc="数据预处理完成")

    # ── 4. PCA 降维（可选）──
    pca_info = ""
    if enable_pca:
        try:
            from numpy.linalg import eigh
            cov = np.cov(X_norm, rowvar=False)
            eigenvalues, eigenvectors = eigh(cov)
            # 按特征值降序排列
            idx = np.argsort(-eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            # 累计方差贡献率
            total_var = eigenvalues.sum()
            cum_var = np.cumsum(eigenvalues) / total_var
            n_components = np.searchsorted(cum_var, 0.85) + 1
            n_components = max(2, min(n_components, m))
            X_pca = X_norm @ eigenvectors[:, :n_components]
            pca_info = f"PCA降维：{m}维 → {n_components}维（累计方差贡献率 ≥ {cum_var[n_components-1]*100:.1f}%）"
            X_norm = X_pca
            valid_cols = [f"PC{i+1}" for i in range(n_components)]
            m = n_components
        except Exception as e:
            pca_info = f"PCA降维失败: {e}，使用原始数据"

    progress(0.45, desc="降维处理完成")

    # ── 5. 权重计算 ──
    weights = np.ones(m) / m  # 默认等权

    if wm == "entropy":
        # 熵权法
        P = X_norm / (X_norm.sum(axis=0, keepdims=True) + 1e-12)
        P = np.clip(P, 1e-12, None)
        k = 1.0 / np.log(max(n, 2))
        E = -k * np.sum(P * np.log(P), axis=0)
        D = 1 - E
        weights = D / (D.sum() + 1e-12)
    elif wm == "critic":
        # CRITIC 法
        std = X_norm.std(axis=0)
        corr = np.corrcoef(X_norm.T)
        if corr.ndim == 0:
            corr = np.array([[1.0]])
        conflict = np.sum(1 - corr, axis=1)
        info = std * conflict
        weights = info / (info.sum() + 1e-12)
    elif wm == "stddev":
        # 标准离差法
        std = X_norm.std(axis=0)
        weights = std / (std.sum() + 1e-12)
    elif wm == "pca_weight":
        # PCA 权重（基于特征值）
        try:
            cov = np.cov(X_norm, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            weights = eigenvalues[:m] / (eigenvalues[:m].sum() + 1e-12)
        except Exception:
            weights = np.ones(m) / m

    weights = weights / (weights.sum() + 1e-12)  # 确保归一化

    progress(0.6, desc="权重计算完成")

    # ── 6. 综合评价 ──
    scores = np.zeros(n)

    if em == "topsis":
        # TOPSIS
        Z = X_norm * weights
        Z_pos = Z.max(axis=0)
        Z_neg = Z.min(axis=0)
        D_pos = np.sqrt(np.sum((Z - Z_pos) ** 2, axis=1))
        D_neg = np.sqrt(np.sum((Z - Z_neg) ** 2, axis=1))
        scores = D_neg / (D_pos + D_neg + 1e-12)

    elif em == "vikor":
        # VIKOR
        Z = X_norm * weights
        f_best = Z.max(axis=0)
        f_worst = Z.min(axis=0)
        denom = f_best - f_worst
        denom[denom == 0] = 1
        S = np.sum(weights * (f_best - Z) / denom, axis=1)
        R = np.max(weights * (f_best - Z) / denom, axis=1)
        S_min, S_max = S.min(), S.max()
        R_min, R_max = R.min(), R.max()
        v = 0.5  # 折衷系数
        Q = v * (S - S_min) / (S_max - S_min + 1e-12) + (1 - v) * (R - R_min) / (R_max - R_min + 1e-12)
        scores = 1 - Q  # 转为越大越好

    elif em == "gra":
        # 灰色关联分析
        ref = X_norm.max(axis=0)  # 参考序列
        rho = 0.5  # 分辨系数
        delta = np.abs(X_norm - ref)
        delta_min = delta.min()
        delta_max = delta.max()
        xi = (delta_min + rho * delta_max) / (delta + rho * delta_max + 1e-12)
        scores = np.sum(weights * xi, axis=1)

    elif em == "rsr":
        # 秩和比法
        ranks = np.zeros_like(X_norm)
        for j in range(m):
            ranks[:, j] = pd.Series(X_norm[:, j]).rank(ascending=True).values
        rsr_values = np.sum(weights * ranks, axis=1) / n
        scores = rsr_values

    else:
        # 默认加权和
        scores = np.sum(X_norm * weights, axis=1)

    progress(0.75, desc="综合评价完成")

    # ── 7. 灵敏度分析（可选）──
    sensitivity_info = ""
    if enable_sensitivity:
        original_rank = np.argsort(-scores) + 1
        perturbation = 0.1
        rank_changes = 0
        for j in range(m):
            w_perturbed = weights.copy()
            w_perturbed[j] *= (1 + perturbation)
            w_perturbed = w_perturbed / w_perturbed.sum()

            if em == "topsis":
                Z2 = X_norm * w_perturbed
                D_p2 = np.sqrt(np.sum((Z2 - Z2.max(axis=0)) ** 2, axis=1))
                D_n2 = np.sqrt(np.sum((Z2 - Z2.min(axis=0)) ** 2, axis=1))
                scores2 = D_n2 / (D_p2 + D_n2 + 1e-12)
            else:
                scores2 = np.sum(X_norm * w_perturbed, axis=1)

            new_rank = np.argsort(-scores2) + 1
            if not np.array_equal(original_rank, new_rank):
                rank_changes += 1

        sensitivity_info = f"灵敏度分析：对 {m} 个指标权重 ±{perturbation*100:.0f}% 扰动，{rank_changes} 次排名变动"
        if rank_changes == 0:
            sensitivity_info += "（排名完全稳健 ✅）"
        elif rank_changes <= m * 0.3:
            sensitivity_info += "（排名基本稳健 ✅）"
        else:
            sensitivity_info += "（排名存在一定波动 ⚠️）"

    progress(0.85, desc="灵敏度分析完成")

    # ── 8. 组装结果 ──

    # 权重表
    weight_df = pd.DataFrame({
        "指标": valid_cols,
        "权重": np.round(weights, 4),
    }).sort_values("权重", ascending=False)
    weights_table = weight_df.to_markdown(index=False)

    # 排名表
    sorted_idx = np.argsort(-scores)
    ranking_data = []
    for rank, idx in enumerate(sorted_idx, 1):
        ranking_data.append({
            "排名": rank,
            "评价对象": objects[idx],
            "综合得分": round(float(scores[idx]), 4),
        })
    ranking_df = pd.DataFrame(ranking_data)
    ranking_table = ranking_df.to_markdown(index=False)

    # 结果摘要
    summary_lines = [
        f"## ✅ 评价完成\n",
        f"- **评价对象数：** {n}",
        f"- **指标数：** {m}",
        f"- **标准化方法：** {nm}",
        f"- **赋权方法：** {wm}",
        f"- **评价模型：** {em}",
    ]
    if pca_info:
        summary_lines.append(f"- **PCA：** {pca_info}")
    if sensitivity_info:
        summary_lines.append(f"- **{sensitivity_info}**")
    summary_lines.append(f"\n### 🏆 最优对象：**{objects[sorted_idx[0]]}**（得分 {scores[sorted_idx[0]]:.4f}）")
    result_summary = "\n".join(summary_lines)

    progress(0.9, desc="生成代码与报告...")

    # ── 9. 生成 Python 代码 ──
    generated_code = _generate_code(
        data_file=file_obj.name if hasattr(file_obj, 'name') else "data.csv",
        object_col=object_col,
        indicator_cols=indicator_cols,
        directions=directions,
        norm_method=nm,
        weight_method=wm,
        eval_method=em,
    )

    # ── 10. 生成 LaTeX ──
    latex_content = _generate_latex(
        wm, em, valid_cols, weights, ranking_data
    )

    progress(1.0, desc="全部完成！")

    return result_summary, weights_table, ranking_table, generated_code, latex_content


def _generate_code(
    data_file: str,
    object_col: Optional[str],
    indicator_cols: List[str],
    directions: Dict[str, str],
    norm_method: str,
    weight_method: str,
    eval_method: str,
) -> str:
    """生成完整的 Python 求解代码"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    code = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoEval-Modeling 自动生成脚本
生成时间: {timestamp}
赋权方法: {weight_method}
评价模型: {eval_method}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════
# 1. 数据加载
# ═══════════════════════════════════════════════
print("=" * 60)
print("Step 1: 数据加载")
print("=" * 60)

df = pd.read_csv("{data_file}")
print(f"数据维度: {{df.shape}}")
print(df.head())

object_col = {repr(object_col)}
indicator_cols = {indicator_cols}
directions = {directions}

X = df[indicator_cols].values.astype(float)
n, m = X.shape
objects = df[object_col].tolist() if object_col and object_col in df.columns else [f"对象{{i+1}}" for i in range(n)]
print(f"\\n评价对象: {{n}} 个")
print(f"评价指标: {{m}} 个")

# ═══════════════════════════════════════════════
# 2. 数据预处理
# ═══════════════════════════════════════════════
print("\\n" + "=" * 60)
print("Step 2: 数据预处理")
print("=" * 60)

# 2.1 负向指标正向化
X_processed = X.copy()
for j, col in enumerate(indicator_cols):
    if directions.get(col, "positive") == "negative":
        X_processed[:, j] = X_processed[:, j].max() - X_processed[:, j]
        print(f"  [负向→正向] {{col}}")

# 2.2 标准化
'''

    if norm_method == "minmax":
        code += '''col_min = X_processed.min(axis=0)
col_max = X_processed.max(axis=0)
denom = col_max - col_min
denom[denom == 0] = 1
X_norm = (X_processed - col_min) / denom
print("标准化方法: Min-Max 归一化")
'''
    elif norm_method == "zscore":
        code += '''col_mean = X_processed.mean(axis=0)
col_std = X_processed.std(axis=0)
col_std[col_std == 0] = 1
X_norm = (X_processed - col_mean) / col_std
X_norm = X_norm - X_norm.min(axis=0) + 0.01  # 平移至正值
print("标准化方法: Z-Score 标准化")
'''
    else:
        code += '''norms = np.sqrt(np.sum(X_processed ** 2, axis=0))
norms[norms == 0] = 1
X_norm = X_processed / norms
print("标准化方法: 向量归一化")
'''

    code += '''
print("标准化后矩阵:")
print(pd.DataFrame(X_norm, columns=indicator_cols, index=objects).round(4))

# ═══════════════════════════════════════════════
# 3. 权重计算
# ═══════════════════════════════════════════════
print("\\n" + "=" * 60)
print("Step 3: 权重计算")
print("=" * 60)
'''

    if weight_method == "entropy":
        code += '''
def entropy_weight(X_norm):
    """熵权法"""
    n, m = X_norm.shape
    P = X_norm / (X_norm.sum(axis=0, keepdims=True) + 1e-12)
    P = np.clip(P, 1e-12, None)
    k = 1.0 / np.log(max(n, 2))
    E = -k * np.sum(P * np.log(P), axis=0)
    D = 1 - E
    W = D / (D.sum() + 1e-12)
    return W, E, D

weights, entropy_vals, diff_coeff = entropy_weight(X_norm)
print(f"赋权方法: 熵权法")
print(f"信息熵: {dict(zip(indicator_cols, np.round(entropy_vals, 4)))}")
print(f"差异系数: {dict(zip(indicator_cols, np.round(diff_coeff, 4)))}")
'''
    elif weight_method == "critic":
        code += '''
def critic_weight(X_norm):
    """CRITIC法"""
    n, m = X_norm.shape
    std = X_norm.std(axis=0)
    corr = np.corrcoef(X_norm.T)
    if corr.ndim == 0:
        corr = np.array([[1.0]])
    conflict = np.sum(1 - corr, axis=1)
    info = std * conflict
    weights = info / (info.sum() + 1e-12)
    return weights

weights = critic_weight(X_norm)
print(f"赋权方法: CRITIC法")
'''
    elif weight_method == "stddev":
        code += '''
std = X_norm.std(axis=0)
weights = std / (std.sum() + 1e-12)
print(f"赋权方法: 标准离差法")
'''
    else:
        code += '''
weights = np.ones(m) / m
print(f"赋权方法: 等权法 (默认)")
'''

    code += '''
print("\\n指标权重:")
weight_df = pd.DataFrame({"指标": indicator_cols, "权重": np.round(weights, 4)})
weight_df = weight_df.sort_values("权重", ascending=False)
print(weight_df.to_string(index=False))

# ═══════════════════════════════════════════════
# 4. 综合评价
# ═══════════════════════════════════════════════
print("\\n" + "=" * 60)
print("Step 4: 综合评价")
print("=" * 60)
'''

    if eval_method == "topsis":
        code += '''
def topsis_eval(X_norm, weights):
    """TOPSIS 逼近理想解排序法"""
    Z = X_norm * weights
    Z_pos = Z.max(axis=0)   # 正理想解
    Z_neg = Z.min(axis=0)   # 负理想解
    D_pos = np.sqrt(np.sum((Z - Z_pos) ** 2, axis=1))
    D_neg = np.sqrt(np.sum((Z - Z_neg) ** 2, axis=1))
    C = D_neg / (D_pos + D_neg + 1e-12)  # 相对贴近度
    return C, D_pos, D_neg

scores, d_pos, d_neg = topsis_eval(X_norm, weights)
method_name = "TOPSIS"
print(f"评价方法: {method_name}")
'''
    elif eval_method == "gra":
        code += '''
def gra_eval(X_norm, weights, rho=0.5):
    """灰色关联分析"""
    ref = X_norm.max(axis=0)
    delta = np.abs(X_norm - ref)
    delta_min = delta.min()
    delta_max = delta.max()
    xi = (delta_min + rho * delta_max) / (delta + rho * delta_max + 1e-12)
    scores = np.sum(weights * xi, axis=1)
    return scores

scores = gra_eval(X_norm, weights)
method_name = "灰色关联分析"
print(f"评价方法: {method_name}")
'''
    elif eval_method == "vikor":
        code += '''
def vikor_eval(X_norm, weights, v=0.5):
    """VIKOR 多准则折衷排序"""
    f_best = X_norm.max(axis=0)
    f_worst = X_norm.min(axis=0)
    denom = f_best - f_worst
    denom[denom == 0] = 1
    S = np.sum(weights * (f_best - X_norm) / denom, axis=1)
    R = np.max(weights * (f_best - X_norm) / denom, axis=1)
    S_min, S_max = S.min(), S.max()
    R_min, R_max = R.min(), R.max()
    Q = v * (S - S_min) / (S_max - S_min + 1e-12) + (1 - v) * (R - R_min) / (R_max - R_min + 1e-12)
    scores = 1 - Q
    return scores

scores = vikor_eval(X_norm, weights)
method_name = "VIKOR"
print(f"评价方法: {method_name}")
'''
    else:
        code += '''
scores = np.sum(X_norm * weights, axis=1)
method_name = "加权求和"
print(f"评价方法: {method_name}")
'''

    code += '''
# 排名
sorted_idx = np.argsort(-scores)
result_df = pd.DataFrame({
    "排名": range(1, n + 1),
    "评价对象": [objects[i] for i in sorted_idx],
    "综合得分": [round(float(scores[i]), 4) for i in sorted_idx],
})
print(f"\\n{method_name} 评价结果:")
print(result_df.to_string(index=False))

# ═══════════════════════════════════════════════
# 5. 结果可视化
# ═══════════════════════════════════════════════
print("\\n" + "=" * 60)
print("Step 5: 结果可视化")
print("=" * 60)

import os
os.makedirs("figures", exist_ok=True)

# 5.1 权重条形图
fig1, ax1 = plt.subplots(figsize=(10, max(4, m * 0.5)))
w_sorted = weight_df.sort_values("权重", ascending=True)
ax1.barh(w_sorted["指标"], w_sorted["权重"], color="steelblue", edgecolor="white")
ax1.set_xlabel("权重", fontsize=12)
ax1.set_title("指标权重分布", fontsize=14)
for i, v in enumerate(w_sorted["权重"]):
    ax1.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=10)
plt.tight_layout()
fig1.savefig("figures/weight_distribution.png", dpi=300, bbox_inches="tight")
print("  ✅ 权重条形图 → figures/weight_distribution.png")

# 5.2 综合得分排名图
fig2, ax2 = plt.subplots(figsize=(10, max(4, n * 0.4)))
result_sorted = result_df.sort_values("综合得分", ascending=True)
colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(result_sorted)))
ax2.barh(result_sorted["评价对象"], result_sorted["综合得分"], color=colors, edgecolor="white")
ax2.set_xlabel("综合得分", fontsize=12)
ax2.set_title(f"{method_name} 综合评价排名", fontsize=14)
for i, (_, row) in enumerate(result_sorted.iterrows()):
    ax2.text(row["综合得分"] + 0.005, i, f'{row["综合得分"]:.4f}', va="center", fontsize=10)
plt.tight_layout()
fig2.savefig("figures/ranking_chart.png", dpi=300, bbox_inches="tight")
print("  ✅ 排名图 → figures/ranking_chart.png")

# 5.3 雷达图（展示前5个对象）
from math import pi
fig3, ax3 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * pi, m, endpoint=False).tolist()
angles += angles[:1]

for rank_i in range(min(5, n)):
    obj_idx = sorted_idx[rank_i]
    values = X_norm[obj_idx].tolist()
    values += values[:1]
    ax3.plot(angles, values, "o-", linewidth=1.5, label=f"{objects[obj_idx]} (#{rank_i+1})")
    ax3.fill(angles, values, alpha=0.08)

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(indicator_cols, fontsize=9)
ax3.set_title("Top-5 对象指标雷达图", fontsize=14, pad=20)
ax3.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
fig3.savefig("figures/radar_chart.png", dpi=300, bbox_inches="tight")
print("  ✅ 雷达图 → figures/radar_chart.png")

print("\\n" + "=" * 60)
print("全部完成！结果保存在 figures/ 目录下。")
print("=" * 60)
'''

    return code


def _generate_latex(
    weight_method: str,
    eval_method: str,
    indicator_cols: List[str],
    weights: np.ndarray,
    ranking_data: List[Dict],
) -> str:
    """生成 LaTeX 报告段落"""
    wm_names = {
        "entropy": "熵权法",
        "critic": "CRITIC法",
        "stddev": "标准离差法",
        "pca_weight": "主成分分析法",
        "ahp": "层次分析法",
    }
    em_names = {
        "topsis": "TOPSIS",
        "vikor": "VIKOR",
        "gra": "灰色关联分析",
        "fuzzy_eval": "模糊综合评价",
        "rsr": "秩和比法",
        "dea": "数据包络分析",
    }

    wm_cn = wm_names.get(weight_method, weight_method)
    em_cn = em_names.get(eval_method, eval_method)

    # 权重表格行
    weight_rows = ""
    for col, w in zip(indicator_cols, weights):
        weight_rows += f"        {col} & {w:.4f} \\\\\n"

    # 排名表格行
    rank_rows = ""
    for item in ranking_data:
        rank_rows += f"        {item['评价对象']} & {item['综合得分']:.4f} & {item['排名']} \\\\\n"

    latex = r"""\documentclass[12pt,a4paper]{ctexart}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{float}
\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}

\begin{document}

\section{评价模型建立与求解}

\subsection{模型概述}

本文采用""" + wm_cn + "确定指标权重，结合" + em_cn + r"""进行综合评价，
构建多属性决策评价模型。模型流程如下：

\begin{enumerate}
    \item 对原始数据进行标准化处理，消除量纲影响；
    \item 利用""" + wm_cn + r"""确定各指标的客观权重；
    \item 基于""" + em_cn + r"""计算各评价对象的综合得分与排名；
    \item 通过灵敏度分析检验结果的稳健性。
\end{enumerate}

\subsection{权重计算结果}

采用""" + wm_cn + r"""计算各指标权重，结果如表\ref{tab:weights}所示。

\begin{table}[H]
    \centering
    \caption{""" + wm_cn + r"""指标权重计算结果}
    \label{tab:weights}
    \begin{tabular}{lc}
        \toprule
        指标 & 权重 \\
        \midrule
""" + weight_rows + r"""        \bottomrule
    \end{tabular}
\end{table}

\subsection{综合评价结果}

基于""" + em_cn + r"""计算各评价对象的综合得分，排名结果如表\ref{tab:ranking}所示。

\begin{table}[H]
    \centering
    \caption{""" + em_cn + r"""综合评价得分与排名}
    \label{tab:ranking}
    \begin{tabular}{lcc}
        \toprule
        评价对象 & 综合得分 & 排名 \\
        \midrule
""" + rank_rows + r"""        \bottomrule
    \end{tabular}
\end{table}

\subsection{结果分析}

由表\ref{tab:ranking}可知，""" + ranking_data[0]["评价对象"] + f"""综合得分最高（{ranking_data[0]["综合得分"]:.4f}），""" + r"""
排名第一，综合表现最优。各评价对象之间的得分差异反映了其在各指标维度上的综合差距。

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/ranking_chart.png}
    \caption{综合评价得分排名图}
    \label{fig:ranking}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\textwidth]{figures/radar_chart.png}
    \caption{Top-5 对象指标雷达图}
    \label{fig:radar}
\end{figure}

\end{document}
"""

    return latex


def save_file_and_return_path(content: str, filename: str) -> Optional[str]:
    """将内容保存为临时文件并返回路径"""
    if not content:
        return None
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


# ╔══════════════════════════════════════════════════════════════╗
# ║                    Gradio 界面构建                            ║
# ╚══════════════════════════════════════════════════════════════╝

def build_gradio_app() -> gr.Blocks:
    """构建 Gradio Blocks 界面"""

    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .main-title {
            text-align: center;
            color: #1f4e79;
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        .result-box {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 16px;
            background: #f9f9f9;
        }
        """,
    ) as app:

        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)

        # ── 全局状态 ──
        state_df = gr.State(value=None)

        with gr.Tabs() as tabs:

            # ═══════════════════════════════════════════
            # Tab 1: 数据上传
            # ═══════════════════════════════════════════
            with gr.TabItem("📊 数据上传", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="上传数据文件 (CSV / Excel)",
                            file_types=[".csv", ".xlsx", ".xls"],
                            type="filepath",
                        )
                        upload_btn = gr.Button("📤 加载数据", variant="primary")

                    with gr.Column(scale=2):
                        data_preview = gr.HTML(
                            label="数据预览",
                            value="<p style='color:#999;'>等待上传数据...</p>",
                        )

                with gr.Row():
                    with gr.Column():
                        col_info = gr.Markdown("", label="列信息")
                    with gr.Column():
                        auto_detect_info = gr.Markdown("", label="自动检测")

                def on_upload(file_obj):
                    df, preview, info, detect = load_and_preview(file_obj)
                    return df, preview, info, detect

                upload_btn.click(
                    fn=on_upload,
                    inputs=[file_input],
                    outputs=[state_df, data_preview, col_info, auto_detect_info],
                )

            # ═══════════════════════════════════════════
            # Tab 2: 参数配置
            # ═══════════════════════════════════════════
            with gr.TabItem("⚙️ 参数配置", id=1):
                gr.Markdown("### 基本配置")

                with gr.Row():
                    object_col_input = gr.Textbox(
                        label="评价对象列名",
                        placeholder="例如: 城市",
                        info="输入代表评价对象名称的列名（留空则自动编号）",
                    )
                    indicator_cols_input = gr.Textbox(
                        label="指标列名（逗号分隔）",
                        placeholder="例如: R&D投入,专利数,高新企业数,GDP增长率",
                        info="输入参与评价的指标列名，用英文逗号分隔",
                    )

                with gr.Row():
                    directions_input = gr.Textbox(
                        label="指标方向（可选，格式: 列名:positive/negative）",
                        placeholder="例如: GDP增长率:positive,能耗:negative",
                        info="默认所有指标为正向(positive)。负向指标用 negative 标注。",
                    )

                gr.Markdown("### 算法选择")

                with gr.Row():
                    norm_method_input = gr.Dropdown(
                        choices=NORM_METHODS,
                        value=NORM_METHODS[0],
                        label="标准化方法",
                    )
                    weight_method_input = gr.Dropdown(
                        choices=WEIGHT_METHODS,
                        value=WEIGHT_METHODS[0],
                        label="赋权方法",
                    )
                    eval_method_input = gr.Dropdown(
                        choices=EVAL_METHODS,
                        value=EVAL_METHODS[0],
                        label="评价模型",
                    )

                gr.Markdown("### 高级选项")

                with gr.Row():
                    enable_pca_input = gr.Checkbox(
                        label="启用 PCA 降维",
                        value=False,
                        info="当指标间存在高相关性时建议启用",
                    )
                    enable_sensitivity_input = gr.Checkbox(
                        label="启用灵敏度分析",
                        value=True,
                        info="检验权重扰动对排名的影响",
                    )
                    language_input = gr.Radio(
                        choices=["中文", "English"],
                        value="中文",
                        label="输出语言",
                    )

                # 数据特征分析按钮
                gr.Markdown("---")
                analyze_btn = gr.Button("🔍 分析数据特征 & 获取推荐", variant="secondary")
                analysis_output = gr.Markdown("")

                analyze_btn.click(
                    fn=lambda df, cols: analyze_data_features(df, cols),
                    inputs=[state_df, indicator_cols_input],
                    outputs=[analysis_output],
                )

            # ═══════════════════════════════════════════
            # Tab 3: 执行评价
            # ═══════════════════════════════════════════
            with gr.TabItem("▶️ 执行评价", id=2):
                run_btn = gr.Button(
                    "🚀 一键执行评价工作流",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("---")

                with gr.Row():
                    result_summary = gr.Markdown("", label="结果摘要")

                with gr.Row():
                    with gr.Column():
                        weights_output = gr.Markdown("", label="权重结果")
                    with gr.Column():
                        ranking_output = gr.Markdown("", label="排名结果")

                run_btn.click(
                    fn=run_evaluation,
                    inputs=[
                        file_input,
                        object_col_input,
                        indicator_cols_input,
                        directions_input,
                        norm_method_input,
                        weight_method_input,
                        eval_method_input,
                        enable_pca_input,
                        enable_sensitivity_input,
                        language_input,
                    ],
                    outputs=[
                        result_summary,
                        weights_output,
                        ranking_output,
                        gr.State(),  # generated_code (传递到下一个tab)
                        gr.State(),  # latex_content
                    ],
                )

            # ═══════════════════════════════════════════
            # Tab 4: 代码与报告
            # ═══════════════════════════════════════════
            with gr.TabItem("📄 代码与报告", id=3):
                gr.Markdown("### 生成的 Python 代码")
                code_output = gr.Code(
                    language="python",
                    label="完整 Python 脚本",
                    lines=30,
                )

                gr.Markdown("### 生成的 LaTeX 报告")
                latex_output = gr.Code(
                    language="latex",
                    label="LaTeX 源码",
                    lines=30,
                )

                with gr.Row():
                    download_code_btn = gr.Button("💾 下载 Python 脚本")
                    download_latex_btn = gr.Button("💾 下载 LaTeX 源码")

                code_file_output = gr.File(label="Python 脚本文件", visible=False)
                latex_file_output = gr.File(label="LaTeX 文件", visible=False)

        # ── 重新绑定执行按钮以更新代码/LaTeX 页 ──
        # 由于 Gradio 的 Tabs 状态管理限制，使用组合输出
        run_btn.click(
            fn=run_evaluation,
            inputs=[
                file_input,
                object_col_input,
                indicator_cols_input,
                directions_input,
                norm_method_input,
                weight_method_input,
                eval_method_input,
                enable_pca_input,
                enable_sensitivity_input,
                language_input,
            ],
            outputs=[
                result_summary,
                weights_output,
                ranking_output,
                code_output,
                latex_output,
            ],
        )

        # 下载按钮逻辑
        def make_code_file(code_text):
            if not code_text:
                return None
            path = save_file_and_return_path(code_text, "auto_eval_script.py")
            return path

        def make_latex_file(latex_text):
            if not latex_text:
                return None
            path = save_file_and_return_path(latex_text, "modeling_report.tex")
            return path

        download_code_btn.click(
            fn=make_code_file,
            inputs=[code_output],
            outputs=[code_file_output],
        )
        download_latex_btn.click(
            fn=make_latex_file,
            inputs=[latex_output],
            outputs=[latex_file_output],
        )

        # ── 页脚 ──
        gr.Markdown("---")
        gr.Markdown(
            f"<center>© 2024 {APP_TITLE} v0.1.0 | "
            f"后端状态: {'✅ 已加载' if BACKEND_AVAILABLE else '⚠️ 演示模式'}</center>"
        )

    return app


# ╔══════════════════════════════════════════════════════════════╗
# ║                        主入口                                ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    """启动 Gradio 应用"""
    app = build_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
    )


if __name__ == "__main__":
    main()