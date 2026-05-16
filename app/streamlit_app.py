"""
AutoEval-Modeling: Streamlit Web 交互界面
==========================================
主入口文件，提供完整的评价类数学建模自动化工作流交互。

启动方式:
    streamlit run app/streamlit_app.py --server.port 8501
"""

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# ── 将项目根目录加入 sys.path，以便导入 src 包 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 延迟导入后端模块（首次运行若后端未完成则使用存根） ──
try:
    from src.utils.config_loader import WorkflowConfig, load_config
    from src.utils.logging_config import setup_logger
    from src.parser.nlp_parser import ProblemParser
    from history.data_profiler import DataProfiler
    from src.core.recommendation import AlgorithmRecommender
    from src.core.workflow import WorkflowEngine
    from src.generators.latex_builder import LatexBuilder
    from src.generators.code_builder import CodeBuilder
    from src.generators.plot_builder import PlotBuilder

    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

# ╔══════════════════════════════════════════════════════════════╗
# ║                     常量 / 配置                              ║
# ╚══════════════════════════════════════════════════════════════╝

APP_TITLE = "AutoEval-Modeling"
APP_SUBTITLE = "面向评价类数学建模的自动化工作流引擎与代码生成系统"
VERSION = "0.1.0"

# 支持的文件格式
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls"]

# 指标方向选项
DIRECTION_OPTIONS = {
    "正向 (越大越好)": "positive",
    "负向 (越小越好)": "negative",
    "适度 (越接近某值越好)": "moderate",
}

# 标准化方法
NORMALIZATION_METHODS = {
    "Min-Max 归一化": "minmax",
    "Z-Score 标准化": "zscore",
    "向量归一化": "vector",
    "最大值归一化": "maxnorm",
}

# 赋权方法
WEIGHT_METHODS = {
    "主观赋权": {
        "层次分析法 (AHP)": "ahp",
        "德尔菲法": "delphi",
        "二项系数法": "binomial",
        "环比评分法": "sequential",
    },
    "客观赋权": {
        "熵权法": "entropy",
        "CRITIC 法": "critic",
        "标准离差法": "stddev",
        "主成分分析 (PCA) 求权重": "pca_weight",
    },
    "组合赋权": {
        "乘法合成": "multiplicative",
        "线性加权 (博弈论)": "game_theory",
        "离差最小化": "min_deviation",
    },
}

# 评价模型
EVAL_METHODS = {
    "TOPSIS (逼近理想解排序)": "topsis",
    "VIKOR (多准则折衷排序)": "vikor",
    "灰色关联分析 (GRA)": "gra",
    "模糊综合评价": "fuzzy_eval",
    "ELECTRE": "electre",
    "秩和比法 (RSR)": "rsr",
    "数据包络分析 (DEA)": "dea",
    "动态时序评价": "dynamic",
}

# 输出语言
OUTPUT_LANGUAGES = {"中文": "zh", "English": "en"}

# 图片格式
IMAGE_FORMATS = {"PNG": "png", "SVG": "svg", "PDF": "pdf"}


# ╔══════════════════════════════════════════════════════════════╗
# ║                   Session State 初始化                       ║
# ╚══════════════════════════════════════════════════════════════╝

def init_session_state():
    """初始化所有 Streamlit session state 变量"""
    defaults = {
        # ── 步骤控制 ──
        "current_step": 0,
        # ── 数据 ──
        "uploaded_file": None,
        "raw_dataframe": None,
        "data_profile": None,
        # ── 问题配置 ──
        "problem_description": "",
        "context_tags": [],
        "object_column": None,
        "indicator_columns": [],
        "indicator_directions": {},
        "moderate_targets": {},  # 适度指标的目标值
        # ── 算法配置 ──
        "normalization_method": "minmax",
        "weight_method_type": "客观赋权",
        "weight_method": "entropy",
        "eval_method": "topsis",
        "enable_pca": False,
        "enable_sensitivity": True,
        "custom_ahp_matrix": None,
        "custom_weights": None,
        # ── 推荐结果 ──
        "recommended_workflows": [],
        "selected_workflow_idx": 0,
        # ── 工作流配置 ──
        "workflow_config": None,
        "workflow_yaml": "",
        # ── 执行结果 ──
        "execution_results": None,
        "execution_log": [],
        "execution_status": "idle",  # idle / running / success / error
        # ── 输出设置 ──
        "output_language": "zh",
        "image_format": "png",
        "latex_template": "mathcup",  # mathcup / mcm / academic
        # ── 生成产物 ──
        "generated_code": "",
        "generated_latex": "",
        "generated_figures": {},
        "generated_tables": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ╔══════════════════════════════════════════════════════════════╗
# ║                    辅助工具函数                               ║
# ╚══════════════════════════════════════════════════════════════╝

def load_dataframe(uploaded_file) -> Optional[pd.DataFrame]:
    """加载上传的数据文件，返回 DataFrame"""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            # 尝试多种编码
            for encoding in ["utf-8", "gbk", "gb2312", "latin1"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    return df
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            st.error("无法解析 CSV 文件编码，请确保为 UTF-8 或 GBK 编码。")
            return None
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
            return df
        else:
            st.error(f"不支持的文件格式: {name}")
            return None
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None


def auto_detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
    """
    自动检测评价对象列和指标列。
    规则：第一列若为字符串类型则作为对象列，其余数值列作为指标列。
    """
    object_col = None
    indicator_cols = []

    if df.shape[1] < 2:
        return None, []

    # 检测对象列
    first_col = df.columns[0]
    if df[first_col].dtype == object or df[first_col].dtype.name == "string":
        object_col = first_col

    # 检测指标列
    for col in df.columns:
        if col == object_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            indicator_cols.append(col)

    return object_col, indicator_cols


def compute_basic_profile(df: pd.DataFrame, indicator_cols: List[str]) -> Dict:
    """计算数据的基本统计特征（在后端模块未就绪时使用的存根版本）"""
    profile = {
        "n_objects": len(df),
        "n_indicators": len(indicator_cols),
        "missing_rate": {},
        "correlation_matrix": None,
        "high_correlation_pairs": [],
        "basic_stats": None,
    }

    if not indicator_cols:
        return profile

    sub = df[indicator_cols]

    # 缺失率
    profile["missing_rate"] = (sub.isnull().sum() / len(sub)).to_dict()

    # 基本统计
    profile["basic_stats"] = sub.describe().round(4)

    # 相关性
    try:
        corr = sub.corr()
        profile["correlation_matrix"] = corr

        # 高相关性指标对
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = abs(corr.iloc[i, j])
                if val > 0.8:
                    high_corr.append(
                        (corr.columns[i], corr.columns[j], round(val, 4))
                    )
        profile["high_correlation_pairs"] = high_corr
    except Exception:
        pass

    return profile


def generate_default_workflow_yaml(config: Dict) -> str:
    """根据当前配置生成 YAML 工作流描述"""
    import yaml

    workflow = {
        "name": "auto_generated_workflow",
        "version": "1.0",
        "description": config.get("problem_description", "评价类问题自动工作流"),
        "data": {
            "file": config.get("data_file", "data.csv"),
            "object_column": config.get("object_column", ""),
            "indicator_columns": config.get("indicator_columns", []),
            "indicator_directions": config.get("indicator_directions", {}),
            "moderate_targets": config.get("moderate_targets", {}),
        },
        "pipeline": [],
        "output": {
            "language": config.get("output_language", "zh"),
            "image_format": config.get("image_format", "png"),
            "latex_template": config.get("latex_template", "mathcup"),
            "output_dir": config.get("output_dir", "output/"),
        },
    }

    # 构建流水线节点
    steps = []

    # Step 1: 数据预处理
    steps.append({
        "id": "preprocessing",
        "type": "preprocess",
        "params": {
            "handle_missing": "mean",
            "normalization": config.get("normalization_method", "minmax"),
        },
    })

    # Step 1.5: PCA 降维（可选）
    if config.get("enable_pca", False):
        steps.append({
            "id": "pca_reduction",
            "type": "reduction",
            "method": "pca",
            "params": {"variance_threshold": 0.85},
            "depends_on": ["preprocessing"],
        })

    # Step 2: 赋权
    weight_depends = "pca_reduction" if config.get("enable_pca", False) else "preprocessing"
    steps.append({
        "id": "weight_calculation",
        "type": "weight",
        "method": config.get("weight_method", "entropy"),
        "params": {},
        "depends_on": [weight_depends],
    })

    # Step 3: 综合评价
    steps.append({
        "id": "evaluation",
        "type": "evaluation",
        "method": config.get("eval_method", "topsis"),
        "params": {},
        "depends_on": ["weight_calculation"],
    })

    # Step 4: 灵敏度分析（可选）
    if config.get("enable_sensitivity", True):
        steps.append({
            "id": "sensitivity_analysis",
            "type": "sensitivity",
            "method": "oat",
            "params": {"perturbation_range": 0.1},
            "depends_on": ["evaluation"],
        })

    # Step 5: 结果汇总
    final_depends = ["evaluation"]
    if config.get("enable_sensitivity", True):
        final_depends.append("sensitivity_analysis")

    steps.append({
        "id": "result_consolidation",
        "type": "consolidation",
        "depends_on": final_depends,
    })

    workflow["pipeline"] = steps

    return yaml.dump(workflow, allow_unicode=True, default_flow_style=False, sort_keys=False)


def mock_execute_workflow(config: Dict, df: pd.DataFrame) -> Dict:
    """
    模拟工作流执行（在后端引擎未完成时使用）。
    返回结构化的执行结果供展示。
    """
    results = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "steps": [],
        "weights": {},
        "scores": {},
        "ranking": [],
        "figures": {},
        "tables": {},
        "latex_sections": {},
        "generated_code": "",
    }

    indicator_cols = config.get("indicator_columns", [])
    object_col = config.get("object_column", "")
    objects = df[object_col].tolist() if object_col and object_col in df.columns else [f"对象{i + 1}" for i in
                                                                                       range(len(df))]

    n_indicators = len(indicator_cols)
    if n_indicators == 0:
        results["status"] = "error"
        results["error_message"] = "未选择任何指标列"
        return results

    # ── 模拟 Step 1: 预处理 ──
    results["steps"].append({
        "id": "preprocessing",
        "status": "completed",
        "message": f"数据预处理完成：{len(df)}个样本，{n_indicators}个指标，"
                   f"标准化方法：{config.get('normalization_method', 'minmax')}",
    })

    # ── 模拟 Step 2: 权重计算 ──
    np.random.seed(42)
    raw_weights = np.random.dirichlet(np.ones(n_indicators))
    weights = {col: round(w, 4) for col, w in zip(indicator_cols, raw_weights)}
    results["weights"] = weights
    results["steps"].append({
        "id": "weight_calculation",
        "status": "completed",
        "message": f"权重计算完成，方法：{config.get('weight_method', 'entropy')}",
    })

    # ── 模拟 Step 3: 综合评价 ──
    np.random.seed(123)
    scores_raw = np.random.uniform(0.3, 0.95, len(objects))
    sorted_idx = np.argsort(-scores_raw)
    ranking = [(objects[i], round(scores_raw[i], 4), rank + 1)
               for rank, i in enumerate(sorted_idx)]

    results["scores"] = {obj: round(s, 4) for obj, s in zip(objects, scores_raw)}
    results["ranking"] = ranking
    results["steps"].append({
        "id": "evaluation",
        "status": "completed",
        "message": f"综合评价完成，方法：{config.get('eval_method', 'topsis')}",
    })

    # ── 模拟 Step 4: 灵敏度分析 ──
    if config.get("enable_sensitivity", True):
        results["steps"].append({
            "id": "sensitivity_analysis",
            "status": "completed",
            "message": "灵敏度分析完成（权重 ±10%），排名稳健性良好。",
        })

    # ── 生成模拟代码 ──
    results["generated_code"] = _generate_mock_code(config, indicator_cols, objects)

    # ── 生成模拟 LaTeX ──
    results["latex_sections"] = _generate_mock_latex(config, weights, ranking)

    return results


def _generate_mock_code(config: Dict, indicators: List[str], objects: List[str]) -> str:
    """生成示例 Python 代码"""
    weight_method = config.get("weight_method", "entropy")
    eval_method = config.get("eval_method", "topsis")
    norm_method = config.get("normalization_method", "minmax")

    code = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoEval-Modeling 自动生成脚本
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
评价方法: {weight_method} + {eval_method}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ═══════════════════════════════════════════════
# 1. 数据加载
# ═══════════════════════════════════════════════
df = pd.read_csv("{config.get("data_file", "data.csv")}")
print("数据维度:", df.shape)
print(df.head())

object_col = "{config.get("object_column", "")}"
indicator_cols = {indicators}
directions = {config.get("indicator_directions", {})}

X = df[indicator_cols].values.astype(float)
objects = df[object_col].tolist() if object_col else [f"对象{{i+1}}" for i in range(len(df))]

# ═══════════════════════════════════════════════
# 2. 数据预处理 - {norm_method} 标准化
# ═══════════════════════════════════════════════
def normalize_{norm_method}(X, directions):
    """对决策矩阵进行标准化处理"""
    X_norm = X.copy().astype(float)
    n, m = X_norm.shape
    for j in range(m):
        col_name = indicator_cols[j]
        direction = directions.get(col_name, "positive")
        col = X_norm[:, j]
        if direction == "negative":
            col = -col  # 先取反再归一化
'''

    if norm_method == "minmax":
        code += '''        col_min, col_max = col.min(), col.max()
        if col_max - col_min == 0:
            X_norm[:, j] = 0.5
        else:
            X_norm[:, j] = (col - col_min) / (col_max - col_min)
'''
    elif norm_method == "zscore":
        code += '''        col_mean, col_std = col.mean(), col.std()
        if col_std == 0:
            X_norm[:, j] = 0
        else:
            X_norm[:, j] = (col - col_mean) / col_std
'''

    code += f'''    return X_norm

X_norm = normalize_{norm_method}(X, directions)
print("\\n标准化后矩阵:")
print(pd.DataFrame(X_norm, columns=indicator_cols, index=objects).round(4))

# ═══════════════════════════════════════════════
# 3. 权重计算 - {weight_method}
# ═══════════════════════════════════════════════
'''

    if weight_method == "entropy":
        code += '''def entropy_weight(X_norm):
    """熵权法计算指标权重"""
    n, m = X_norm.shape
    # 归一化为比重矩阵
    P = X_norm / X_norm.sum(axis=0, keepdims=True)
    P = np.clip(P, 1e-12, None)  # 避免 log(0)

    # 计算信息熵
    k = 1.0 / np.log(n)
    E = -k * np.sum(P * np.log(P), axis=0)

    # 计算权重
    D = 1 - E  # 差异系数
    W = D / D.sum()

    return W, E, D

weights, entropy_values, diff_coefficients = entropy_weight(X_norm)
print("\\n熵权法结果:")
for col, w in zip(indicator_cols, weights):
    print(f"  {{col}}: {{w:.4f}}")
'''
    elif weight_method == "critic":
        code += '''def critic_weight(X_norm):
    """CRITIC法计算指标权重"""
    n, m = X_norm.shape
    std = X_norm.std(axis=0)
    corr = np.corrcoef(X_norm.T)
    conflict = np.sum(1 - corr, axis=1)
    info = std * conflict
    weights = info / info.sum()
    return weights, std, conflict

weights, std_values, conflict_values = critic_weight(X_norm)
print("\\nCRITIC法结果:")
for col, w in zip(indicator_cols, weights):
    print(f"  {{col}}: {{w:.4f}}")
'''

    if eval_method == "topsis":
        code += '''
# ═══════════════════════════════════════════════
# 4. TOPSIS 综合评价
# ═══════════════════════════════════════════════
def topsis(X_norm, weights):
    """TOPSIS法计算综合得分与排名"""
    n, m = X_norm.shape

    # 加权标准化矩阵
    Z = X_norm * weights

    # 正理想解与负理想解
    Z_pos = Z.max(axis=0)
    Z_neg = Z.min(axis=0)

    # 计算距离
    D_pos = np.sqrt(np.sum((Z - Z_pos) ** 2, axis=1))
    D_neg = np.sqrt(np.sum((Z - Z_neg) ** 2, axis=1))

    # 相对贴近度
    C = D_neg / (D_pos + D_neg + 1e-12)

    return C, D_pos, D_neg

scores, d_pos, d_neg = topsis(X_norm, weights)

# 排名
ranking = np.argsort(-scores) + 1
result_df = pd.DataFrame({{
    "评价对象": objects,
    "综合得分": np.round(scores, 4),
    "排名": ranking,
}}).sort_values("排名")

print("\\nTOPSIS 评价结果:")
print(result_df.to_string(index=False))

# ═══════════════════════════════════════════════
# 5. 结果可视化
# ═══════════════════════════════════════════════
# 5.1 权重条形图
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.barh(indicator_cols, weights, color="steelblue")
ax1.set_xlabel("权重")
ax1.set_title("指标权重分布")
plt.tight_layout()
fig1.savefig("figures/weight_bar.png", dpi=300)
plt.show()

# 5.2 综合得分排序图
sorted_df = result_df.sort_values("综合得分", ascending=True)
fig2, ax2 = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_df)))
ax2.barh(sorted_df["评价对象"], sorted_df["综合得分"], color=colors)
ax2.set_xlabel("综合得分")
ax2.set_title("TOPSIS 综合评价排名")
plt.tight_layout()
fig2.savefig("figures/topsis_ranking.png", dpi=300)
plt.show()

# 5.3 雷达图
from matplotlib.patches import FancyBboxPatch
angles = np.linspace(0, 2 * np.pi, len(indicator_cols), endpoint=False).tolist()
angles += angles[:1]

fig3, ax3 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, obj in enumerate(objects[:5]):  # 最多展示5个对象
    values = X_norm[i].tolist()
    values += values[:1]
    ax3.plot(angles, values, "o-", linewidth=1.5, label=obj)
    ax3.fill(angles, values, alpha=0.1)
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(indicator_cols)
ax3.set_title("指标雷达图")
ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
fig3.savefig("figures/radar_chart.png", dpi=300)
plt.show()

print("\\n所有结果已保存至 figures/ 目录。")
'''

    return code


def _generate_mock_latex(config: Dict, weights: Dict, ranking: List) -> Dict:
    """生成示例 LaTeX 段落"""
    weight_method = config.get("weight_method", "entropy")
    eval_method = config.get("eval_method", "topsis")

    sections = {}

    # ── 模型建立 ──
    sections["model_setup"] = r"""\subsection{评价模型建立}

本文针对所给评价问题，采用""" + f"**{weight_method}**赋权 + **{eval_method}**" + r"""综合评价的方法，
构建多属性决策评价模型。具体思路如下：

\begin{enumerate}
    \item 首先对原始数据进行预处理与标准化，消除量纲影响；
    \item 利用""" + weight_method + r"""确定各指标的客观权重；
    \item 基于""" + eval_method + r"""计算各评价对象的综合得分与排名；
    \item 最后通过灵敏度分析检验结果的稳健性。
\end{enumerate}
"""

    # ── 权重表格 ──
    weight_rows = "\n".join(
        [f"        {col} & {w} \\\\" for col, w in weights.items()]
    )
    sections["weight_table"] = r"""\begin{table}[htbp]
    \centering
    \caption{指标权重计算结果}
    \label{tab:weights}
    \begin{tabular}{lc}
        \toprule
        指标 & 权重 \\
        \midrule
""" + weight_rows + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""

    # ── 排名表格 ──
    rank_rows = "\n".join(
        [f"        {obj} & {score} & {rank} \\\\" for obj, score, rank in ranking]
    )
    sections["ranking_table"] = r"""\begin{table}[htbp]
    \centering
    \caption{综合评价得分与排名}
    \label{tab:ranking}
    \begin{tabular}{lcc}
        \toprule
        评价对象 & 综合得分 & 排名 \\
        \midrule
""" + rank_rows + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""

    return sections


# ╔══════════════════════════════════════════════════════════════╗
# ║                     页面组件                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def render_sidebar():
    """渲染侧边栏导航与全局设置"""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/combo-chart.png",
            width=64,
        )
        st.title(APP_TITLE)
        st.caption(f"v{VERSION}")
        st.markdown("---")

        # 步骤导航
        steps = [
            "📋 问题定义",
            "📊 数据上传",
            "⚙️ 算法配置",
            "🔄 工作流预览",
            "▶️ 执行与结果",
            "📥 导出下载",
        ]

        st.subheader("🧭 操作步骤")
        for i, step_name in enumerate(steps):
            if i == st.session_state.current_step:
                st.markdown(f"**→ {step_name}**")
            else:
                if st.button(step_name, key=f"nav_{i}", use_container_width=True):
                    st.session_state.current_step = i
                    st.rerun()

        st.markdown("---")

        # 全局设置
        st.subheader("🌐 全局设置")
        lang = st.selectbox(
            "输出语言",
            list(OUTPUT_LANGUAGES.keys()),
            index=0,
            key="sidebar_lang",
        )
        st.session_state.output_language = OUTPUT_LANGUAGES[lang]

        img_fmt = st.selectbox(
            "图片格式",
            list(IMAGE_FORMATS.keys()),
            index=0,
            key="sidebar_img_fmt",
        )
        st.session_state.image_format = IMAGE_FORMATS[img_fmt]

        latex_tpl = st.selectbox(
            "LaTeX 模板",
            ["数学建模国赛 (mathcup)", "美赛 (mcm)", "通用学术 (academic)"],
            index=0,
            key="sidebar_latex_tpl",
        )
        st.session_state.latex_template = latex_tpl.split("(")[1].rstrip(")")

        st.markdown("---")

        # 后端状态
        if BACKEND_AVAILABLE:
            st.success("✅ 后端引擎已加载")
        else:
            st.warning("⚠️ 后端引擎未就绪，使用演示模式")

        st.caption("© 2024 AutoEval-Modeling")


def render_step0_problem_definition():
    """Step 0: 问题定义"""
    st.header("📋 Step 1: 问题定义")
    st.markdown("描述您的评价问题，系统将自动解析关键信息并推荐最佳算法组合。")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.session_state.problem_description = st.text_area(
            "问题描述",
            value=st.session_state.problem_description,
            height=150,
            placeholder="例如：请对10个城市的科技创新能力进行综合评价与排名，数据包含R&D投入、专利授权数、高新技术企业数等指标...",
            help="描述越详细，系统推荐越精准。",
        )

    with col2:
        st.markdown("##### 快速选择场景")
        preset_scenes = {
            "城市/区域综合评价": "对多个城市/区域基于多项指标进行综合实力排名评价",
            "企业绩效评估": "对多个企业的经营绩效进行评估与排序",
            "方案优选": "从多个备选方案中选出最优方案",
            "风险评估": "对多个对象的风险等级进行评价分类",
            "动态趋势评价": "基于多年数据对评价对象的发展趋势进行动态评价",
        }
        for label, desc in preset_scenes.items():
            if st.button(label, key=f"scene_{label}", use_container_width=True):
                st.session_state.problem_description = desc
                st.rerun()

    st.markdown("---")

    # 情境解析结果
    if st.session_state.problem_description:
        with st.expander("🔍 情境解析结果", expanded=True):
            # 简单关键词匹配（后端完成后替换为 NLP 解析）
            desc = st.session_state.problem_description
            tags = []
            if any(kw in desc for kw in ["排名", "排序", "综合评价", "综合实力"]):
                tags.append("multi_attribute_ranking")
            if any(kw in desc for kw in ["风险", "等级", "分级", "分类"]):
                tags.append("risk_classification")
            if any(kw in desc for kw in ["方案", "优选", "选择"]):
                tags.append("alternative_selection")
            if any(kw in desc for kw in ["动态", "趋势", "年度", "时序", "多年"]):
                tags.append("dynamic_evaluation")
            if any(kw in desc for kw in ["企业", "绩效", "经营"]):
                tags.append("performance_benchmarking")
            if not tags:
                tags.append("general_evaluation")

            st.session_state.context_tags = tags
            tag_labels = {
                "multi_attribute_ranking": "📊 多属性排序评价",
                "risk_classification": "⚠️ 风险等级分类",
                "alternative_selection": "🎯 方案优选",
                "dynamic_evaluation": "📈 动态时序评价",
                "performance_benchmarking": "🏢 绩效基准比较",
                "general_evaluation": "📋 通用综合评价",
            }
            st.write("**识别的情境标签：**")
            for tag in tags:
                st.info(tag_labels.get(tag, tag))

    # 导航按钮
    st.markdown("---")
    col_prev, col_spacer, col_next = st.columns([1, 4, 1])
    with col_next:
        if st.button("下一步 ➡️", use_container_width=True, type="primary"):
            if not st.session_state.problem_description:
                st.warning("请先输入问题描述。")
            else:
                st.session_state.current_step = 1
                st.rerun()


def render_step1_data_upload():
    """Step 1: 数据上传与预览"""
    st.header("📊 Step 2: 数据上传与预览")

    uploaded = st.file_uploader(
        "上传数据文件",
        type=SUPPORTED_FILE_TYPES,
        help="支持 CSV、XLSX、XLS 格式",
    )

    if uploaded is not None:
        st.session_state.uploaded_file = uploaded
        df = load_dataframe(uploaded)
        if df is not None:
            st.session_state.raw_dataframe = df

    if st.session_state.raw_dataframe is not None:
        df = st.session_state.raw_dataframe

        # 数据预览
        st.subheader("📋 数据预览")
        st.dataframe(df.head(20), use_container_width=True, height=400)
        st.caption(f"共 {df.shape[0]} 行 × {df.shape[1]} 列")

        st.markdown("---")

        # 列配置
        st.subheader("🏷️ 列角色配置")

        auto_obj, auto_inds = auto_detect_columns(df)

        col1, col2 = st.columns(2)
        with col1:
            obj_options = ["(无)"] + list(df.columns)
            default_obj_idx = obj_options.index(auto_obj) if auto_obj in obj_options else 0
            st.session_state.object_column = st.selectbox(
                "评价对象列",
                obj_options,
                index=default_obj_idx,
                help="选择代表评价对象名称的列（如城市名、企业名）",
            )
            if st.session_state.object_column == "(无)":
                st.session_state.object_column = None

        with col2:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            default_inds = [c for c in auto_inds if c in numeric_cols]
            st.session_state.indicator_columns = st.multiselect(
                "指标列（可多选）",
                numeric_cols,
                default=default_inds,
                help="选择参与评价的数值型指标列",
            )

        # 指标方向设置
        if st.session_state.indicator_columns:
            st.markdown("---")
            st.subheader("🧭 指标方向设置")
            st.caption("设置每个指标的优化方向")

            dirs = {}
            mod_targets = {}
            n_cols = 3
            cols = st.columns(n_cols)
            for idx, col_name in enumerate(st.session_state.indicator_columns):
                with cols[idx % n_cols]:
                    d = st.selectbox(
                        f"{col_name}",
                        list(DIRECTION_OPTIONS.keys()),
                        index=0,
                        key=f"dir_{col_name}",
                    )
                    dirs[col_name] = DIRECTION_OPTIONS[d]
                    if DIRECTION_OPTIONS[d] == "moderate":
                        target = st.number_input(
                            f"{col_name} 目标值",
                            value=float(df[col_name].mean()),
                            key=f"target_{col_name}",
                        )
                        mod_targets[col_name] = target

            st.session_state.indicator_directions = dirs
            st.session_state.moderate_targets = mod_targets

        # 数据概况
        if st.session_state.indicator_columns:
            st.markdown("---")
            st.subheader("📈 数据特征概况")

            profile = compute_basic_profile(df, st.session_state.indicator_columns)
            st.session_state.data_profile = profile

            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("评价对象数", profile["n_objects"])
            with info_col2:
                st.metric("指标数量", profile["n_indicators"])
            with info_col3:
                max_missing = max(profile["missing_rate"].values()) if profile["missing_rate"] else 0
                st.metric("最大缺失率", f"{max_missing:.1%}")

            # 基本统计
            if profile["basic_stats"] is not None:
                with st.expander("📊 描述性统计", expanded=False):
                    st.dataframe(profile["basic_stats"], use_container_width=True)

            # 相关性矩阵
            if profile["correlation_matrix"] is not None:
                with st.expander("🔗 指标相关性矩阵", expanded=False):
                    import matplotlib.pyplot as plt
                    import matplotlib
                    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
                    matplotlib.rcParams["axes.unicode_minus"] = False

                    fig, ax = plt.subplots(figsize=(10, 8))
                    corr = profile["correlation_matrix"]
                    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_yticks(range(len(corr.columns)))
                    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                    ax.set_yticklabels(corr.columns)
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    ax.set_title("指标相关性热力图")
                    st.pyplot(fig)

                    if profile["high_correlation_pairs"]:
                        st.warning("⚠️ 发现高相关性指标对（|r| > 0.8）：")
                        for c1, c2, r in profile["high_correlation_pairs"]:
                            st.write(f"  - {c1} ↔ {c2}：r = {r}")
                        st.info("💡 建议：可启用 PCA 降维以消除多重共线性。")

    # 导航按钮
    st.markdown("---")
    col_prev, col_spacer, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("⬅️ 上一步", use_container_width=True):
            st.session_state.current_step = 0
            st.rerun()
    with col_next:
        if st.button("下一步 ➡️", use_container_width=True, type="primary"):
            if st.session_state.raw_dataframe is None:
                st.warning("请先上传数据文件。")
            elif not st.session_state.indicator_columns:
                st.warning("请至少选择一个指标列。")
            else:
                st.session_state.current_step = 2
                st.rerun()


def render_step2_algorithm_config():
    """Step 2: 算法配置"""
    st.header("⚙️ Step 3: 算法配置")

    # 智能推荐区
    st.subheader("🤖 算法智能推荐")
    profile = st.session_state.data_profile
    tags = st.session_state.context_tags

    if profile:
        recommendations = []
        reasons = []

        # 基于数据特征推荐
        has_high_corr = bool(profile.get("high_correlation_pairs", []))
        n_obj = profile.get("n_objects", 0)
        n_ind = profile.get("n_indicators", 0)
        ratio = n_ind / max(n_obj, 1)

        if has_high_corr:
            recommendations.append("PCA降维 → CRITIC → TOPSIS")
            reasons.append("指标间高相关性，建议先 PCA 降维消除共线性")
        if ratio > 0.5:
            recommendations.append("熵权法 → 灰色关联分析 (GRA)")
            reasons.append("指标多样本少，灰色关联对数据量要求低")
        if "dynamic_evaluation" in tags:
            recommendations.append("熵权法 → TOPSIS（按年分列）→ 加权汇总")
            reasons.append("动态评价场景，需保留时序信息")
        if "risk_classification" in tags:
            recommendations.append("AHP（主观）→ 模糊综合评价")
            reasons.append("风险分级适合模糊综合评价处理等级模糊性")

        # 默认推荐
        if not recommendations:
            recommendations.append("熵权法 → TOPSIS")
            reasons.append("通用性强，客观权重 + 理想解排序")

        for i, (rec, reason) in enumerate(zip(recommendations, reasons)):
            with st.container():
                col_rec, col_btn = st.columns([4, 1])
                with col_rec:
                    st.info(f"**推荐方案 {i + 1}：** {rec}\n\n_理由：{reason}_")
                with col_btn:
                    if st.button("采用", key=f"adopt_{i}", use_container_width=True):
                        # 解析推荐方案并自动配置
                        if "PCA" in rec:
                            st.session_state.enable_pca = True
                        if "CRITIC" in rec:
                            st.session_state.weight_method = "critic"
                            st.session_state.weight_method_type = "客观赋权"
                        elif "熵权" in rec:
                            st.session_state.weight_method = "entropy"
                            st.session_state.weight_method_type = "客观赋权"
                        elif "AHP" in rec:
                            st.session_state.weight_method = "ahp"
                            st.session_state.weight_method_type = "主观赋权"
                        if "TOPSIS" in rec:
                            st.session_state.eval_method = "topsis"
                        elif "GRA" in rec or "灰色关联" in rec:
                            st.session_state.eval_method = "gra"
                        elif "模糊" in rec:
                            st.session_state.eval_method = "fuzzy_eval"
                        st.rerun()

    st.markdown("---")

    # 手动配置区
    st.subheader("🛠️ 详细配置")

    tab_preprocess, tab_weight, tab_eval, tab_extra = st.tabs(
        ["📐 预处理", "⚖️ 赋权方法", "📊 评价模型", "🔧 高级选项"]
    )

    with tab_preprocess:
        col1, col2 = st.columns(2)
        with col1:
            norm_method = st.selectbox(
                "标准化方法",
                list(NORMALIZATION_METHODS.keys()),
                index=0,
            )
            st.session_state.normalization_method = NORMALIZATION_METHODS[norm_method]
        with col2:
            st.session_state.enable_pca = st.checkbox(
                "启用 PCA 降维",
                value=st.session_state.enable_pca,
                help="当指标间存在较强相关性时建议启用",
            )
            if st.session_state.enable_pca:
                pca_threshold = st.slider(
                    "PCA 累计方差贡献率阈值",
                    min_value=0.7,
                    max_value=0.99,
                    value=0.85,
                    step=0.01,
                )

    with tab_weight:
        weight_type = st.radio(
            "赋权类型",
            list(WEIGHT_METHODS.keys()),
            horizontal=True,
            index=list(WEIGHT_METHODS.keys()).index(st.session_state.weight_method_type),
        )
        st.session_state.weight_method_type = weight_type

        method_options = WEIGHT_METHODS[weight_type]
        current_method_name = None
        for name, code in method_options.items():
            if code == st.session_state.weight_method:
                current_method_name = name
                break
        if current_method_name is None:
            current_method_name = list(method_options.keys())[0]
            st.session_state.weight_method = method_options[current_method_name]

        selected_method = st.selectbox(
            "具体方法",
            list(method_options.keys()),
            index=list(method_options.keys()).index(current_method_name),
        )
        st.session_state.weight_method = method_options[selected_method]

        # AHP 判断矩阵输入
        if st.session_state.weight_method == "ahp" and st.session_state.indicator_columns:
            st.markdown("---")
            st.markdown("##### AHP 判断矩阵")
            st.caption("请输入上三角元素（1-9标度），下三角将自动计算。对角线为1。")

            n = len(st.session_state.indicator_columns)
            cols_names = st.session_state.indicator_columns

            ahp_matrix = np.ones((n, n))
            for i in range(n):
                cols_row = st.columns(n)
                for j in range(n):
                    with cols_row[j]:
                        if i == j:
                            st.text_input(
                                f"{cols_names[i][:4]}↔{cols_names[j][:4]}",
                                value="1",
                                disabled=True,
                                key=f"ahp_{i}_{j}",
                            )
                        elif i < j:
                            val = st.number_input(
                                f"{cols_names[i][:4]}↔{cols_names[j][:4]}",
                                min_value=0.111,
                                max_value=9.0,
                                value=1.0,
                                step=0.5,
                                key=f"ahp_{i}_{j}",
                            )
                            ahp_matrix[i, j] = val
                            ahp_matrix[j, i] = 1.0 / val
                        else:
                            st.text_input(
                                f"{cols_names[i][:4]}↔{cols_names[j][:4]}",
                                value=f"{1.0 / ahp_matrix[j, i]:.3f}",
                                disabled=True,
                                key=f"ahp_{i}_{j}",
                            )

            st.session_state.custom_ahp_matrix = ahp_matrix

        # 自定义权重
        if weight_type == "主观赋权" and st.session_state.weight_method != "ahp":
            st.markdown("---")
            st.markdown("##### 自定义权重向量")
            custom_w = {}
            cols = st.columns(min(len(st.session_state.indicator_columns), 4))
            for idx, col_name in enumerate(st.session_state.indicator_columns):
                with cols[idx % len(cols)]:
                    w = st.number_input(
                        col_name,
                        min_value=0.0,
                        max_value=1.0,
                        value=round(1.0 / max(len(st.session_state.indicator_columns), 1), 4),
                        step=0.01,
                        key=f"custom_w_{col_name}",
                    )
                    custom_w[col_name] = w
            total_w = sum(custom_w.values())
            if abs(total_w - 1.0) > 0.01:
                st.warning(f"权重之和为 {total_w:.4f}，应归一化为 1。")
            st.session_state.custom_weights = custom_w

    with tab_eval:
        eval_method_name = st.selectbox(
            "综合评价模型",
            list(EVAL_METHODS.keys()),
            index=list(EVAL_METHODS.values()).index(st.session_state.eval_method)
            if st.session_state.eval_method in EVAL_METHODS.values()
            else 0,
        )
        st.session_state.eval_method = EVAL_METHODS[eval_method_name]

        # 模型特定参数
        if st.session_state.eval_method == "vikor":
            st.slider("VIKOR 折衷系数 ν", 0.0, 1.0, 0.5, 0.05, key="vikor_v")
        elif st.session_state.eval_method == "gra":
            st.slider("灰色关联分辨系数 ρ", 0.0, 1.0, 0.5, 0.05, key="gra_rho")
        elif st.session_state.eval_method == "fuzzy_eval":
            st.selectbox("模糊算子类型", ["加权平均", "最大最小", "加权取大"], key="fuzzy_op")
        elif st.session_state.eval_method == "dea":
            st.selectbox("DEA 模型类型", ["CCR", "BCC", "Super-Efficiency"], key="dea_type")

    with tab_extra:
        st.session_state.enable_sensitivity = st.checkbox(
            "启用灵敏度分析",
            value=st.session_state.enable_sensitivity,
            help="对权重施加 ±10% 扰动，观察排名变动",
        )

        st.checkbox(
            "生成排名一致性检验 (Kendall τ)",
            value=True,
            key="enable_kendall",
            help="比较不同方法的排名一致性",
        )

        st.selectbox(
            "缺失值处理策略",
            ["均值填充", "中位数填充", "线性插值", "删除含缺失行"],
            index=0,
            key="missing_strategy",
        )

        st.selectbox(
            "异常值处理策略",
            ["不处理", "3σ 原则截断", "IQR 方法截断", "Winsorize (1%-99%)"],
            index=0,
            key="outlier_strategy",
        )

    # 导航按钮
    st.markdown("---")
    col_prev, col_spacer, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("⬅️ 上一步", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    with col_next:
        if st.button("下一步 ➡️", use_container_width=True, type="primary"):
            st.session_state.current_step = 3
            st.rerun()


def render_step3_workflow_preview():
    """Step 3: 工作流预览"""
    st.header("🔄 Step 4: 工作流预览")

    # 构建配置字典
    config = {
        "problem_description": st.session_state.problem_description,
        "data_file": st.session_state.uploaded_file.name if st.session_state.uploaded_file else "data.csv",
        "object_column": st.session_state.object_column,
        "indicator_columns": st.session_state.indicator_columns,
        "indicator_directions": st.session_state.indicator_directions,
        "moderate_targets": st.session_state.moderate_targets,
        "normalization_method": st.session_state.normalization_method,
        "weight_method": st.session_state.weight_method,
        "eval_method": st.session_state.eval_method,
        "enable_pca": st.session_state.enable_pca,
        "enable_sensitivity": st.session_state.enable_sensitivity,
        "output_language": st.session_state.output_language,
        "image_format": st.session_state.image_format,
        "latex_template": st.session_state.latex_template,
    }

    # 生成 YAML
    yaml_text = generate_default_workflow_yaml(config)
    st.session_state.workflow_yaml = yaml_text
    st.session_state.workflow_config = config

    # 可视化流水线
    st.subheader("📐 工作流 DAG 可视化")

    # 简易流程图（ASCII/Markdown）
    pipeline_steps = [
        ("📥 数据加载", "加载并解析原始数据"),
        ("🔧 数据预处理", f"标准化: {st.session_state.normalization_method}"),
    ]
    if st.session_state.enable_pca:
        pipeline_steps.append(("📉 PCA 降维", "消除多重共线性"))
    pipeline_steps.append(("⚖️ 权重计算", f"方法: {st.session_state.weight_method}"))
    pipeline_steps.append(("📊 综合评价", f"模型: {st.session_state.eval_method}"))
    if st.session_state.enable_sensitivity:
        pipeline_steps.append(("📈 灵敏度分析", "权重 ±10% 扰动检验"))
    pipeline_steps.append(("📋 结果汇总", "生成报告与图表"))

    for i, (name, desc) in enumerate(pipeline_steps):
        col_icon, col_content = st.columns([1, 5])
        with col_icon:
            st.markdown(f"### {'│' if i > 0 else ''}")
            st.markdown(f"### ● Step {i + 1}")
        with col_content:
            st.markdown(f"**{name}**")
            st.caption(desc)
        if i < len(pipeline_steps) - 1:
            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**↓**")

    # YAML 编辑
    st.markdown("---")
    st.subheader("📝 工作流配置文件 (YAML)")
    edited_yaml = st.text_area(
        "您可以手动编辑工作流配置",
        value=yaml_text,
        height=400,
        key="yaml_editor",
    )
    st.session_state.workflow_yaml = edited_yaml

    # 配置摘要
    with st.expander("📋 配置摘要", expanded=True):
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.markdown("**数据信息**")
            st.write(f"- 评价对象列: `{st.session_state.object_column}`")
            st.write(f"- 指标数量: {len(st.session_state.indicator_columns)}")
            st.write(
                f"- 样本数: {len(st.session_state.raw_dataframe) if st.session_state.raw_dataframe is not None else 'N/A'}")
        with summary_cols[1]:
            st.markdown("**算法选择**")
            st.write(f"- 标准化: {st.session_state.normalization_method}")
            st.write(f"- 赋权法: {st.session_state.weight_method}")
            st.write(f"- 评价模型: {st.session_state.eval_method}")
            st.write(f"- PCA 降维: {'✅' if st.session_state.enable_pca else '❌'}")
        with summary_cols[2]:
            st.markdown("**输出设置**")
            st.write(f"- 语言: {st.session_state.output_language}")
            st.write(f"- 图片格式: {st.session_state.image_format}")
            st.write(f"- LaTeX 模板: {st.session_state.latex_template}")
            st.write(f"- 灵敏度分析: {'✅' if st.session_state.enable_sensitivity else '❌'}")

    # 导航按钮
    st.markdown("---")
    col_prev, col_spacer, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("⬅️ 上一步", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
    with col_next:
        if st.button("🚀 开始执行", use_container_width=True, type="primary"):
            st.session_state.current_step = 4
            st.rerun()


def render_step4_execution():
    """Step 4: 执行与结果展示"""
    st.header("▶️ Step 5: 执行与结果")

    config = st.session_state.workflow_config
    df = st.session_state.raw_dataframe

    if config is None or df is None:
        st.warning("请先完成前置步骤的配置。")
        if st.button("⬅️ 返回配置"):
            st.session_state.current_step = 3
            st.rerun()
        return

    # 执行按钮
    if st.session_state.execution_status in ("idle", "error"):
        if st.button("🚀 执行工作流", type="primary", use_container_width=True):
            st.session_state.execution_status = "running"
            st.rerun()

    # 执行过程
    if st.session_state.execution_status == "running":
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_area = st.empty()

        try:
            steps = ["数据加载", "预处理", "权重计算", "综合评价", "灵敏度分析", "结果汇总"]
            total_steps = len(steps)

            for i, step_name in enumerate(steps):
                status_text.markdown(f"**⏳ 正在执行：{step_name}...**")
                time.sleep(0.8)  # 模拟执行时间
                progress_bar.progress((i + 1) / total_steps)
                st.session_state.execution_log.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {step_name} 完成"
                )
                log_area.code("\n".join(st.session_state.execution_log))

            # 执行核心逻辑
            if BACKEND_AVAILABLE:
                # 使用真实后端
                engine = WorkflowEngine()
                results = engine.execute(config, df)
            else:
                # 使用模拟结果
                results = mock_execute_workflow(config, df)

            st.session_state.execution_results = results
            st.session_state.execution_status = "success"

            status_text.markdown("**✅ 工作流执行完成！**")
            progress_bar.progress(1.0)
            time.sleep(0.5)
            st.rerun()

        except Exception as e:
            st.session_state.execution_status = "error"
            st.error(f"执行出错：{e}")
            st.code(traceback.format_exc())
            if st.button("🔄 重试"):
                st.session_state.execution_status = "idle"
                st.session_state.execution_log = []
                st.rerun()
            return

    # 结果展示
    if st.session_state.execution_status == "success" and st.session_state.execution_results:
        results = st.session_state.execution_results

        st.balloons()

        # 执行日志
        with st.expander("📋 执行日志", expanded=False):
            st.code("\n".join(st.session_state.execution_log))

        st.markdown("---")

        # 结果标签页
        tab_weights, tab_scores, tab_viz, tab_code, tab_latex = st.tabs(
            ["⚖️ 权重结果", "📊 评价得分", "📈 可视化", "💻 生成代码", "📄 LaTeX 报告"]
        )

        with tab_weights:
            st.subheader("指标权重")
            weights = results.get("weights", {})
            if weights:
                weight_df = pd.DataFrame(
                    list(weights.items()), columns=["指标", "权重"]
                ).sort_values("权重", ascending=False)
                st.dataframe(weight_df, use_container_width=True, hide_index=True)

                # 权重条形图
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False

                fig_w, ax_w = plt.subplots(figsize=(10, max(4, len(weights) * 0.5)))
                sorted_w = dict(sorted(weights.items(), key=lambda x: x[1]))
                ax_w.barh(list(sorted_w.keys()), list(sorted_w.values()), color="steelblue")
                ax_w.set_xlabel("权重")
                ax_w.set_title("指标权重分布")
                plt.tight_layout()
                st.pyplot(fig_w)

        with tab_scores:
            st.subheader("综合评价得分与排名")
            ranking = results.get("ranking", [])
            if ranking:
                rank_df = pd.DataFrame(ranking, columns=["评价对象", "综合得分", "排名"])
                st.dataframe(
                    rank_df.style.background_gradient(
                        subset=["综合得分"], cmap="RdYlGn"
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # 排名柱状图
                fig_r, ax_r = plt.subplots(figsize=(10, max(4, len(ranking) * 0.4)))
                sorted_rank = sorted(ranking, key=lambda x: x[1])
                names = [r[0] for r in sorted_rank]
                scores_vals = [r[1] for r in sorted_rank]
                colors = plt.cm.RdYlGn(
                    np.linspace(0.2, 0.8, len(sorted_rank))
                )
                ax_r.barh(names, scores_vals, color=colors)
                ax_r.set_xlabel("综合得分")
                ax_r.set_title(f"{st.session_state.eval_method.upper()} 综合评价排名")
                plt.tight_layout()
                st.pyplot(fig_r)

        with tab_viz:
            st.subheader("结果可视化")

            viz_type = st.selectbox(
                "选择图表类型",
                ["雷达图", "得分分布箱线图", "指标热力图", "排名对比图"],
            )

            indicator_cols = st.session_state.indicator_columns
            obj_col = st.session_state.object_column

            if viz_type == "雷达图" and df is not None and indicator_cols:
                from math import pi
                fig_radar, ax_radar = plt.subplots(
                    figsize=(8, 8), subplot_kw=dict(polar=True)
                )
                angles = np.linspace(0, 2 * pi, len(indicator_cols), endpoint=False).tolist()
                angles += angles[:1]

                # 标准化数据用于雷达图
                sub = df[indicator_cols].values.astype(float)
                sub_min = sub.min(axis=0)
                sub_max = sub.max(axis=0)
                sub_norm = (sub - sub_min) / (sub_max - sub_min + 1e-12)

                objects_list = (
                    df[obj_col].tolist()
                    if obj_col and obj_col in df.columns
                    else [f"对象{i + 1}" for i in range(len(df))]
                )

                max_show = min(6, len(objects_list))
                for i in range(max_show):
                    values = sub_norm[i].tolist()
                    values += values[:1]
                    ax_radar.plot(angles, values, "o-", linewidth=1.5, label=objects_list[i])
                    ax_radar.fill(angles, values, alpha=0.08)

                ax_radar.set_xticks(angles[:-1])
                ax_radar.set_xticklabels(indicator_cols, fontsize=9)
                ax_radar.set_title("指标雷达图")
                ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_radar)

            elif viz_type == "得分分布箱线图":
                scores_dict = results.get("scores", {})
                if scores_dict:
                    fig_box, ax_box = plt.subplots(figsize=(8, 5))
                    ax_box.boxplot(list(scores_dict.values()), vert=True)
                    ax_box.set_title("综合得分分布")
                    ax_box.set_ylabel("得分")
                    st.pyplot(fig_box)

            elif viz_type == "指标热力图" and df is not None and indicator_cols:
                fig_hm, ax_hm = plt.subplots(figsize=(12, max(5, len(df) * 0.3)))
                sub = df[indicator_cols].values.astype(float)
                sub_min = sub.min(axis=0)
                sub_max = sub.max(axis=0)
                sub_norm = (sub - sub_min) / (sub_max - sub_min + 1e-12)

                im = ax_hm.imshow(sub_norm, cmap="YlOrRd", aspect="auto")
                ax_hm.set_xticks(range(len(indicator_cols)))
                ax_hm.set_xticklabels(indicator_cols, rotation=45, ha="right")
                objects_list = (
                    df[obj_col].tolist()
                    if obj_col and obj_col in df.columns
                    else [f"对象{i + 1}" for i in range(len(df))]
                )
                ax_hm.set_yticks(range(len(objects_list)))
                ax_hm.set_yticklabels(objects_list)
                plt.colorbar(im, ax=ax_hm)
                ax_hm.set_title("指标值热力图（归一化后）")
                plt.tight_layout()
                st.pyplot(fig_hm)

        with tab_code:
            st.subheader("生成的 Python 代码")
            generated_code = results.get("generated_code", "# 暂无生成代码")
            st.code(generated_code, language="python", line_numbers=True)

            st.download_button(
                "💾 下载 Python 脚本",
                data=generated_code,
                file_name=f"eval_{st.session_state.eval_method}_{datetime.now().strftime('%Y%m%d_%H%M')}.py",
                mime="text/x-python",
                use_container_width=True,
            )

        with tab_latex:
            st.subheader("LaTeX 报告段落")
            latex_sections = results.get("latex_sections", {})

            for section_name, content in latex_sections.items():
                with st.expander(f"📄 {section_name}", expanded=True):
                    st.code(content, language="latex")

            # 合并所有 LaTeX 内容
            full_latex = "\n\n".join(latex_sections.values())
            if full_latex:
                st.download_button(
                    "💾 下载 LaTeX 源码",
                    data=full_latex,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.tex",
                    mime="application/x-tex",
                    use_container_width=True,
                )

        # 重新执行
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("⬅️ 返回配置", use_container_width=True):
                st.session_state.current_step = 2
                st.session_state.execution_status = "idle"
                st.session_state.execution_log = []
                st.rerun()
        with col2:
            if st.button("🔄 重新执行", use_container_width=True):
                st.session_state.execution_status = "idle"
                st.session_state.execution_log = []
                st.rerun()

    # 导航到导出
    if st.session_state.execution_status == "success":
        st.markdown("---")
        _, _, col_next = st.columns([1, 4, 1])
        with col_next:
            if st.button("下一步 ➡️ 导出", use_container_width=True, type="primary"):
                st.session_state.current_step = 5
                st.rerun()


def render_step5_export():
    """Step 5: 导出与下载"""
    st.header("📥 Step 6: 导出下载")

    results = st.session_state.execution_results
    if not results:
        st.warning("暂无执行结果，请先执行工作流。")
        if st.button("⬅️ 返回执行"):
            st.session_state.current_step = 4
            st.rerun()
        return

    st.success("✅ 所有文件已准备就绪，选择需要下载的内容：")

    st.markdown("---")

    # 下载区域
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 文档类")

        # Python 脚本
        code = results.get("generated_code", "")
        if code:
            st.download_button(
                "💻 完整 Python 脚本",
                data=code,
                file_name=f"auto_eval_{st.session_state.eval_method}.py",
                mime="text/x-python",
                use_container_width=True,
            )

        # LaTeX 报告
        latex_sections = results.get("latex_sections", {})
        full_latex = _build_full_latex_document(latex_sections)
        if full_latex:
            st.download_button(
                "📝 LaTeX 完整报告",
                data=full_latex,
                file_name="modeling_report.tex",
                mime="application/x-tex",
                use_container_width=True,
            )

        # YAML 工作流配置
        yaml_text = st.session_state.workflow_yaml
        if yaml_text:
            st.download_button(
                "⚙️ 工作流配置 (YAML)",
                data=yaml_text,
                file_name="workflow_config.yaml",
                mime="text/yaml",
                use_container_width=True,
            )

    with col2:
        st.subheader("📊 数据类")

        # 权重结果
        weights = results.get("weights", {})
        if weights:
            weight_df = pd.DataFrame(list(weights.items()), columns=["指标", "权重"])
            st.download_button(
                "⚖️ 权重结果 (CSV)",
                data=weight_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="weights.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # 排名结果
        ranking = results.get("ranking", [])
        if ranking:
            rank_df = pd.DataFrame(ranking, columns=["评价对象", "综合得分", "排名"])
            st.download_button(
                "📊 排名结果 (CSV)",
                data=rank_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="ranking.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # 执行日志
        log_text = "\n".join(st.session_state.execution_log)
        st.download_button(
            "📋 执行日志 (TXT)",
            data=log_text,
            file_name="execution_log.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # 一键打包
    st.markdown("---")
    st.subheader("📦 一键打包下载")
    st.caption("将所有生成文件打包为 ZIP 压缩包")

    if st.button("📦 打包全部文件", type="primary", use_container_width=True):
        import zipfile
        import io

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Python 脚本
            if code:
                zf.writestr(f"scripts/auto_eval_{st.session_state.eval_method}.py", code)
            # LaTeX
            if full_latex:
                zf.writestr("reports/modeling_report.tex", full_latex)
            # YAML
            if yaml_text:
                zf.writestr("configs/workflow_config.yaml", yaml_text)
            # 权重
            if weights:
                zf.writestr("data/weights.csv", weight_df.to_csv(index=False, encoding="utf-8-sig"))
            # 排名
            if ranking:
                rank_csv = pd.DataFrame(ranking, columns=["评价对象", "综合得分", "排名"])
                zf.writestr("data/ranking.csv", rank_csv.to_csv(index=False, encoding="utf-8-sig"))
            # 日志
            zf.writestr("logs/execution_log.txt", log_text)
            # README
            readme = f"""# AutoEval-Modeling 输出结果
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
评价方法: {st.session_state.weight_method} + {st.session_state.eval_method}

## 文件说明
- scripts/    : 可独立运行的 Python 脚本
- reports/    : LaTeX 报告源码
- configs/    : 工作流配置文件
- data/       : 权重与排名结果
- logs/       : 执行日志
"""
            zf.writestr("README.md", readme)

        zip_buffer.seek(0)
        st.download_button(
            "💾 下载 ZIP 压缩包",
            data=zip_buffer.getvalue(),
            file_name=f"AutoEval_Output_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    # 返回
    st.markdown("---")
    col_prev, _, _ = st.columns([1, 4, 1])
    with col_prev:
        if st.button("⬅️ 返回结果页", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()


def _build_full_latex_document(sections: Dict[str, str]) -> str:
    """将各 LaTeX 段落组装为完整文档"""
    header = r"""\documentclass[12pt,a4paper]{ctexart}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{float}
\usepackage{hyperref}
\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}

\title{基于多属性决策的综合评价模型}
\author{AutoEval-Modeling 自动生成}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""
    body = "\n\n".join(sections.values())
    footer = r"""
\end{document}
"""
    return header + body + footer


# ╔══════════════════════════════════════════════════════════════╗
# ║                        主程序                                ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    """Streamlit 应用主入口"""
    # 页面配置
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 自定义 CSS
    st.markdown("""
    <style>
    /* 全局字体 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* 步骤标题 */
    h1 {
        color: #1f4e79;
        border-bottom: 2px solid #1f4e79;
        padding-bottom: 0.3em;
    }
    h2 {
        color: #2e75b6;
    }
    /* 按钮样式优化 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # 初始化状态
    init_session_state()

    # 渲染侧边栏
    render_sidebar()

    # 根据当前步骤渲染主内容
    step = st.session_state.current_step
    if step == 0:
        render_step0_problem_definition()
    elif step == 1:
        render_step1_data_upload()
    elif step == 2:
        render_step2_algorithm_config()
    elif step == 3:
        render_step3_workflow_preview()
    elif step == 4:
        render_step4_execution()
    elif step == 5:
        render_step5_export()
    else:
        st.error(f"未知步骤: {step}")
        st.session_state.current_step = 0
        st.rerun()


if __name__ == "__main__":
    main()