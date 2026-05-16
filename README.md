<div align="center">

# 🏗️ AutoEval-Modeling

**面向评价类数学建模的自动化工作流引擎与代码生成系统**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange)]()

*问题驱动 · 一键生成 · 学术规范 · 完全可复现*

</div>

---

## 📖 目录

- [项目简介](#-项目简介)
- [核心特性](#-核心特性)
- [系统架构](#-系统架构)
- [快速开始](#-快速开始)
  - [环境要求](#环境要求)
  - [安装](#安装)
  - [三分钟上手](#三分钟上手)
- [使用方式](#-使用方式)
  - [命令行模式](#命令行模式)
  - [配置文件驱动](#配置文件驱动)
  - [Web 交互界面](#web-交互界面)
  - [Python API 调用](#python-api-调用)
- [配置文件详解](#-配置文件详解)
- [支持的算法](#-支持的算法)
- [输出产物说明](#-输出产物说明)
- [项目结构](#-项目结构)
- [开发路线图](#-开发路线图)
- [常见问题](#-常见问题)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)
- [致谢](#-致谢)

---

## 🎯 项目简介

**AutoEval-Modeling** 是一个「问题驱动」的自动化评价建模工具。用户只需：

1. 📋 描述评价问题（或选择预设模板）
2. 📊 上传数据文件（CSV / Excel）
3. ⚙️ 设定少量关键参数

系统即可自动完成从 **问题定义 → 数据预处理 → 权重确定 → 综合评价 → 结果分析 → 报告输出** 的全流程，生成：

| 产物 | 说明 |
|------|------|
| 🐍 **完整 Python 脚本** | 分节注释，可独立运行复现全部结果 |
| 📝 **LaTeX 论文源码** | 符合数学建模竞赛规范，含公式、表格、图片引用 |
| 📊 **高质量可视化图表** | 权重分布、排名对比、雷达图、灵敏度曲线等 |
| 📋 **运行日志与中间数据** | 便于调试与结果验证 |

> 🎓 **适用场景**：数学建模竞赛（国赛/美赛）、课程论文、学术研究中的多属性综合评价问题。

---

## ✨ 核心特性

### 🔄 全流程自动化

数据上传 → 智能预处理 → 算法推荐 → 权重计算 → 综合评价 → 灵敏度分析 → 报告生成

全程无需手动编写代码，一条命令或一次点击即可完成。

### 🧠 智能算法推荐
- 自动分析数据特征（缺失率、相关性、指标数/样本数比等）
- 基于知识图谱匹配最优算法组合
- 提供多套备选方案附选择理由，保留用户最终决定权

### 🧩 模块化可插拔
- 所有算法统一 `BaseMethod` 接口
- 工作流通过 YAML 配置文件编排，支持条件分支
- 轻松扩展自定义算法或模板

### 📄 学术规范输出
- LaTeX 模板覆盖 14 种主流评价算法
- 自动生成三线表（booktabs）、规范公式编号、图表交叉引用
- 支持中文（ctex）/ 英文论文模板

### 🛡️ 健壮可靠
- 全面的异常捕获与自动修复建议
- 一致性检验不通过时自动提示调整
- 矩阵奇异等数值问题的降级处理策略

---

## 🏛️ 系统架构
```
┌─────────────────────────────────────────────────────────┐
│ 用户输入层 │
│ [问题描述] [数据文件] [YAML配置] [输出偏好] │
└────────────────────┬────────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────┐
│ 工作流编排引擎 (Orchestrator) │
│ ┌──────────────┐ ┌──────────────┐ ┌────────────────┐ │
│ │ 情境解析器 │ │ 算法知识图谱 │ │ DAG工作流构建 │ │
│ └──────────────┘ └──────────────┘ └────────────────┘ │
└────────────┬──────────────────┬──────────────────────────┘
▼ ▼
┌────────────────────┐ ┌────────────────────────────────┐
│ 算法库 (AlgoLib) │ │ 数据预处理模块 (Preprocessor) │
│ · 主观赋权 (3种) │ │ · 缺失值/异常值处理 │
│ · 客观赋权 (4种) │ │ · 标准化/归一化/正向化 │
│ · 综合评价 (8种) │ │ · 相关性分析/PCA降维 │
│ · 组合/检验 (4种) │ └────────────┬───────────────────┘
└─────────┬──────────┘ │
▼ ▼
┌─────────────────────────────────────────────────────────┐
│ 核心执行引擎 (Python 沙箱) │
└────────────────────┬────────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────┐
│ 结果生成器 │
│ [LaTeX报告] [Python脚本] [可视化图表] │
└─────────────────────────────────────────────────────────┘
```


---

## 🚀 快速开始

### 环境要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.10+ | 核心运行环境 |
| pip | 22.0+ | 包管理器 |
| LaTeX | TeX Live 2022+ | （可选）编译生成的 .tex 文件 |
| Git | 2.30+ | 克隆仓库 |

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/xassqdq/auto-eval-modeling.git
cd auto-eval-modeling

# 2. 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "from src import __version__; print(f'AutoEval-Modeling v{__version__} installed successfully!')"
```

### 三分钟上手
```bash
# 使用预置示例数据 + 默认配置，一键运行
python -m src.main run --config configs/entropy_topsis.yaml --data data/city_innovation.csv

# 查看输出
ls output/
# ├── reports/report.tex          ← LaTeX 论文源码
# ├── figures/                    ← 所有图表
# ├── scripts/eval_script.py      ← 可独立运行的 Python 脚本
# └── logs/run.log                ← 运行日志
```

## 📘 使用方式
### 命令行模式

```bash
# 初始化配置模板（交互式问答生成 YAML）
python -m src.main init

# 从配置文件运行
python -m src.main run --config configs/entropy_topsis.yaml \
                       --data data/city_innovation.csv \
                       --output output/my_project/

# 仅生成推荐方案（不执行）
python -m src.main recommend --data data/city_innovation.csv

# 仅生成 LaTeX 报告（从已有结果）
python -m src.main report --results output/my_project/results.json \
                          --template cn_competition
```

### 配置文件驱动

编写或修改 YAML 配置文件，定义完整工作流：
```yaml
# my_config.yaml - 最简示例
project:
  name: "城市创新能力评价"

data:
  file: "data/city_innovation.csv"
  object_column: "城市"
  criteria_columns: ["R&D投入", "专利数", "高新企业数", "GDP增长率"]
  directions: [max, max, max, max]

workflow:
  - step: normalize
    method: minmax
  - step: weight
    method: entropy
  - step: evaluate
    method: topsis
  - step: sensitivity
    method: oat
    params:
      perturbation: 0.1

output:
  format: [latex, python, figures]
  language: cn
```

```bash
python -m src.main run --config my_config.yaml
```

```bash
# 启动 Streamlit 界面
streamlit run app/streamlit_app.py

# 或 Gradio 界面
python app/gradio_app.py
```

浏览器打开 http://localhost:8501，通过图形界面：

- 上传数据文件

- 勾选指标方向

- 选择算法组合（或使用智能推荐）

- 点击「一键运行」

- 在线预览并下载全部结果


### Python API 调用

```Python
from src.core.workflow import WorkflowEngine
from src.utils.config_loader import load_config

# 加载配置
config = load_config("configs/entropy_topsis.yaml")

# 创建并运行工作流
engine = WorkflowEngine(config)
engine.load_data("data/city_innovation.csv")
results = engine.run()

# 访问结果
print(results.weights)          # 权重向量
print(results.scores)           # 综合得分
print(results.rankings)         # 排名

# 生成报告
engine.generate_report(output_dir="output/", fmt=["latex", "python", "figures"])
```

### 单独使用某个算法：

```Python
from src.algorithms.weights.objective import EntropyWeight
from src.algorithms.evaluation.topsis import TOPSIS
import pandas as pd

data = pd.read_csv("data/city_innovation.csv")
matrix = data[["R&D投入", "专利数", "高新企业数", "GDP增长率"]].values

# 熵权法
ew = EntropyWeight()
ew.fit(matrix)
print("权重:", ew.weights_)
print("信息熵:", ew.entropy_values_)

# TOPSIS
topsis = TOPSIS(weights=ew.weights_, directions=[1, 1, 1, 1])
topsis.fit(matrix)
print("贴近度:", topsis.closeness_)
print("排名:", topsis.rankings_)

# 获取 LaTeX 描述
print(topsis.tex_description())
```

## 📋 配置文件详解

配置文件采用 YAML 格式，通过 Pydantic 进行严格校验。完整字段说明如下：

```Yaml
# ============================================================
# AutoEval-Modeling 工作流配置文件 · 完整字段参考
# ============================================================

# -------------------- 项目元信息 --------------------
project:
  name: "项目名称"                          # 必填
  description: "项目描述"                   # 可选
  author: "作者"                            # 可选，默认 "AutoEval-Modeling"
  version: "1.0"                            # 可选

# -------------------- 数据配置 --------------------
data:
  file: "path/to/data.csv"                  # 必填，支持 csv/xlsx/xls
  sheet_name: "Sheet1"                      # Excel 时可选
  object_column: "对象列名"                  # 必填，标识评价对象
  criteria_columns:                         # 必填，评价指标列名列表
    - "指标1"
    - "指标2"
  directions:                               # 必填，与 criteria_columns 一一对应
    - max                                   # max=正向(越大越好)
    - min                                   # min=负向(越小越好)
    - moderate                              # moderate=适度型(需配合 moderate_params)
  moderate_params:                          # 仅适度型指标需要
    指标名: {best: 50, tolerance: 10}       # 最优值及容忍范围
  time_column: null                         # 动态评价时的时间列名
  group_column: null                        # 分组评价时的分组列名

# -------------------- 预处理配置 --------------------
preprocessing:
  missing_strategy: "mean"                  # 缺失值: mean/median/mode/drop/knn
  outlier_strategy: "clip"                  # 异常值: clip/remove/none
  outlier_method: "iqr"                     # 检测方法: iqr/zscore/isolation_forest
  outlier_threshold: 1.5                    # IQR 倍数或 Z-score 阈值
  normalization: "minmax"                   # 标准化: minmax/zscore/maxabs/none
  correlation_check: true                   # 是否检查指标相关性
  correlation_threshold: 0.9                # 高相关阈值
  auto_reduce: false                        # 高相关时是否自动 PCA 降维
  reduce_variance_threshold: 0.85           # PCA 累计方差贡献率阈值

# -------------------- 工作流步骤 --------------------
workflow:
  # 每个步骤为一个节点，按顺序执行
  # 可用 step 类型: preprocess, reduce, weight, evaluate, combine, sensitivity, cluster
  
  - step: preprocess
    # 使用上方 preprocessing 配置，无需额外参数
  
  - step: weight
    method: entropy                         # 赋权方法 (见下方算法表)
    params: {}                              # 算法专属参数
    output_key: "w_entropy"                 # 结果键名（用于后续引用）
  
  - step: weight
    method: ahp
    params:
      judgment_matrix:                      # AHP 判断矩阵（手动输入）
        - [1, 3, 5, 2]
        - [0.333, 1, 3, 1]
        - [0.2, 0.333, 1, 0.5]
        - [0.5, 1, 2, 1]
      auto_fix_consistency: true            # 一致性不通过时自动微调
    output_key: "w_ahp"
  
  - step: combine
    method: game_theory                     # 组合策略: linear/multiplicative/game_theory
    sources: ["w_ahp", "w_entropy"]         # 要组合的权重键名
    params:
      alpha: 0.5                            # 仅 linear 模式
    output_key: "w_combined"
  
  - step: evaluate
    method: topsis
    weight_key: "w_combined"                # 引用哪个权重
    params: {}
    output_key: "eval_topsis"
  
  - step: evaluate
    method: vikor
    weight_key: "w_combined"
    params:
      v: 0.5                               # 决策机制系数
    output_key: "eval_vikor"
  
  - step: sensitivity
    method: oat
    eval_key: "eval_topsis"                 # 对哪个评价结果做灵敏度
    weight_key: "w_combined"
    params:
      perturbation_range: 0.1              # ±10%
      perturbation_steps: 20
      consistency_metric: kendall_tau
  
  - step: rank_compare
    eval_keys: ["eval_topsis", "eval_vikor"]  # 多模型排名对比
    params:
      metrics: [kendall_tau, spearman_rho]

# -------------------- 输出配置 --------------------
output:
  directory: "output/"                      # 输出根目录
  format:                                   # 要生成的产物列表
    - latex                                 # LaTeX 论文源码
    - python                                # 可独立运行的 Python 脚本
    - figures                               # 可视化图表
    - excel                                 # Excel 结果汇总
    - json                                  # 结构化 JSON 结果
  
  language: "cn"                            # 报告语言: cn/en
  latex_template: "cn_competition"          # 模板: cn_competition/en_mcm/academic
  figure_format: "png"                      # 图片格式: png/pdf/svg
  figure_dpi: 300                           # 图片分辨率
  
  figures:                                  # 要生成的图表类型
    - weights_bar                           # 权重条形图
    - ranking_bar                           # 排名柱状图
    - radar_chart                           # 雷达图
    - correlation_heatmap                   # 相关性热力图
    - sensitivity_curve                     # 灵敏度曲线
    - score_distribution                    # 得分分布图
  
  code_style:
    add_comments: true                      # 代码是否添加详细注释
    add_docstrings: true                    # 是否添加文档字符串
    standalone: true                        # 生成的脚本是否可独立运行

# -------------------- 高级选项 --------------------
advanced:
  random_seed: 42                           # 随机种子
  float_precision: 6                        # 小数位数
  parallel: false                           # 是否并行执行独立节点
  verbose: true                             # 详细日志
  log_file: "output/logs/run.log"           # 日志文件路径
  save_intermediate: true                   # 是否保存中间结果
```

### 算法方法名速查
| step 类型 | method 值 | 说明 |
| --------- | ---------- | -------------- |
| weight    | ahp        | 层次分析法 |
| weight    | delphi     | 德尔菲法 |
| weight    | binomial   | 二项系数法 |
| weight    | entropy    | 熵权法 |
| weight    | critic     | CRITIC法 |
| weight    | std_deviation | 标准离差法 |
| weight    | pca_weight | PCA求权重 |
| combine   | linear     | 线性加权组合 |
| combine   | multiplicative | 乘法归一化组合 |
| combine   | game_theory | 博弈论组合 |
| evaluate  | topsis     | TOPSIS |
| evaluate  | vikor      | VIKOR |
| evaluate  | gra        | 灰色关联分析 |
| evaluate  | fuzzy      | 模糊综合评价 |
| evaluate  | electre    | ELECTRE |
| evaluate  | rsr        | 秩和比法 |
| evaluate  | dea        | 数据包络分析 |
| evaluate  | dynamic    | 动态时序评价 |
| reduce    | pca        | 主成分分析降维 |
| reduce    | factor_analysis | 因子分析 |
| sensitivity | oat      | OAT权重灵敏度 |
| sensitivity | monte_carlo | 蒙特卡洛灵敏度 |


## 📊 支持的算法
### 赋权方法
| 类别 | 算法 | 适用场景 | 关键输出 |
| ---- | ---- | ---- | ---- |
| 主观 | AHP 层次分析法 | 有专家经验，指标较少 | 权重、一致性比例、判断矩阵热力图 |
| 主观 | 德尔菲法 | 多轮专家咨询 | 权重、专家一致性系数 |
| 主观 | 二项系数法 | 快速排序赋权 | 权重向量 |
| 客观 | 熵权法 | 通用，数据驱动 | 权重、信息熵、差异系数 |
| 客观 | CRITIC法 | 指标相关性显著 | 权重、对比强度、冲突性 |
| 客观 | 标准离差法 | 快速客观赋权 | 权重、标准差 |
| 客观 | PCA求权重 | 指标高度相关 | 权重、贡献率 |
| 组合 | 线性/乘法/博弈论 | 平衡主客观 | 组合权重、对比图 |


## 评价方法
| 算法 | 核心思想 | 适用场景 | 关键输出 |
| ---- | ---- | ---- | ---- |
| TOPSIS | 距理想解距离 | 多属性排序（最通用） | 贴近度、排名、雷达图 |
| VIKOR | 群体效用+个体遗憾折中 | 需要折中解 | S/R/Q值、折中检验 |
| 灰色关联(GRA) | 序列几何相似度 | 小样本、信息不完全 | 关联度、排名 |
| 模糊综合评价 | 隶属度合成 | 定性+定量混合 | 评价向量、等级 |
| ELECTRE | 优超关系 | 需区分不可比 | 一致/不一致矩阵、核心集 |
| 秩和比(RSR) | 非参数编秩 | 对分布无要求 | RSR值、Probit分档 |
| DEA | 线性规划效率前沿 | 投入-产出效率 | 效率值、是否有效 |
| 动态评价 | 时间加权汇总 | 多年度面板数据 | 趋势图、动态排名 |

## 📁 项目结构

```aiignore
auto-eval-modeling/
│
├── 📄 README.md                     ← 本文件
├── 📄 requirements.txt              ← Python 依赖
├── 📄 setup.py                      ← 安装脚本
├── 📄 LICENSE                       ← MIT 许可证
├── 📄 .gitignore
│
├── 📁 configs/                      ← 工作流配置文件
│   ├── __schema__.yaml              ← 配置字段说明
│   ├── default_ahp_topsis.yaml
│   ├── entropy_topsis.yaml
│   ├── entropy_gra.yaml
│   ├── dynamic_eval.yaml
│   ├── full_pipeline.yaml
│   └── ...
│
├── 📁 data/                         ← 示例数据集
│   ├── city_innovation.csv
│   ├── enterprise_performance.xlsx
│   └── regional_economy.csv
│
├── 📁 examples/                     ← 完整运行示例
│   ├── run_from_config.py
│   └── interactive_demo.py
│
├── 📁 tests/                        ← 单元测试
│   ├── test_weights.py
│   ├── test_evaluation.py
│   ├── test_workflow.py
│   └── test_generators.py
│
├── 📁 output/                       ← 运行输出（自动生成）
│   ├── reports/
│   ├── figures/
│   ├── scripts/
│   └── logs/
│
├── 📁 src/                          ← 核心源码
│   ├── __init__.py
│   ├── main.py                      ← CLI 入口
│   │
│   ├── 📁 parser/                   ← 问题情境解析
│   │   ├── nlp_parser.py
│   │   ├── context_type.py
│   │   └── data_profiler.py
│   │
│   ├── 📁 algorithms/              ← 评价算法库
│   │   ├── base.py                  ← BaseMethod 抽象基类
│   │   ├── 📁 preprocess/          ← 预处理
│   │   ├── 📁 weights/             ← 赋权方法
│   │   ├── 📁 evaluation/          ← 评价模型
│   │   └── 📁 sensitivity/         ← 灵敏度检验
│   │
│   ├── 📁 core/                    ← 工作流引擎
│   │   ├── workflow.py
│   │   ├── node.py
│   │   ├── sandbox.py
│   │   ├── data_bus.py
│   │   └── recommendation.py
│   │
│   ├── 📁 generators/             ← 输出生成器
│   │   ├── latex_builder.py
│   │   ├── code_builder.py
│   │   └── plot_builder.py
│   │
│   ├── 📁 templates/              ← 模板文件
│   │   ├── 📁 latex/              ← LaTeX 模板
│   │   └── 📁 python/             ← 代码生成模板
│   │
│   └── 📁 utils/                  ← 通用工具
│       ├── config_loader.py
│       ├── logging_config.py
│       └── file_handler.py
│
└── 📁 app/                         ← Web 界面
    ├── streamlit_app.py
    └── gradio_app.py

```


# 🗺️ 开发路线图
| 阶段 | 内容 | 状态 |
| ---- | ---- | ---- |
| Phase 1 | 核心算法库封装（14种算法统一接口） | 🔧 进行中 |
| Phase 2 | DAG 工作流引擎 + YAML 配置驱动 | 📋 计划中 |
| Phase 3 | 智能推荐 + Streamlit Web 界面 | 📋 计划中 |
| Phase 4 | LaTeX 完整报告生成（已完成模板设计） | ✅ 模板就绪 |
| Phase 5 | 高级功能：PROMETHEE、并行执行、交互调参 | 📋 计划中 |


<details>
<summary><b>Q: 支持哪些数据格式？</b></summary>

支持 CSV（.csv）、Excel（.xlsx / .xls）格式。数据要求：
- 第一行为列标题
- 至少包含一列评价对象标识、一列以上数值型指标
- 缺失值可自动处理（可配置策略）
</details>

<details>
<summary><b>Q: AHP 判断矩阵一致性检验不通过怎么办？</b></summary>

配置文件中设置 `auto_fix_consistency: true`，系统会自动微调判断矩阵使其通过一致性检验，并在报告中说明调整过程。也可手动修改判断矩阵后重新运行。
</details>

<details>
<summary><b>Q: 如何添加自定义算法？</b></summary>

1. 在 `src/algorithms/` 对应子目录创建新文件
2. 继承 `BaseMethod` 类，实现 `fit()`, `compute()`, `summary()`, `tex_description()` 方法
3. 在工作流配置中使用新算法的 method 名即可
</details>

<details>
<summary><b>Q: 生成的 LaTeX 如何编译？</b></summary>

```bash
cd output/reports/
xelatex report.tex    # 中文需使用 XeLaTeX
bibtex report          # 如有参考文献
xelatex report.tex
xelatex report.tex     # 编译两次确保交叉引用正确
```
</details> <details> <summary><b>Q: 可以不装 LaTeX 只生成代码和图表吗？</b></summary>

可以。在配置文件中设置 output.format: [python, figures] 即可跳过 LaTeX 生成。

</details>


# 🤝 贡献指南
欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

## 规范要求
- 代码通过所有现有测试：`pytest tests/`
- 新功能附带对应测试
- 遵循现有代码风格（Black 格式化 + isort 排序）


# 📜 许可证
本项目采用 **MIT License** 开源。

# 🙏 致谢
- scikit-learn — 机器学习基础库
- pyDecision — MCDA 算法参考
- Jinja2 — 模板引擎
- Streamlit — Web 界面框架
- 数学建模社区的开源贡献者们

<div align="center">

⭐ 如果本项目对你有帮助，请给一个 Star！⭐

Made with ❤️ for the mathematical modeling community

</div> 



# main.py各节职责速查
| 节次 | 内容 | 关键类/函数 |
| ---- | ---- | ---- |
| 第一节 | 日志系统配置 | setup_logging() |
| 第二节 | 运行状态数据结构 | RunStatus, StepResult, EngineResult |
| 第三节 | 模块延迟导入与降级 | ModuleRegistry |
| 第四节 | 配置对象 | EngineConfig |
| 第五节 | 核心编排引擎 | AutoEvalEngine |
| 第六节 | 内置降级算法 | _simple_entropy_weight(), _simple_topsis(), _rule_based_recommendation() |
| 第七节 | 自定义异常 | _FatalError, ConfigError, DataError |
| 第八节 | CLI 参数解析 | _build_cli_parser() |
| 第九节 | 子命令处理器 | _cmd_run(), _cmd_init(), _cmd_analyze() 等 |
| 第十节 | YAML 模板常量 | _YAML_TEMPLATE_* |
| 第十一节 | 程序入口 | main() |



```bash
# 安装最小依赖
pip install numpy pandas matplotlib pyyaml

# 查看帮助
python -m src.main --help

# 列出算法
python -m src.main list-algorithms

# 生成配置模板
python -m src.main init --output my_test.yaml --non-interactive

# 数据分析诊断
python -m src.main analyze --data data/city_innovation.csv

# 完整运行（需配置文件 + 数据）
python -m src.main run --config configs/default_ahp_topsis.yaml
```





