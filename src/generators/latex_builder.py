"""
LaTeX 报告生成器
================================================
将评价算法的计算结果转换为符合学术规范的 LaTeX 源代码。

支持特性：
  - 中文（ctex）/ 英文双语模板
  - booktabs 风格数据表格
  - 各算法专属公式段落（AHP / 熵权 / CRITIC / TOPSIS / GRA / VIKOR /
    模糊评价 / PCA / 灵敏度分析）
  - 图片引用、双栏并排图
  - 完整文档骨架组装 & 分节输出
  - 无外部模板文件依赖（模板内嵌于模块中）

模板占位符约定
--------------
模板字符串使用 <<VAR_NAME>> 作为变量占位符（无 Python 格式符冲突），
内部通过 _render() 方法完成替换，LaTeX 原生 {}、\ 无需转义。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  模板字符串（内嵌，无需外部 .tex 模板文件）
# ══════════════════════════════════════════════════════════════════════════════

# ── 文档骨架 ──────────────────────────────────────────────────────────────────
_DOC_CN = r"""
\documentclass[12pt,a4paper]{article}
\usepackage[UTF8]{ctex}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow,makecell}
\usepackage{longtable}
\usepackage{xcolor,colortbl}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{caption,subcaption}
\usepackage{array,tabularx}
\usepackage{float}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{setspace}

\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}
\setstretch{1.5}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=green,urlcolor=cyan}
\captionsetup{labelsep=space,font=small}

\pagestyle{fancy}
\fancyhf{}
\rhead{\leftmark}
\cfoot{\thepage}

\title{\textbf{<<TITLE>>}}
\author{<<AUTHOR>>}
\date{<<DATE>>}

\begin{document}
\maketitle

\begin{abstract}
<<ABSTRACT>>
\end{abstract}

\tableofcontents
\newpage

<<CONTENT>>

\end{document}
"""

_DOC_EN = r"""
\documentclass[12pt,a4paper]{article}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow,makecell}
\usepackage{longtable}
\usepackage{xcolor,colortbl}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{caption,subcaption}
\usepackage{array,tabularx}
\usepackage{float}
\usepackage{setspace}

\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}
\setstretch{1.5}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=green,urlcolor=cyan}

\title{\textbf{<<TITLE>>}}
\author{<<AUTHOR>>}
\date{<<DATE>>}

\begin{document}
\maketitle

\begin{abstract}
<<ABSTRACT>>
\end{abstract}

\tableofcontents
\newpage

<<CONTENT>>

\end{document}
"""

# ── AHP 段落模板 ──────────────────────────────────────────────────────────────
_AHP_CN = r"""
\subsection{层次分析法（AHP）权重确定}

层次分析法（Analytic Hierarchy Process，AHP）由 Saaty 于1980年提出，
通过构造判断矩阵对指标进行两两比较，将决策者的主观判断量化为权重向量，
适用于指标数量适中且专家意见可量化的评价场景。

\subsubsection{判断矩阵构建}

依据1--9标度法，对 <<N>> 个评价指标进行两两重要性比较，
构建判断矩阵 $\mathbf{A} \in \mathbb{R}^{<<N>>\times<<N>>}$，
如表\ref{tab:ahp_matrix_<<LABEL>>}所示。

<<MATRIX_TABLE>>

\subsubsection{权重计算}

采用算术平均法计算各指标归一化权重 $w_i$：
\begin{equation}
    w_i = \frac{1}{n}\sum_{j=1}^{n}\frac{a_{ij}}{\sum_{k=1}^{n}a_{kj}},
    \quad i=1,2,\ldots,n
    \label{eq:ahp_weight}
\end{equation}

权重计算结果见表\ref{tab:ahp_weights_<<LABEL>>}。

<<WEIGHT_TABLE>>

\subsubsection{一致性检验}

计算最大特征根 $\lambda_{\max}$、一致性指标 $CI$ 及一致性比率 $CR$：
\begin{equation}
    CI = \frac{\lambda_{\max} - n}{n - 1}, \qquad
    CR = \frac{CI}{RI}
    \label{eq:ahp_cr}
\end{equation}

其中随机一致性指标 $RI=<<RI>>$（$n=<<N>>$ 阶矩阵）。
经计算，$\lambda_{\max}=<<LAMBDA_MAX>>$，$CI=<<CI>>$，$CR=<<CR>>$。<<CONSIST_MSG>>
"""

_AHP_EN = r"""
\subsection{Analytic Hierarchy Process (AHP) Weight Determination}

AHP (Saaty, 1980) quantifies expert subjective judgments into weight vectors through
pairwise comparison of criteria using a 1--9 scale.

\subsubsection{Judgment Matrix}

The judgment matrix $\mathbf{A} \in \mathbb{R}^{<<N>>\times<<N>>}$ is shown
in Table~\ref{tab:ahp_matrix_<<LABEL>>}.

<<MATRIX_TABLE>>

\subsubsection{Weight Calculation}

\begin{equation}
    w_i = \frac{1}{n}\sum_{j=1}^{n}\frac{a_{ij}}{\sum_{k=1}^{n}a_{kj}}
    \label{eq:ahp_weight}
\end{equation}

Results are in Table~\ref{tab:ahp_weights_<<LABEL>>}.

<<WEIGHT_TABLE>>

\subsubsection{Consistency Check}

\begin{equation}
    CI = \frac{\lambda_{\max}-n}{n-1}, \quad CR = \frac{CI}{RI}
\end{equation}

With $RI=<<RI>>$ ($n=<<N>>$): $\lambda_{\max}=<<LAMBDA_MAX>>$,
$CI=<<CI>>$, $CR=<<CR>>$.<<CONSIST_MSG>>
"""

# ── 熵权法段落模板 ────────────────────────────────────────────────────────────
_ENTROPY_CN = r"""
\subsection{熵权法（Entropy Weight Method）权重确定}

信息熵权法基于 Shannon 信息熵理论，通过计算各指标的信息熵客观确定权重：
熵值越小，差异程度越大，权重越高，可有效规避主观因素干扰。

\subsubsection{概率矩阵构建}

设正向化、标准化后的决策矩阵为 $\mathbf{X}=(x_{ij})_{m\times n}$，
构造概率矩阵 $\mathbf{P}=(p_{ij})_{m\times n}$：
\begin{equation}
    p_{ij} = \frac{x_{ij}}{\displaystyle\sum_{i=1}^{m}x_{ij}},
    \quad i=1,\ldots,m;\ j=1,\ldots,n
    \label{eq:entropy_prob}
\end{equation}

\subsubsection{信息熵与差异系数}

\begin{equation}
    e_j = -\frac{1}{\ln m}\sum_{i=1}^{m}p_{ij}\ln p_{ij}
    \quad (\text{约定}\ 0\ln 0=0),
    \quad d_j = 1 - e_j
    \label{eq:entropy_ej}
\end{equation}

\subsubsection{权重确定}

\begin{equation}
    w_j = \frac{d_j}{\displaystyle\sum_{k=1}^{n}d_k},
    \quad j=1,2,\ldots,n
    \label{eq:entropy_wj}
\end{equation}

信息熵、差异系数与权重汇总见表\ref{tab:entropy_<<LABEL>>}。

<<ENTROPY_TABLE>>
"""

_ENTROPY_EN = r"""
\subsection{Entropy Weight Method (EWM) for Objective Weighting}

EWM assigns weights based on Shannon entropy: lower entropy implies greater
information content and thus a higher weight.

\begin{equation}
    e_j = -\frac{1}{\ln m}\sum_{i=1}^{m}p_{ij}\ln p_{ij},\quad
    d_j = 1-e_j,\quad
    w_j = \frac{d_j}{\sum_{k=1}^{n}d_k}
\end{equation}

Results are in Table~\ref{tab:entropy_<<LABEL>>}.

<<ENTROPY_TABLE>>
"""

# ── CRITIC 法段落模板 ─────────────────────────────────────────────────────────
_CRITIC_CN = r"""
\subsection{CRITIC 法权重确定}

CRITIC（Criteria Importance Through Inter-criteria Correlation）法综合考虑
指标的对比强度（标准差）与指标间的冲突性（相关系数），属全客观赋权方法。

\subsubsection{对比强度与冲突性}

\begin{equation}
    \sigma_j = \sqrt{\frac{1}{m-1}\sum_{i=1}^{m}(x_{ij}-\bar{x}_j)^2},
    \qquad
    c_j = \sum_{k=1}^{n}(1-r_{jk})
    \label{eq:critic_sigma}
\end{equation}

其中 $r_{jk}$ 为指标 $j$ 与 $k$ 之间的 Pearson 相关系数。

\subsubsection{信息量与权重}

\begin{equation}
    C_j = \sigma_j \cdot c_j, \qquad
    w_j = \frac{C_j}{\displaystyle\sum_{k=1}^{n}C_k}
    \label{eq:critic_wj}
\end{equation}

CRITIC 法计算结果见表\ref{tab:critic_<<LABEL>>}。

<<CRITIC_TABLE>>
"""

_CRITIC_EN = r"""
\subsection{CRITIC Method for Objective Weighting}

The CRITIC method combines variability (standard deviation) and conflict
(correlation-based) to determine objective weights.

\begin{equation}
    C_j = \sigma_j\cdot\sum_{k=1}^{n}(1-r_{jk}),\qquad
    w_j = \frac{C_j}{\sum_{k=1}^{n}C_k}
\end{equation}

Results are in Table~\ref{tab:critic_<<LABEL>>}.

<<CRITIC_TABLE>>
"""

# ── TOPSIS 段落模板 ───────────────────────────────────────────────────────────
_TOPSIS_CN = r"""
\subsection{TOPSIS 综合评价模型}

TOPSIS（Technique for Order Preference by Similarity to Ideal Solution）法
通过度量各评价对象与正理想解 $V^+$ 和负理想解 $V^-$ 的欧氏距离，
计算相对贴近度 $C_i$ 作为综合评价得分。

\subsubsection{加权规范化矩阵}

设正向化后的归一化矩阵为 $\mathbf{Z}=(z_{ij})_{m\times n}$，
加权规范化矩阵为：
\begin{equation}
    v_{ij} = w_j\cdot z_{ij},\quad i=1,\ldots,m;\ j=1,\ldots,n
\end{equation}

\subsubsection{正负理想解}

\begin{align}
    V^+ &= \bigl(\max_i v_{ij}\mid j\in\Omega^+\bigr)\cup
           \bigl(\min_i v_{ij}\mid j\in\Omega^-\bigr) \\
    V^- &= \bigl(\min_i v_{ij}\mid j\in\Omega^+\bigr)\cup
           \bigl(\max_i v_{ij}\mid j\in\Omega^-\bigr)
\end{align}

$\Omega^+$ 为正向指标集，$\Omega^-$ 为负向指标集。

\subsubsection{欧氏距离与相对贴近度}

\begin{equation}
    D_i^+ = \sqrt{\sum_{j=1}^{n}(v_{ij}-v_j^+)^2},\quad
    D_i^- = \sqrt{\sum_{j=1}^{n}(v_{ij}-v_j^-)^2},\quad
    C_i = \frac{D_i^-}{D_i^++D_i^-}\in[0,1]
\end{equation}

$C_i$ 越接近1，综合表现越优。
评价结果见表\ref{tab:topsis_<<LABEL>>}，
得分分布见图\ref{fig:ranking_<<LABEL>>}。

<<TOPSIS_TABLE>>
"""

_TOPSIS_EN = r"""
\subsection{TOPSIS Multi-Criteria Evaluation}

TOPSIS computes the relative closeness $C_i$ to the Positive Ideal Solution
($V^+$) and Negative Ideal Solution ($V^-$). A higher $C_i$ is better.

\begin{equation}
    C_i = \frac{D_i^-}{D_i^++D_i^-},\quad D_i^\pm =
    \sqrt{\sum_{j=1}^{n}(v_{ij}-v_j^\pm)^2}
\end{equation}

Results are in Table~\ref{tab:topsis_<<LABEL>>}.

<<TOPSIS_TABLE>>
"""

# ── GRA 段落模板 ──────────────────────────────────────────────────────────────
_GRA_CN = r"""
\subsection{灰色关联分析（GRA）综合评价}

灰色关联分析通过计算各评价对象序列与参考序列（理想方案）间的关联程度，
实现综合排序，对数据量要求低，适用于小样本不完全信息场景。

\subsubsection{灰色关联系数}

设参考序列 $x_0=(x_{0j})$，比较序列经正向化后为 $x_i=(x_{ij})$，
关联系数：
\begin{equation}
    \xi_{ij} = \frac{\Delta_{\min}+\rho\Delta_{\max}}
                    {\Delta_{ij}+\rho\Delta_{\max}},
    \quad \Delta_{ij}=|x_{0j}-x_{ij}|,\quad\rho=<<RHO>>
    \label{eq:gra_coeff}
\end{equation}

$\Delta_{\min}$、$\Delta_{\max}$ 分别为所有差值的全局最小值与最大值，
分辨系数 $\rho\in(0,1)$（通常取0.5）。

\subsubsection{加权综合关联度}

\begin{equation}
    r_i = \sum_{j=1}^{n}w_j\cdot\xi_{ij},\quad r_i\in(0,1]
    \label{eq:gra_ri}
\end{equation}

$r_i$ 越大表明综合表现越优。
灰色关联分析结果见表\ref{tab:gra_<<LABEL>>}。

<<GRA_TABLE>>
"""

_GRA_EN = r"""
\subsection{Grey Relational Analysis (GRA) Evaluation}

GRA measures the similarity between each alternative's data sequence and
the reference sequence (ideal solution), suitable for small-sample data.

\begin{equation}
    \xi_{ij}=\frac{\Delta_{\min}+\rho\Delta_{\max}}
                  {\Delta_{ij}+\rho\Delta_{\max}},\quad\rho=<<RHO>>,\qquad
    r_i=\sum_{j=1}^{n}w_j\xi_{ij}
\end{equation}

Results are in Table~\ref{tab:gra_<<LABEL>>}.

<<GRA_TABLE>>
"""

# ── VIKOR 段落模板 ────────────────────────────────────────────────────────────
_VIKOR_CN = r"""
\subsection{VIKOR 多准则妥协决策模型}

VIKOR 以"最大化群体效益、最小化个体遗憾"为原则，
在多准则约束下寻找妥协最优方案，综合指标 $Q_i$ 越小越优。

\subsubsection{群体效益值与最大遗憾值}

\begin{equation}
    S_i = \sum_{j=1}^{n}w_j\frac{f_j^*-f_{ij}}{f_j^*-f_j^-},\qquad
    R_i = \max_j\Bigl[w_j\frac{f_j^*-f_{ij}}{f_j^*-f_j^-}\Bigr]
\end{equation}

\subsubsection{综合指标 $Q$}

\begin{equation}
    Q_i = v\frac{S_i-S^*}{S^--S^*}+(1-v)\frac{R_i-R^*}{R^--R^*},
    \quad v=<<V_PARAM>>
    \label{eq:vikor_Q}
\end{equation}

$S^*=\min_i S_i$，$S^-=\max_i S_i$，$R^*=\min_i R_i$，$R^-=\max_i R_i$。

VIKOR 综合评价结果见表\ref{tab:vikor_<<LABEL>>}。

<<VIKOR_TABLE>>
"""

_VIKOR_EN = r"""
\subsection{VIKOR Multi-Criteria Compromise Ranking}

VIKOR ranks alternatives based on the compromise solution.
Smaller $Q_i$ is better.

\begin{equation}
    Q_i = v\frac{S_i-S^*}{S^--S^*}+(1-v)\frac{R_i-R^*}{R^--R^*},\quad v=<<V_PARAM>>
\end{equation}

Results are in Table~\ref{tab:vikor_<<LABEL>>}.

<<VIKOR_TABLE>>
"""

# ── 模糊综合评价段落模板 ──────────────────────────────────────────────────────
_FUZZY_CN = r"""
\subsection{模糊综合评价模型}

模糊综合评价法借助模糊数学处理评价中的不确定性与语义模糊性，
适用于含定性描述或等级划分的评价问题。

\subsubsection{建立评价集与隶属度矩阵}

评价集 $V=\{v_1,v_2,\ldots,v_k\}$（如：优、良、中、差），
因素集 $U=\{u_1,u_2,\ldots,u_n\}$。
依据隶属函数构建模糊关系矩阵 $\mathbf{R}\in[0,1]^{n\times k}$。

\subsubsection{模糊综合评判}

\begin{equation}
    \mathbf{B} = \mathbf{W}\circ\mathbf{R},
    \quad \mathbf{W}=(w_1,\ldots,w_n)
    \label{eq:fuzzy_B}
\end{equation}

$\circ$ 为模糊合成算子（本文采用加权平均型 $M(\cdot,+)$）。
依据最大隶属度原则确定综合评价等级。

模糊综合评价结果见表\ref{tab:fuzzy_<<LABEL>>}。

<<FUZZY_TABLE>>
"""

# ── PCA 段落模板 ──────────────────────────────────────────────────────────────
_PCA_CN = r"""
\subsection{主成分分析（PCA）降维预处理}

为消除原始指标间的多重共线性，在正式评价前采用 PCA 对指标体系降维，
提取累计方差贡献率超过 <<THRESHOLD>>\% 的前 <<N_COMP>> 个主成分。

\subsubsection{主成分提取}

对标准化矩阵 $\mathbf{Z}$ 计算协方差矩阵并特征分解，
主成分特征值与方差贡献率见表\ref{tab:pca_<<LABEL>>}（见图\ref{fig:pca_<<LABEL>>}）。

<<PCA_TABLE>>

\subsubsection{综合得分}

以方差贡献率为权重，计算降维后综合得分：
\begin{equation}
    F_i = \frac{\displaystyle\sum_{l=1}^{<<N_COMP>>}\lambda_l\cdot Z_{il}}
               {\displaystyle\sum_{l=1}^{<<N_COMP>>}\lambda_l}
    \label{eq:pca_score}
\end{equation}
"""

_PCA_EN = r"""
\subsection{Principal Component Analysis (PCA) for Dimensionality Reduction}

To address multicollinearity, PCA extracts the first <<N_COMP>> principal
components explaining more than <<THRESHOLD>>\% of total variance.

<<PCA_TABLE>>

\begin{equation}
    F_i = \frac{\sum_{l=1}^{<<N_COMP>>}\lambda_l Z_{il}}
               {\sum_{l=1}^{<<N_COMP>>}\lambda_l}
\end{equation}
"""

# ── 灵敏度分析段落模板 ────────────────────────────────────────────────────────
_SENSITIVITY_CN = r"""
\subsection{权重灵敏度分析}

采用单因素敏感性分析法（OAT）验证综合评价结果对权重扰动的稳健性：
固定其余指标权重不变，对每个指标权重在
$[-<<DELTA>>\%,+<<DELTA>>\%]$ 范围内以 $1\%$ 步长扰动，
观察各评价对象综合排名的变动情况（见图\ref{fig:sensitivity_<<LABEL>>}）。

<<ROBUST_MSG>>
"""

_SENSITIVITY_EN = r"""
\subsection{Weight Sensitivity Analysis}

One-At-a-Time (OAT) analysis perturbs each weight within
$[\!-<<DELTA>>\%,+<<DELTA>>\%\!]$ to assess ranking robustness
(see Figure~\ref{fig:sensitivity_<<LABEL>>}).

<<ROBUST_MSG>>
"""

# ── 综合结果段落模板 ──────────────────────────────────────────────────────────
_RESULT_CN = r"""
\subsection{综合评价结果与分析}

综合运用 <<ALGO_STR>> 方法，
对 <<N_OBJ>> 个评价对象进行了全面系统的评价分析。
最终综合得分与排名如表\ref{tab:final_result_<<LABEL>>}所示，
得分分布见图\ref{fig:ranking_<<LABEL>>}（排序图）和图\ref{fig:radar_<<LABEL>>}（雷达图）。

<<RESULT_TABLE>>

\subsubsection{结果分析}

从综合评分来看，\textbf{<<TOP_OBJ>>} 综合表现最优，得分为 $<<TOP_SCORE>>$；
\textbf{<<BOT_OBJ>>} 综合表现相对较弱，得分为 $<<BOT_SCORE>>$。
灵敏度分析验证了权重设定的合理性，评价结果稳健可靠。
"""

_RESULT_EN = r"""
\subsection{Comprehensive Evaluation Results}

Combining <<ALGO_STR>>, a comprehensive evaluation of
<<N_OBJ>> objects is conducted. Results are in
Table~\ref{tab:final_result_<<LABEL>>} with visualization in
Figure~\ref{fig:ranking_<<LABEL>>} and Figure~\ref{fig:radar_<<LABEL>>}.

<<RESULT_TABLE>>

\textbf{<<TOP_OBJ>>} achieves the highest score of $<<TOP_SCORE>>$;
\textbf{<<BOT_OBJ>>} ranks lowest with score $<<BOT_SCORE>>$.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  主类
# ══════════════════════════════════════════════════════════════════════════════

class LatexBuilder:
    """
    LaTeX 报告生成器

    设计原则
    --------
    1. 每次调用 add_*_section() 向内部队列追加一个 LaTeX 段落字符串。
    2. build_document() 将所有段落拼接并套入文档骨架，返回完整源码。
    3. save() 将结果写入文件。
    4. 使用 <<VAR>> 占位符渲染模板，完全兼容 LaTeX 语法（无转义冲突）。

    Parameters
    ----------
    language : str
        输出语言，``"cn"`` 或 ``"en"``。
    output_dir : str or Path
        LaTeX 文件输出目录。
    fig_dir : str
        图片引用的相对路径前缀（在 LaTeX 中使用）。
    decimal_places : int
        数值显示的小数位数。
    """

    # AHP 随机一致性指标 RI（阶数 1~15）
    _RI = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90,  5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
        11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59,
    }

    def __init__(
        self,
        language: str = "cn",
        output_dir: Union[str, Path] = "output/reports",
        fig_dir: str = "figures",
        decimal_places: int = 4,
    ) -> None:
        self.language = language.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = fig_dir
        self.dp = decimal_places

        self._sections: List[str] = []       # 已添加的 LaTeX 段落
        self._used_algorithms: List[str] = []  # 记录已使用的算法（用于自动摘要）
        self._table_counter: int = 0
        self._fig_counter: int = 0

        logger.info(
            "LatexBuilder 初始化 [language=%s, output_dir=%s]",
            language, output_dir,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 内部渲染工具
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _render(template: str, **kwargs: Any) -> str:
        """将模板中的 <<KEY>> 占位符替换为对应值（不影响 LaTeX 语法）。"""
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"<<{key}>>", str(value))
        return result

    def _fmt(self, value: float) -> str:
        """将数值格式化为指定小数位的字符串。"""
        return f"{value:.{self.dp}f}"

    # ──────────────────────────────────────────────────────────────────────────
    # 表格生成工具
    # ──────────────────────────────────────────────────────────────────────────

    def _df_to_booktabs(
        self,
        df: pd.DataFrame,
        caption: str,
        label: str,
        position: str = "htbp",
        fontsize: str = "small",
        col_format: Optional[str] = None,
    ) -> str:
        """
        将 DataFrame 转换为 booktabs 风格 LaTeX 表格。

        Parameters
        ----------
        df : pd.DataFrame
            数据框（已完成格式化，各列均为字符串或数值）。
        caption : str
            表格标题。
        label : str
            LaTeX label（函数内自动添加 ``tab:`` 前缀）。
        position : str
            浮动体位置参数。
        fontsize : str
            表格字体大小命令（不带反斜线）。
        col_format : str, optional
            自定义列格式，默认第一列左对齐、其余居中。

        Returns
        -------
        str
            完整 LaTeX 表格代码。
        """
        self._table_counter += 1
        n_cols = len(df.columns)
        if col_format is None:
            col_format = "l" + "c" * (n_cols - 1)

        lines: List[str] = [
            rf"\begin{{table}}[{position}]",
            r"    \centering",
            rf"    \{fontsize}",
            rf"    \caption{{{caption}}}",
            rf"    \label{{tab:{label}}}",
            rf"    \begin{{tabular}}{{{col_format}}}",
            r"    \toprule",
        ]
        # 表头
        header = " & ".join(rf"\textbf{{{c}}}" for c in df.columns)
        lines += [f"    {header} \\\\", r"    \midrule"]
        # 数据行
        for _, row in df.iterrows():
            row_str = " & ".join(str(v) for v in row.values)
            lines.append(f"    {row_str} \\\\")
        lines += [
            r"    \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _matrix_to_booktabs(
        self,
        matrix: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        caption: str,
        label: str,
        position: str = "htbp",
    ) -> str:
        """将方阵（含行列标签）转换为 booktabs 表格。"""
        self._table_counter += 1
        n = matrix.shape[0]
        col_format = "l" + "c" * n

        lines: List[str] = [
            rf"\begin{{table}}[{position}]",
            r"    \centering",
            r"    \small",
            rf"    \caption{{{caption}}}",
            rf"    \label{{tab:{label}}}",
            rf"    \begin{{tabular}}{{{col_format}}}",
            r"    \toprule",
        ]
        header = " & " + " & ".join(rf"\textbf{{{c}}}" for c in col_labels)
        lines += [f"    {header} \\\\", r"    \midrule"]
        for i, rl in enumerate(row_labels):
            vals = " & ".join(self._fmt(matrix[i, j]) for j in range(n))
            lines.append(rf"    \textbf{{{rl}}} & {vals} \\")
        lines += [
            r"    \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # 图片插入工具
    # ──────────────────────────────────────────────────────────────────────────

    def add_figure(
        self,
        filename: str,
        caption: str,
        label: str,
        width: str = r"0.85\linewidth",
        position: str = "htbp",
    ) -> None:
        """
        插入单幅图片。

        Parameters
        ----------
        filename : str
            图片文件名（不含目录前缀，路径由 ``fig_dir`` 控制）。
        caption : str
            图题文字。
        label : str
            LaTeX 交叉引用标签（自动添加 ``fig:`` 前缀）。
        width : str
            宽度（LaTeX 宽度表达式）。
        position : str
            浮动体位置。
        """
        self._fig_counter += 1
        block = "\n".join([
            rf"\begin{{figure}}[{position}]",
            r"    \centering",
            rf"    \includegraphics[width={width}]{{{self.fig_dir}/{filename}}}",
            rf"    \caption{{{caption}}}",
            rf"    \label{{fig:{label}}}",
            r"\end{figure}",
        ])
        self._sections.append(block)

    def add_double_figure(
        self,
        file1: str, caption1: str, label1: str,
        file2: str, caption2: str, label2: str,
        main_caption: str = "",
        main_label: str = "double_fig",
        position: str = "htbp",
    ) -> None:
        """插入并排双图（使用 subcaption 包）。"""
        self._fig_counter += 2
        block = "\n".join([
            rf"\begin{{figure}}[{position}]",
            r"    \centering",
            r"    \begin{subfigure}[b]{0.48\textwidth}",
            r"        \centering",
            rf"        \includegraphics[width=\textwidth]{{{self.fig_dir}/{file1}}}",
            rf"        \caption{{{caption1}}}",
            rf"        \label{{fig:{label1}}}",
            r"    \end{subfigure}",
            r"    \hfill",
            r"    \begin{subfigure}[b]{0.48\textwidth}",
            r"        \centering",
            rf"        \includegraphics[width=\textwidth]{{{self.fig_dir}/{file2}}}",
            rf"        \caption{{{caption2}}}",
            rf"        \label{{fig:{label2}}}",
            r"    \end{subfigure}",
            rf"    \caption{{{main_caption}}}",
            rf"    \label{{fig:{main_label}}}",
            r"\end{figure}",
        ])
        self._sections.append(block)

    # ──────────────────────────────────────────────────────────────────────────
    # 自定义段落
    # ──────────────────────────────────────────────────────────────────────────

    def add_section(self, title: str, content: str, level: int = 2) -> None:
        """
        添加自定义任意章节。

        Parameters
        ----------
        title : str
            章节标题。
        content : str
            LaTeX 正文内容。
        level : int
            标题层级（1=section, 2=subsection, 3=subsubsection）。
        """
        cmd = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection"}.get(level, r"\subsection")
        self._sections.append(f"{cmd}{{{title}}}\n\n{content}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 算法专属段落
    # ──────────────────────────────────────────────────────────────────────────

    def add_ahp_section(self, result: Dict[str, Any], label: str = "ahp") -> None:
        """
        添加 AHP 层次分析法段落。

        Parameters
        ----------
        result : dict
            AHP 计算结果，必须包含：

            * ``indicators`` : List[str] — 指标名称列表
            * ``weights``    : array-like — 归一化权重向量
            * ``judgment_matrix`` : 2-D array-like — 判断矩阵
            * ``lambda_max`` : float — 最大特征根
            * ``ci``         : float — 一致性指标
            * ``cr``         : float — 一致性比率

        label : str
            用于表格 / 图片 label 唯一标识。
        """
        indicators = list(result.get("indicators", []))
        weights    = np.asarray(result.get("weights", []))
        jm         = np.asarray(result.get("judgment_matrix", [[]]))
        lam_max    = float(result.get("lambda_max", 0.0))
        ci         = float(result.get("ci", 0.0))
        cr         = float(result.get("cr", 0.0))
        n          = len(indicators)
        ri         = self._RI.get(n, 1.49)

        # 判断矩阵表格
        matrix_table = self._matrix_to_booktabs(
            jm, indicators, indicators,
            caption="AHP 判断矩阵" if self.language == "cn" else "AHP Judgment Matrix",
            label=f"ahp_matrix_{label}",
        )

        # 权重汇总表格
        w_df = pd.DataFrame({
            ("指标" if self.language == "cn" else "Indicator"): indicators,
            r"权重 $w_i$" if self.language == "cn" else r"Weight $w_i$": [
                self._fmt(w) for w in weights
            ],
        })
        weight_table = self._df_to_booktabs(
            w_df,
            caption="AHP 权重计算结果" if self.language == "cn" else "AHP Weight Results",
            label=f"ahp_weights_{label}",
        )

        # 一致性结论
        if self.language == "cn":
            consist_msg = (
                r" 判断矩阵满足一致性要求（$CR<0.1$），权重结果可信。"
                if cr < 0.1 else
                r" \textbf{警告：}$CR\geq0.1$，判断矩阵不满足一致性要求，建议重新调整。"
            )
        else:
            consist_msg = (
                r" The judgment matrix satisfies consistency ($CR<0.1$)."
                if cr < 0.1 else
                r" \textbf{Warning:} $CR\geq0.1$; please revise the judgment matrix."
            )

        template = _AHP_CN if self.language == "cn" else _AHP_EN
        section = self._render(
            template,
            N=n,
            LABEL=label,
            RI=ri,
            LAMBDA_MAX=self._fmt(lam_max),
            CI=self._fmt(ci),
            CR=self._fmt(cr),
            MATRIX_TABLE=matrix_table,
            WEIGHT_TABLE=weight_table,
            CONSIST_MSG=consist_msg,
        )
        self._sections.append(section)
        self._used_algorithms.append("AHP")
        logger.debug("AHP 段落已添加 [label=%s]", label)

    def add_entropy_section(self, result: Dict[str, Any], label: str = "entropy") -> None:
        """
        添加熵权法段落。

        Parameters
        ----------
        result : dict
            熵权法结果，必须包含：

            * ``indicators`` : List[str]
            * ``weights``    : array-like
            * ``entropy``    : array-like — 信息熵 $e_j$
            * ``diversity``  : array-like — 差异系数 $d_j$（可选，默认 1-entropy）
        """
        indicators = list(result.get("indicators", []))
        weights    = np.asarray(result.get("weights", []))
        entropy    = np.asarray(result.get("entropy", np.zeros(len(indicators))))
        diversity  = np.asarray(result.get("diversity", 1.0 - entropy))

        df = pd.DataFrame({
            ("指标" if self.language == "cn" else "Indicator"): indicators,
            r"信息熵 $e_j$"      if self.language == "cn" else r"Entropy $e_j$":
                [self._fmt(v) for v in entropy],
            r"差异系数 $d_j$"    if self.language == "cn" else r"Diversity $d_j$":
                [self._fmt(v) for v in diversity],
            r"权重 $w_j$"        if self.language == "cn" else r"Weight $w_j$":
                [self._fmt(v) for v in weights],
        })
        table = self._df_to_booktabs(
            df,
            caption="熵权法计算结果" if self.language == "cn" else "Entropy Weight Results",
            label=f"entropy_{label}",
        )

        template = _ENTROPY_CN if self.language == "cn" else _ENTROPY_EN
        section = self._render(template, LABEL=label, ENTROPY_TABLE=table)
        self._sections.append(section)
        self._used_algorithms.append("熵权法" if self.language == "cn" else "EWM")
        logger.debug("熵权法段落已添加 [label=%s]", label)

    def add_critic_section(self, result: Dict[str, Any], label: str = "critic") -> None:
        """
        添加 CRITIC 法段落。

        Parameters
        ----------
        result : dict
            * ``indicators``   : List[str]
            * ``weights``      : array-like
            * ``std_devs``     : array-like — 各指标标准差
            * ``conflicts``    : array-like — 冲突性指标
            * ``info_amounts`` : array-like — 信息量 $C_j$
        """
        indicators   = list(result.get("indicators", []))
        weights      = np.asarray(result.get("weights", []))
        std_devs     = np.asarray(result.get("std_devs",     np.zeros(len(indicators))))
        conflicts    = np.asarray(result.get("conflicts",    np.zeros(len(indicators))))
        info_amounts = np.asarray(result.get("info_amounts", std_devs * conflicts))

        df = pd.DataFrame({
            ("指标"    if self.language == "cn" else "Indicator"):  indicators,
            r"$\sigma_j$":   [self._fmt(v) for v in std_devs],
            r"$c_j$":        [self._fmt(v) for v in conflicts],
            r"$C_j$":        [self._fmt(v) for v in info_amounts],
            (r"权重 $w_j$" if self.language == "cn" else r"Weight $w_j$"):
                             [self._fmt(v) for v in weights],
        })
        table = self._df_to_booktabs(
            df,
            caption="CRITIC 法计算结果" if self.language == "cn" else "CRITIC Method Results",
            label=f"critic_{label}",
        )

        template = _CRITIC_CN if self.language == "cn" else _CRITIC_EN
        section = self._render(template, LABEL=label, CRITIC_TABLE=table)
        self._sections.append(section)
        self._used_algorithms.append("CRITIC")
        logger.debug("CRITIC 段落已添加 [label=%s]", label)

    def add_topsis_section(self, result: Dict[str, Any], label: str = "topsis") -> None:
        """
        添加 TOPSIS 综合评价段落。

        Parameters
        ----------
        result : dict
            * ``objects``    : List[str] — 评价对象名称
            * ``scores``     : array-like — 相对贴近度 $C_i$
            * ``ranking``    : array-like — 排名（1-based）
            * ``d_positive`` : array-like — $D_i^+$（可选）
            * ``d_negative`` : array-like — $D_i^-$（可选）
        """
        objects  = list(result.get("objects", []))
        scores   = np.asarray(result.get("scores", []))
        ranking  = list(result.get("ranking", np.argsort(-scores) + 1))
        d_pos    = np.asarray(result.get("d_positive", np.zeros(len(objects))))
        d_neg    = np.asarray(result.get("d_negative", np.zeros(len(objects))))

        df = pd.DataFrame({
            ("评价对象"    if self.language == "cn" else "Object"):        objects,
            r"$D_i^+$":  [self._fmt(v) for v in d_pos],
            r"$D_i^-$":  [self._fmt(v) for v in d_neg],
            (r"相对贴近度 $C_i$" if self.language == "cn" else r"Closeness $C_i$"):
                         [self._fmt(v) for v in scores],
            ("排名"       if self.language == "cn" else "Rank"): ranking,
        })
        rank_col = "排名" if self.language == "cn" else "Rank"
        df = df.sort_values(rank_col).reset_index(drop=True)

        table = self._df_to_booktabs(
            df,
            caption="TOPSIS 综合评价结果" if self.language == "cn" else "TOPSIS Evaluation Results",
            label=f"topsis_{label}",
        )

        template = _TOPSIS_CN if self.language == "cn" else _TOPSIS_EN
        section = self._render(template, LABEL=label, TOPSIS_TABLE=table)
        self._sections.append(section)
        self._used_algorithms.append("TOPSIS")
        logger.debug("TOPSIS 段落已添加 [label=%s]", label)

    def add_gra_section(self, result: Dict[str, Any], label: str = "gra") -> None:
        """
        添加灰色关联分析段落。

        Parameters
        ----------
        result : dict
            * ``objects``       : List[str] — 评价对象名称
            * ``scores``        : array-like — 加权综合关联度 $r_i$
            * ``ranking``       : array-like — 排名（1-based）
            * ``rho``           : float — 分辨系数（默认 0.5）
            * ``coeff_matrix``  : 2-D array-like — 关联系数矩阵（可选）
            * ``indicators``    : List[str] — 指标名（可选）
        """
        objects     = list(result.get("objects", []))
        scores      = np.asarray(result.get("scores", []))
        ranking     = list(result.get("ranking", np.argsort(-scores) + 1))
        rho         = float(result.get("rho", 0.5))
        indicators  = list(result.get("indicators", []))
        coeff_mat   = result.get("coeff_matrix", None)

        # 综合结果表
        df = pd.DataFrame({
            ("评价对象" if self.language == "cn" else "Object"): objects,
            (r"关联度 $r_i$" if self.language == "cn" else r"Grey Degree $r_i$"):
                [self._fmt(v) for v in scores],
            ("排名" if self.language == "cn" else "Rank"): ranking,
        })
        rank_col = "排名" if self.language == "cn" else "Rank"
        df = df.sort_values(rank_col).reset_index(drop=True)

        result_table = self._df_to_booktabs(
            df,
            caption="灰色关联分析综合结果" if self.language == "cn"
                    else "Grey Relational Analysis Results",
            label=f"gra_{label}",
        )

        # 关联系数矩阵（可选）
        coeff_table_str = ""
        if coeff_mat is not None and len(indicators) > 0:
            coeff_arr = np.asarray(coeff_mat)
            coeff_df_data = {
                ("评价对象" if self.language == "cn" else "Object"): objects
            }
            for j, ind in enumerate(indicators):
                coeff_df_data[ind] = [self._fmt(coeff_arr[i, j])
                                      for i in range(len(objects))]
            coeff_df = pd.DataFrame(coeff_df_data)
            cap = ("灰色关联系数矩阵" if self.language == "cn"
                   else "Grey Relational Coefficient Matrix")
            coeff_table_str = self._df_to_booktabs(
                coeff_df, caption=cap, label=f"gra_coeff_{label}"
            )

        template = _GRA_CN if self.language == "cn" else _GRA_EN
        section = self._render(
            template,
            LABEL=label,
            RHO=self._fmt(rho),
            GRA_TABLE=result_table + ("\n\n" + coeff_table_str if coeff_table_str else ""),
        )
        self._sections.append(section)
        self._used_algorithms.append(
            "灰色关联分析" if self.language == "cn" else "GRA"
        )
        logger.debug("GRA 段落已添加 [label=%s]", label)

    def add_vikor_section(self, result: Dict[str, Any], label: str = "vikor") -> None:
        """
        添加 VIKOR 综合评价段落。

        Parameters
        ----------
        result : dict
            * ``objects``  : List[str]
            * ``S``        : array-like — 群体效益值
            * ``R``        : array-like — 个体最大遗憾值
            * ``Q``        : array-like — 综合指标
            * ``ranking``  : array-like — 排名
            * ``v``        : float — 决策偏好系数（默认 0.5）
        """
        objects = list(result.get("objects", []))
        S_vals  = np.asarray(result.get("S", []))
        R_vals  = np.asarray(result.get("R", []))
        Q_vals  = np.asarray(result.get("Q", []))
        ranking = list(result.get("ranking", np.argsort(Q_vals) + 1))
        v_param = float(result.get("v", 0.5))

        df = pd.DataFrame({
            ("评价对象" if self.language == "cn" else "Object"): objects,
            r"$S_i$": [self._fmt(v) for v in S_vals],
            r"$R_i$": [self._fmt(v) for v in R_vals],
            r"$Q_i$": [self._fmt(v) for v in Q_vals],
            ("排名" if self.language == "cn" else "Rank"): ranking,
        })
        rank_col = "排名" if self.language == "cn" else "Rank"
        df = df.sort_values(rank_col).reset_index(drop=True)

        table = self._df_to_booktabs(
            df,
            caption="VIKOR 综合评价结果" if self.language == "cn"
                    else "VIKOR Evaluation Results",
            label=f"vikor_{label}",
        )

        template = _VIKOR_CN if self.language == "cn" else _VIKOR_EN
        section = self._render(
            template, LABEL=label, V_PARAM=self._fmt(v_param), VIKOR_TABLE=table
        )
        self._sections.append(section)
        self._used_algorithms.append("VIKOR")
        logger.debug("VIKOR 段落已添加 [label=%s]", label)

    def add_fuzzy_section(self, result: Dict[str, Any], label: str = "fuzzy") -> None:
        """
        添加模糊综合评价段落。

        Parameters
        ----------
        result : dict
            * ``objects``      : List[str]
            * ``levels``       : List[str] — 评语集，如 ['优','良','中','差']
            * ``B_matrix``     : 2-D array-like — 模糊评判矩阵（对象 × 等级）
            * ``final_levels`` : List[str] — 各对象最终评价等级
            * ``scores``       : array-like — 量化综合分（可选）
        """
        objects     = list(result.get("objects", []))
        levels      = list(result.get("levels", []))
        B_mat       = np.asarray(result.get("B_matrix", [[]]))
        final_lvls  = list(result.get("final_levels", []))
        scores      = result.get("scores", None)

        # B 矩阵表格
        cols = {("评价对象" if self.language == "cn" else "Object"): objects}
        for j, lv in enumerate(levels):
            cols[lv] = [self._fmt(B_mat[i, j]) for i in range(len(objects))]
        if final_lvls:
            cols[("综合等级" if self.language == "cn" else "Level")] = final_lvls
        if scores is not None:
            cols[("综合分" if self.language == "cn" else "Score")] = [
                self._fmt(s) for s in scores
            ]

        df = pd.DataFrame(cols)
        table = self._df_to_booktabs(
            df,
            caption="模糊综合评价结果" if self.language == "cn"
                    else "Fuzzy Comprehensive Evaluation Results",
            label=f"fuzzy_{label}",
        )

        template = _FUZZY_CN
        section = self._render(template, LABEL=label, FUZZY_TABLE=table)
        self._sections.append(section)
        self._used_algorithms.append(
            "模糊综合评价" if self.language == "cn" else "Fuzzy Evaluation"
        )
        logger.debug("模糊综合评价段落已添加 [label=%s]", label)

    def add_pca_section(self, result: Dict[str, Any], label: str = "pca") -> None:
        """
        添加 PCA 降维预处理段落。

        Parameters
        ----------
        result : dict
            * ``n_components``      : int — 保留主成分数
            * ``threshold``         : float — 累计方差贡献率阈值（%）
            * ``eigenvalues``       : array-like — 特征值
            * ``explained_var``     : array-like — 各主成分方差贡献率（%）
            * ``cumulative_var``    : array-like — 累计方差贡献率（%）
            * ``loadings``          : 2-D array-like — 因子载荷矩阵（可选）
            * ``indicators``        : List[str] — 原始指标名（可选）
        """
        n_comp      = int(result.get("n_components", 2))
        threshold   = float(result.get("threshold", 85.0))
        eigenvalues = np.asarray(result.get("eigenvalues", []))
        exp_var     = np.asarray(result.get("explained_var", []))
        cum_var     = np.asarray(result.get("cumulative_var", []))

        comp_names = [
            (f"主成分{i+1}" if self.language == "cn" else f"PC{i+1}")
            for i in range(len(eigenvalues))
        ]
        df = pd.DataFrame({
            ("主成分" if self.language == "cn" else "Component"): comp_names,
            ("特征值" if self.language == "cn" else "Eigenvalue"):
                [self._fmt(v) for v in eigenvalues],
            ("方差贡献率 (\%)" if self.language == "cn" else "Var. (\%)"):
                [self._fmt(v) for v in exp_var],
            ("累计方差贡献率 (\%)" if self.language == "cn" else "Cumul. Var. (\%)"):
                [self._fmt(v) for v in cum_var],
        })
        table = self._df_to_booktabs(
            df,
            caption="主成分分析方差解释" if self.language == "cn"
                    else "PCA Variance Explained",
            label=f"pca_{label}",
        )

        template = _PCA_CN if self.language == "cn" else _PCA_EN
        section = self._render(
            template,
            LABEL=label,
            N_COMP=n_comp,
            THRESHOLD=self._fmt(threshold),
            PCA_TABLE=table,
        )
        self._sections.append(section)
        self._used_algorithms.append("PCA")
        logger.debug("PCA 段落已添加 [label=%s]", label)

    def add_sensitivity_section(
        self,
        result: Dict[str, Any],
        label: str = "sens",
    ) -> None:
        """
        添加权重灵敏度分析段落。

        Parameters
        ----------
        result : dict
            * ``delta``       : float — 扰动幅度百分比（默认 20）
            * ``is_robust``   : bool — 排名是否稳健
            * ``robust_msg``  : str — 稳健性文字描述（可选，覆盖自动生成）
        """
        delta   = float(result.get("delta", 20.0))
        robust  = bool(result.get("is_robust", True))

        if "robust_msg" in result:
            robust_msg = result["robust_msg"]
        else:
            if self.language == "cn":
                robust_msg = (
                    r"分析结果表明，在权重扰动范围内各评价对象的综合排名保持稳定，"
                    r"评价模型具有良好的稳健性。"
                    if robust else
                    r"\textbf{注意：}部分权重配置下综合排名发生变动，"
                    r"建议审慎解读评价结果，并结合领域专家意见进行综合研判。"
                )
            else:
                robust_msg = (
                    "Results show that rankings remain stable under weight perturbations,"
                    " confirming model robustness."
                    if robust else
                    r"\textbf{Note:} Some rankings change under perturbation; "
                    "interpret results cautiously."
                )

        template = _SENSITIVITY_CN if self.language == "cn" else _SENSITIVITY_EN
        section = self._render(
            template, LABEL=label,
            DELTA=self._fmt(delta),
            ROBUST_MSG=robust_msg,
        )
        self._sections.append(section)
        logger.debug("灵敏度分析段落已添加 [label=%s]", label)

    def add_combination_weight_section(
        self,
        result: Dict[str, Any],
        label: str = "comb_weight",
    ) -> None:
        """
        添加组合赋权段落。

        Parameters
        ----------
        result : dict
            * ``indicators``        : List[str]
            * ``subjective_weights``: array-like — 主观权重
            * ``objective_weights`` : array-like — 客观权重
            * ``combined_weights``  : array-like — 组合权重
            * ``method``            : str — 组合方式（如 "乘法合成" / "线性加权"）
            * ``alpha``             : float — 线性加权系数（可选）
        """
        indicators  = list(result.get("indicators", []))
        subj_w      = np.asarray(result.get("subjective_weights", []))
        obj_w       = np.asarray(result.get("objective_weights", []))
        comb_w      = np.asarray(result.get("combined_weights", []))
        method      = str(result.get("method", "乘法合成"))
        alpha       = result.get("alpha", None)

        df = pd.DataFrame({
            ("指标" if self.language == "cn" else "Indicator"): indicators,
            ("主观权重" if self.language == "cn" else "Subj. Weight"):
                [self._fmt(v) for v in subj_w],
            ("客观权重" if self.language == "cn" else "Obj. Weight"):
                [self._fmt(v) for v in obj_w],
            ("组合权重" if self.language == "cn" else "Combined Weight"):
                [self._fmt(v) for v in comb_w],
        })
        cap = (f"组合赋权结果（{method}）" if self.language == "cn"
               else f"Combined Weights ({method})")
        table = self._df_to_booktabs(df, caption=cap, label=f"comb_{label}")

        # 组合方式公式
        if "乘法" in method or "multiplicative" in method.lower():
            formula = (
                r"\begin{equation}"
                r"\tilde{w}_j = \frac{w_j^s \cdot w_j^o}{\sum_{k=1}^n w_k^s \cdot w_k^o}"
                r"\end{equation}"
            )
        elif alpha is not None:
            formula = self._render(
                r"\begin{equation}"
                r"\tilde{w}_j = <<ALPHA>> w_j^s + (1-<<ALPHA>>) w_j^o"
                r"\end{equation}",
                ALPHA=self._fmt(float(alpha)),
            )
        else:
            formula = (
                r"\begin{equation}"
                r"\tilde{w}_j = \alpha w_j^s + (1-\alpha) w_j^o"
                r"\end{equation}"
            )

        intro = (
            f"采用{method}将主观权重与客观权重进行组合，"
            "平衡主观经验与数据客观信息，计算公式如下：\n\n"
            if self.language == "cn"
            else f"The {method} approach combines subjective and objective weights:\n\n"
        )
        content = intro + formula + "\n\n" + "组合权重结果见下表。\n\n" + table
        self._sections.append(
            (r"\subsection{组合赋权结果}" if self.language == "cn"
             else r"\subsection{Combined Weighting Results}") + "\n\n" + content
        )
        self._used_algorithms.append(
            f"组合赋权({method})" if self.language == "cn" else f"Combined ({method})"
        )
        logger.debug("组合赋权段落已添加 [label=%s]", label)

    def add_final_result_section(
        self,
        result: Dict[str, Any],
        label: str = "final",
    ) -> None:
        """
        添加综合结果与分析总结段落。

        Parameters
        ----------
        result : dict
            * ``objects``   : List[str]
            * ``scores``    : array-like — 综合得分
            * ``ranking``   : array-like — 最终排名
        """
        objects = list(result.get("objects", []))
        scores  = np.asarray(result.get("scores", []))
        ranking = list(result.get("ranking", np.argsort(-scores) + 1))

        rank_arr = np.asarray(ranking)
        top_idx  = int(np.argmin(rank_arr))
        bot_idx  = int(np.argmax(rank_arr))

        df = pd.DataFrame({
            ("评价对象" if self.language == "cn" else "Object"): objects,
            ("综合得分" if self.language == "cn" else "Score"):
                [self._fmt(v) for v in scores],
            ("排名" if self.language == "cn" else "Rank"): ranking,
        })
        rank_col = "排名" if self.language == "cn" else "Rank"
        df = df.sort_values(rank_col).reset_index(drop=True)

        table = self._df_to_booktabs(
            df,
            caption="综合评价最终结果" if self.language == "cn"
                    else "Final Comprehensive Evaluation Results",
            label=f"final_result_{label}",
        )

        algo_str = "、".join(self._used_algorithms) if self.language == "cn" \
                   else " + ".join(self._used_algorithms)

        template = _RESULT_CN if self.language == "cn" else _RESULT_EN
        section = self._render(
            template,
            LABEL=label,
            ALGO_STR=algo_str,
            N_OBJ=len(objects),
            TOP_OBJ=objects[top_idx],
            TOP_SCORE=self._fmt(scores[top_idx]),
            BOT_OBJ=objects[bot_idx],
            BOT_SCORE=self._fmt(scores[bot_idx]),
            RESULT_TABLE=table,
        )
        self._sections.append(section)
        logger.debug("综合结果段落已添加 [label=%s]", label)

    # ──────────────────────────────────────────────────────────────────────────
    # 文档组装与保存
    # ──────────────────────────────────────────────────────────────────────────

    def build_document(
        self,
        title: str = "综合评价模型分析报告",
        author: str = "AutoEval-Modeling",
        abstract: str = "",
        date: Optional[str] = None,
    ) -> str:
        """
        组装完整 LaTeX 文档字符串。

        Parameters
        ----------
        title : str
            文档标题。
        author : str
            作者。
        abstract : str
            摘要正文（LaTeX 语法）。
        date : str, optional
            日期字符串，默认当天日期。

        Returns
        -------
        str
            完整的 LaTeX 文档源码。
        """
        if date is None:
            date = datetime.now().strftime("%Y年%m月%d日"
                                           if self.language == "cn"
                                           else "%B %d, %Y")
        if not abstract:
            abstract = (
                "本报告基于 AutoEval-Modeling 自动工作流，"
                f"综合运用 {', '.join(self._used_algorithms) if self._used_algorithms else '多种评价方法'}，"
                "对评价对象进行了全面系统的量化分析，给出综合排名与评价结论。"
                if self.language == "cn"
                else
                "This report is auto-generated by AutoEval-Modeling. "
                f"Methods applied: {', '.join(self._used_algorithms) or 'multiple'}."
            )

        content = "\n\n".join(self._sections)
        template = _DOC_CN if self.language == "cn" else _DOC_EN
        doc = self._render(
            template,
            TITLE=title,
            AUTHOR=author,
            DATE=date,
            ABSTRACT=abstract,
            CONTENT=content,
        )
        logger.info(
            "LaTeX 文档组装完成 [sections=%d, tables=%d, figures=%d]",
            len(self._sections), self._table_counter, self._fig_counter,
        )
        return doc

    def save(
        self,
        filename: str = "report.tex",
        title: str = "综合评价模型分析报告",
        author: str = "AutoEval-Modeling",
        abstract: str = "",
        date: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> Path:
        """
        生成完整文档并写入 .tex 文件。

        Parameters
        ----------
        filename : str
            输出文件名（含 .tex 后缀）。
        encoding : str
            文件编码（默认 utf-8）。

        Returns
        -------
        Path
            输出文件路径。
        """
        doc = self.build_document(title=title, author=author,
                                  abstract=abstract, date=date)
        out_path = self.output_dir / filename
        out_path.write_text(doc, encoding=encoding)
        logger.info("LaTeX 文件已保存: %s", out_path)
        return out_path

    def save_sections_only(
        self,
        filename: str = "sections.tex",
        encoding: str = "utf-8",
    ) -> Path:
        """
        仅保存各段落内容（不含文档骨架），方便嵌入已有 LaTeX 项目。

        Returns
        -------
        Path
            输出文件路径。
        """
        content = "\n\n".join(self._sections)
        out_path = self.output_dir / filename
        out_path.write_text(content, encoding=encoding)
        logger.info("LaTeX 段落文件已保存: %s", out_path)
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # 辅助属性
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def section_count(self) -> int:
        """已追加段落数。"""
        return len(self._sections)

    @property
    def used_algorithms(self) -> List[str]:
        """已使用的算法列表（按添加顺序）。"""
        return list(self._used_algorithms)

    def clear(self) -> None:
        """清空所有已追加段落（保留配置）。"""
        self._sections.clear()
        self._used_algorithms.clear()
        self._table_counter = 0
        self._fig_counter   = 0
        logger.debug("LatexBuilder 内容已清空")

    def __repr__(self) -> str:
        return (
            f"LatexBuilder(language={self.language!r}, "
            f"sections={self.section_count}, "
            f"tables={self._table_counter}, "
            f"figures={self._fig_counter})"
        )