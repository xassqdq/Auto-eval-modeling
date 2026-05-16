# src/templates/python/CONTEXT_SCHEMA.py
"""
script_template.py.j2 和 function_blocks.py.j2 的
完整渲染上下文 (Render Context) 结构规范。

由 src/generators/code_builder.py 负责构建并传入 Jinja2 Environment。
"""

CONTEXT_SCHEMA = {
    # ── 元信息 ────────────────────────────────────────────────
    "meta": {
        "project_name":  str,      # 项目名称，用于文件标题
        "generated_at":  str,      # ISO 格式时间字符串
        "author":        str,      # 生成工具名称
        "version":       str,      # AutoEval-Modeling 版本号
        "description":   str,      # 项目简要描述
        "config_source": str,      # 配置文件路径
    },

    # ── 数据与输出配置 ─────────────────────────────────────────
    "config": {
        "data_file":            str,              # 数据文件路径
        "file_type":            str,              # "csv" | "excel"
        "encoding":             str,              # 默认 "utf-8"
        "sheet_name":           "str | int | None",
        "eval_objects_col":     str,              # 评价对象列名
        "indicator_cols":       "List[str]",      # 原始列名（与 CSV 一致）
        "indicator_names":      "List[str]",      # 展示名（可含中文）
        "indicator_directions": "List[int]",      # +1/-1/0
        "moderate_opts":        "List[float|None]",
        "output_dir":           str,
        "lang":                 str,              # "zh" | "en"
        "fig_format":           str,              # "png"/"pdf"/"svg"
        "fig_dpi":              int,
    },

    # ── 预处理配置 ─────────────────────────────────────────────
    "preprocessing": {
        "handle_missing":        bool,
        "missing_strategy":      str,   # mean/median/drop/interpolate
        "handle_outliers":       bool,
        "outlier_method":        str,   # iqr/zscore
        "normalize_method":      str,   # minmax/zscore/vector/sum
        "check_correlation":     bool,
        "correlation_threshold": float,
        "use_pca":               bool,
        "pca_variance":          float,
    },

    # ── 赋权配置 ───────────────────────────────────────────────
    "weights": {
        "method":             str,               # 主赋权方法名称（用于标签）
        "objective_methods":  "List[str]",       # 客观方法列表
        "ahp_matrix":         "List[List[float]] | None",
        "expert_weights":     "List[float] | None",
        "combination_method": str,               # linear/multiplicative/min_deviation
        "combination_coef":   "List[float] | None",
    },

    # ── 评价模型配置 ───────────────────────────────────────────
    "evaluation": {
        "model":    str,            # 主模型名称
        "models":   "List[str]",    # 所有运行的模型（多模型对比时）
        "vikor_v":  float,          # VIKOR v 参数
        "gra_rho":  float,          # GRA ρ 参数
    },

    # ── 灵敏度分析配置 ─────────────────────────────────────────
    "sensitivity": {
        "enabled":           bool,
        "perturbation_range": float,
        "steps":              int,
    },

    # ── 输出控制 ───────────────────────────────────────────────
    "output": {
        "save_csv":     bool,
        "save_figures": bool,
        "show_plots":   bool,
        "verbose":      bool,
    },

    # ── 工作流步骤列表（有序）────────────────────────────────────
    "workflow_steps": "List[str]",   # 如 ["数据加载", "预处理", ...]

    # ── 算法使能标志（由 code_builder 预处理，避免模板中复杂判断）──
    "use_ahp":               bool,
    "use_entropy":           bool,
    "use_critic":            bool,
    "use_stddev":            bool,
    "use_combination_weights": bool,
    "use_topsis":            bool,
    "use_vikor":             bool,
    "use_gra":               bool,
    "use_fuzzy":             bool,
    "use_rsr":               bool,
    "use_pca":               bool,
    "use_sensitivity":       bool,
    "use_multiple_models":   bool,
}