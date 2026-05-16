#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AutoEval-Modeling CLI 主入口
============================

命令行接口:
    python -m src.main init [--template basic|advanced]
    python -m src.main run  --config <yaml_path>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import pandas as pd
import yaml

__version__ = "1.0.0"

logger = logging.getLogger("auto_eval")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. 日志初始化                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def setup_logging(level: str = "INFO") -> None:
    """配置 Rich 日志（可优雅降级）。"""
    lvl = getattr(logging, level.upper(), logging.INFO)
    try:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=lvl,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=False)],
            force=True,
        )
    except ImportError:
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s [%(levelname)s] %(message)s",
            force=True,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. 扁平 YAML 配置解析器（兼容 init 生成的格式）                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class FlatConfig:
    """解析 init 命令生成的扁平 YAML 配置。"""

    _DIR_MAP = {
        "pos": 1, "positive": 1, "1": 1,
        "neg": -1, "negative": -1, "-1": -1,
        "mod": 0, "moderate": 0, "0": 0,
    }

    def __init__(self, raw: dict) -> None:
        self.project_name: str = raw.get("project_name", "evaluation_project")
        self.problem_description: str = raw.get("problem_description", "")

        # 数据
        self.data_path: str = raw.get("data_path", "")
        self.object_col: str = raw.get("object_col", "")
        self.indicator_cols: List[str] = raw.get("indicator_cols", [])
        self.indicator_types: List[str] = raw.get("indicator_types", [])

        # 算法
        self.weight_method: str = raw.get("weight_method", "auto")
        self.eval_method: str = raw.get("eval_method", "auto")
        self.enable_combination: bool = raw.get("enable_combination", False)
        self.enable_sensitivity: bool = raw.get("enable_sensitivity", True)
        self.enable_dynamic: bool = raw.get("enable_dynamic", False)

        # 输出
        self.output_dir: str = raw.get("output_dir", "output/")
        self.report_lang: str = raw.get("report_lang", "zh")
        self.generate_latex: bool = raw.get("generate_latex", True)
        self.generate_code: bool = raw.get("generate_code", True)
        self.generate_plots: bool = raw.get("generate_plots", True)
        self.fig_format: str = raw.get("fig_format", "png")
        self.fig_dpi: int = int(raw.get("fig_dpi", 300))

        # 运行
        self.log_level: str = raw.get("log_level", "INFO")
        self.random_seed: int = int(raw.get("random_seed", 42))
        self.max_retries: int = int(raw.get("max_retries", 2))

    @property
    def indicator_directions(self) -> List[int]:
        """将 pos/neg/mod 转换为 1/-1/0。"""
        return [self._DIR_MAP.get(str(t).lower().strip(), 1)
                for t in self.indicator_types]

    @property
    def language(self) -> str:
        return "cn" if self.report_lang == "zh" else "en"


def load_flat_config(yaml_path: str) -> FlatConfig:
    path = Path(yaml_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    logger.info("正在加载配置文件: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return FlatConfig(raw)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. 数据加载                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_data(file_path: str) -> pd.DataFrame:
    """加载 CSV 或 Excel 数据文件。"""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在: {p}")

    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)

    # CSV: 自动检测分隔符
    for sep in [",", "\t", ";", None]:
        try:
            df = pd.read_csv(p, sep=sep, engine="python",
                             encoding="utf-8-sig")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    # 最后兜底：用空白分隔符
    return pd.read_csv(p, sep=r"\s+", engine="python", encoding="utf-8-sig")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. 内置简化算法（不依赖外部 AlgoLib 也能运行）                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def forward_transform(matrix: np.ndarray, directions: List[int]) -> np.ndarray:
    mat = matrix.copy().astype(float)
    for j, d in enumerate(directions):
        if d == -1:
            mat[:, j] = mat[:, j].max() - mat[:, j]
        elif d == 0:
            opt = mat[:, j].mean()
            mat[:, j] = 1.0 / (1.0 + np.abs(mat[:, j] - opt))
    return mat


def minmax_normalize(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    lo, hi = matrix.min(axis=0), matrix.max(axis=0)
    denom = np.where((hi - lo) < eps, eps, hi - lo)
    return (matrix - lo) / denom


def run_entropy_weight(matrix: np.ndarray, eps: float = 1e-10) -> dict:
    m, n = matrix.shape
    cs = np.where(matrix.sum(0) < eps, eps, matrix.sum(0))
    P = matrix / cs
    with np.errstate(divide="ignore", invalid="ignore"):
        lnP = np.where(P > eps, np.log(P), 0.0)
    e = np.clip(-1.0 / np.log(m + eps) * (P * lnP).sum(0), 0.0, 1.0)
    d = 1.0 - e
    w = d / (d.sum() + eps)
    return {"weights": w, "entropy": e, "diversity": d}


def run_critic_weight(matrix: np.ndarray, eps: float = 1e-10) -> dict:
    sigma = matrix.std(0, ddof=1)
    corr = np.nan_to_num(np.corrcoef(matrix.T), nan=0.0)
    n = matrix.shape[1]
    conflict = np.array([(1 - corr[j]).sum() for j in range(n)])
    C = sigma * conflict
    w = C / (C.sum() + eps)
    return {"weights": w, "std_devs": sigma, "conflicts": conflict,
            "info_amounts": C, "corr_matrix": corr}


def _make_ranking(scores: np.ndarray) -> np.ndarray:
    """分数越高排名越靠前（1-based）。"""
    rank = np.empty_like(scores, dtype=int)
    rank[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
    return rank


def run_topsis(matrix: np.ndarray, weights: np.ndarray,
               eps: float = 1e-10) -> dict:
    V = matrix * weights
    D_pos = np.sqrt(((V - V.max(0)) ** 2).sum(1))
    D_neg = np.sqrt(((V - V.min(0)) ** 2).sum(1))
    scores = D_neg / (D_pos + D_neg + eps)
    return {"scores": scores, "ranking": _make_ranking(scores),
            "d_positive": D_pos, "d_negative": D_neg}


def run_gra(matrix: np.ndarray, weights: np.ndarray,
            rho: float = 0.5, eps: float = 1e-10) -> dict:
    ref = matrix.max(0)
    delta = np.abs(matrix - ref)
    coeff = (delta.min() + rho * delta.max()) / (delta + rho * delta.max() + eps)
    scores = (coeff * weights).sum(1)
    return {"scores": scores, "ranking": _make_ranking(scores),
            "coeff_matrix": coeff}


def run_sensitivity(matrix: np.ndarray, base_w: np.ndarray,
                    eval_fn: Callable, names: List[str],
                    delta: float = 0.2, steps: int = 21) -> dict:
    ratios = np.linspace(-delta, delta, steps)
    base_rank = eval_fn(matrix, base_w)["ranking"]
    records: Dict[str, np.ndarray] = {}
    robust = True
    for j, name in enumerate(names):
        rm = []
        for r in ratios:
            w = base_w.copy()
            w[j] = max(0.0, w[j] * (1 + r))
            s = w.sum()
            if s < 1e-10:
                rm.append(base_rank.copy())
            else:
                rm.append(eval_fn(matrix, w / s)["ranking"].copy())
        rm = np.vstack(rm)
        records[name] = rm
        if not np.all(rm == base_rank):
            robust = False
    return {"perturb_ratios": ratios, "rank_records": records,
            "is_robust": robust}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. JSON 序列化（处理 numpy 类型）                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  6. LaTeX 简易生成（不依赖 Jinja2 模板也可工作）                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def generate_simple_latex(
    cfg: FlatConfig,
    objects: List[str],
    indicator_cols: List[str],
    weights: Optional[np.ndarray],
    weight_method: str,
    eval_result: Optional[dict],
    eval_method: str,
    sens_result: Optional[dict],
    weight_result: Optional[dict],
) -> str:
    wm = {"entropy": "熵权法", "critic": "CRITIC法",
           "ahp": "AHP法"}.get(weight_method, weight_method)
    em = {"topsis": "TOPSIS", "gra": "灰色关联分析",
           "vikor": "VIKOR"}.get(eval_method, eval_method)

    L = [  # noqa: E741
        r"\documentclass[12pt]{ctexart}",
        r"\usepackage{booktabs,amsmath,graphicx,geometry}",
        r"\geometry{a4paper,margin=2.5cm}",
        r"\begin{document}",
        rf"\title{{{cfg.project_name} 综合评价报告}}",
        r"\author{AutoEval-Modeling 自动生成}",
        rf"\date{{{datetime.now().strftime('%Y年%m月%d日')}}}",
        r"\maketitle", "",
        r"\section{模型建立}", "",
        f"本文采用{wm}确定指标权重，并运用{em}法进行综合评价。", "",
    ]

    # 权重表
    if weights is not None:
        n = len(indicator_cols)
        cfmt = "l" + "c" * n
        L += [
            r"\subsection{指标权重}",
            r"\begin{table}[htbp]\centering",
            rf"\caption{{{wm}计算结果}}",
            rf"\begin{{tabular}}{{{cfmt}}}",
            r"\toprule",
            "指标 & " + " & ".join(indicator_cols) + r" \\",
            r"\midrule",
            "权重 & " + " & ".join(f"{w:.4f}" for w in weights) + r" \\",
            r"\bottomrule",
            r"\end{tabular}\end{table}", "",
        ]

    # 评价结果表
    if eval_result is not None:
        L += [
            r"\subsection{综合评价结果}",
            r"\begin{table}[htbp]\centering",
            rf"\caption{{{em}综合评价结果}}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"评价对象 & 综合得分 & 排名 \\",
            r"\midrule",
        ]
        order = np.argsort(eval_result["ranking"])
        for idx in order:
            L.append(
                f"{objects[idx]} & {eval_result['scores'][idx]:.4f} & "
                f"{eval_result['ranking'][idx]}" + r" \\"
            )
        L += [r"\bottomrule", r"\end{tabular}\end{table}", ""]

    # 灵敏度
    if sens_result is not None:
        robust_txt = "稳健" if sens_result["is_robust"] else "部分不稳健"
        L += [
            r"\subsection{灵敏度分析}",
            f"对各指标权重在 $\\pm 20\\%$ 范围内进行 OAT 扰动分析，"
            f"结果表明排名整体{robust_txt}。", "",
        ]

    L.append(r"\end{document}")
    return "\n".join(L)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  7. 验证列名并自动修正                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def resolve_columns(
    df: pd.DataFrame,
    cfg: FlatConfig,
) -> tuple[str, List[str], List[int]]:
    """
    校验配置中的列名是否存在于 DataFrame 中，
    若不匹配则尝试自动推断并打印警告。
    自动推断时只选取可转为数值的列。

    Returns
    -------
    (object_col, indicator_cols, directions)
    """
    object_col = cfg.object_col
    indicator_cols = list(cfg.indicator_cols)
    directions = cfg.indicator_directions

    actual_cols = df.columns.tolist()

    # ---------- 检查 object_col ----------
    if object_col not in actual_cols:
        logger.warning("对象列 '%s' 在数据中不存在，自动使用第一列 '%s'",
                       object_col, actual_cols[0])
        object_col = actual_cols[0]

    # ---------- 检查 indicator_cols ----------
    missing = [c for c in indicator_cols if c not in actual_cols]
    if missing:
        logger.warning("以下指标列在数据中不存在: %s", missing)
        logger.info("数据实际列名: %s", actual_cols)

        # ★ 自动推断：除对象列外，只保留"至少有一个值可转为数字"的列
        candidate_cols = [c for c in actual_cols if c != object_col]
        indicator_cols = []
        for c in candidate_cols:
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().any():          # 至少有一个数值
                indicator_cols.append(c)
            else:
                logger.debug("  列 '%s' 全为非数值，跳过", c)

        directions = [1] * len(indicator_cols)
        logger.warning("已自动切换为: 对象列='%s', 指标列=%s (全部正向)",
                       object_col, indicator_cols)
    elif len(directions) != len(indicator_cols):
        directions = [1] * len(indicator_cols)
        logger.warning("指标方向数量不匹配，已重置为全部正向")

    return object_col, indicator_cols, directions


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  8. init 命令                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def cmd_init(args: argparse.Namespace) -> None:
    project_name = (input("请输入项目名称 [evaluation_project]: ").strip()
                    or "evaluation_project")
    data_path = (input("请输入数据文件路径 [data/city_innovation.csv]: ").strip()
                 or "data/city_innovation.csv")

    # 自动检测列名
    auto_object = "对象列名"
    auto_ind_section = "  - 指标1\n  - 指标2\n  - 指标3"
    auto_type_section = "  - pos\n  - pos\n  - pos"

    try:
        df = load_data(data_path)
        cols = df.columns.tolist()
        if len(cols) >= 2:
            auto_object = cols[0]
            ind_cols = cols[1:]
            auto_ind_section = "\n".join(f"  - {c}" for c in ind_cols)
            auto_type_section = "\n".join("  - pos" for _ in ind_cols)
            print(f"\n  ✓ 自动检测到列名: {cols}")
            print(f"    对象列: {auto_object}")
            print(f"    指标列: {ind_cols}")
    except Exception as exc:
        print(f"  ⚠ 无法自动检测列名: {exc}")

    template = f"""\
# AutoEval-Modeling 基础配置模板
# 生成时间: {datetime.now().strftime('%Y-%m-%d')}

project_name: {project_name}
problem_description: "请在此填写评价问题描述"

data_path: {data_path}
object_col: {auto_object}
indicator_cols:
{auto_ind_section}
indicator_types:
{auto_type_section}

weight_method: auto
eval_method: auto
enable_combination: false
enable_sensitivity: true
enable_dynamic: false

output_dir: output/
report_lang: zh
generate_latex: true
generate_code: true
generate_plots: true
fig_format: png
fig_dpi: 300

log_level: INFO
random_seed: 42
max_retries: 2
"""

    out = Path("my_project.yaml")
    out.write_text(template, encoding="utf-8")
    print(f"\n✅ 配置模板已生成: {out}")
    print("   请编辑文件中的数据路径、指标列名等参数后运行：")
    print(f"   python -m src.main run --config {out}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  9. run 命令                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def cmd_run(args: argparse.Namespace) -> None:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    t_total = time.time()

    # ── 追踪器 ────────────────────────────────────────────────────────────
    steps: Dict[str, Dict[str, Any]] = {}
    warnings_list: List[str] = []
    output_files: List[str] = []
    data_store: Dict[str, Any] = {}

    def step(name: str, fn: Callable, *a: Any, **kw: Any) -> Any:
        """执行一步并自动记录耗时/状态。"""
        t0 = time.time()
        try:
            result = fn(*a, **kw)
            steps[name] = {"status": "ok", "time": time.time() - t0}
            return result
        except Exception as exc:
            steps[name] = {"status": "fail", "time": time.time() - t0,
                           "error": str(exc)}
            logger.error("  %s 失败: %s", name, exc)
            return None

    # ── 阶段 0: 加载配置 ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("AutoEval-Modeling v%s  会话 %s", __version__, session_id)
    logger.info("=" * 60)
    logger.info("[阶段0] 配置校验...")

    try:
        cfg = load_flat_config(args.config)
        logger.info("  配置校验通过 ✓")
    except Exception as exc:
        logger.error("  配置加载失败: %s", exc)
        return

    setup_logging(cfg.log_level)
    np.random.seed(cfg.random_seed)

    # ── 阶段 1: 数据加载 ─────────────────────────────────────────────────
    logger.info("[阶段1] 数据加载: %s", cfg.data_path)

    df: Optional[pd.DataFrame] = step("data_loading", load_data, cfg.data_path)
    if df is None:
        logger.error("  数据加载失败，流程终止")
        return
    logger.info("  数据加载完成: %d 行 × %d 列", *df.shape)
    logger.info("  列名: %s", df.columns.tolist())

    # 列名校验与自动修正
    object_col, indicator_cols, directions = resolve_columns(df, cfg)
    objects: List[str] = df[object_col].astype(str).tolist()

    # ── 数值安全转换 ──────────────────────────────────────────────────
    # 将指标列强制转为数值型，无法转换的单元格变为 NaN
    numeric_df = df[indicator_cols].apply(pd.to_numeric, errors="coerce")

    # 移除完全非数值的列（文本/分类列被误选为指标的情况）
    all_nan_mask = numeric_df.isnull().all()
    if all_nan_mask.any():
        bad_cols = numeric_df.columns[all_nan_mask].tolist()
        logger.warning("以下列全为非数值数据，已自动排除: %s", bad_cols)
        keep_flags = (~all_nan_mask).tolist()
        indicator_cols = [c for c, k in zip(indicator_cols, keep_flags) if k]
        directions    = [d for d, k in zip(directions, keep_flags) if k]
        numeric_df    = numeric_df.loc[:, ~all_nan_mask]

    if len(indicator_cols) == 0:
        logger.error("没有可用的数值型指标列，流程终止")
        return

    # 统计因类型转换而新增的 NaN（将在预处理阶段用均值填充）
    coerced_nan_count = int(numeric_df.isnull().sum().sum())
    if coerced_nan_count > 0:
        logger.warning(
            "指标数据中有 %d 个单元格无法转为数值，已置为 NaN（后续自动填充）",
            coerced_nan_count,
        )

    X_raw: np.ndarray = numeric_df.values.astype(float)

    # ── 阶段 2: 预处理 ───────────────────────────────────────────────────
    logger.info("[阶段2] 数据预处理...")
    t0 = time.time()
    try:
        # 缺失值
        if np.isnan(X_raw).any():
            means = np.nanmean(X_raw, axis=0)
            inds = np.where(np.isnan(X_raw))
            for r, c in zip(*inds):
                X_raw[r, c] = means[c]
            logger.info("  缺失值已用列均值填充")

        X_pos = forward_transform(X_raw, directions)
        X_norm = minmax_normalize(X_pos)

        # 数据画像（可选）
        profile_info: Dict[str, Any] = {}
        try:
            from src.parser.data_profiler import DataProfiler
            profiler = DataProfiler(X_norm, indicator_cols)
            profile_info = profiler.analyze()
            data_store["profile"] = {
                k: v for k, v in profile_info.items()
                if not isinstance(v, np.ndarray)
            }
        except Exception as exc:
            logger.debug("  数据画像跳过: %s", exc)

        steps["preprocessing"] = {"status": "ok", "time": time.time() - t0}
        logger.info("  预处理完成 ✓  shape=%s", X_norm.shape)
    except Exception as exc:
        steps["preprocessing"] = {"status": "fail", "time": time.time() - t0,
                                  "error": str(exc)}
        logger.error("  预处理失败: %s", exc)
        X_norm = X_raw.copy()
        warnings_list.append(f"预处理阶段异常，使用原始数据继续: {exc}")

    # ── 阶段 3: 算法推荐 ─────────────────────────────────────────────────
    logger.info("[阶段3] 算法推荐...")
    weight_method = cfg.weight_method
    eval_method = cfg.eval_method
    t0 = time.time()

    if "auto" in (weight_method, eval_method):
        try:
            from src.core.recommendation import (
                AlgorithmRecommender, DataProfile, UserPreferences,
            )
            profile = DataProfile.from_dataframe(
                pd.DataFrame(X_norm, columns=indicator_cols),
                direction=directions,
            )
            recommender = AlgorithmRecommender(
                preferences=UserPreferences(
                    prefer_objective=True,
                    allow_subjective=False,
                )
            )
            plans = recommender.recommend(profile)
            if plans:
                best = plans[0]
                if weight_method == "auto":
                    weight_method = best.weight_method
                if eval_method == "auto":
                    eval_method = best.eval_method
                logger.info("  推荐方案: %s (score=%.3f)", best.name, best.score)
                for r in best.reasons:
                    logger.info("    ✓ %s", r)
            steps["algorithm_recommendation"] = {
                "status": "ok", "time": time.time() - t0}
        except Exception as exc:
            steps["algorithm_recommendation"] = {
                "status": "fail", "time": time.time() - t0, "error": str(exc)}
            logger.warning("  算法推荐失败，使用默认值 entropy-topsis: %s", exc)
            warnings_list.append(f"算法推荐异常，使用默认配置: {exc}")
            weight_method = weight_method if weight_method != "auto" else "entropy"
            eval_method = eval_method if eval_method != "auto" else "topsis"
    else:
        steps["algorithm_recommendation"] = {
            "status": "ok", "time": time.time() - t0}

    logger.info("  最终选定: 赋权=%s  评价=%s", weight_method, eval_method)
    data_store["weight_method"] = weight_method
    data_store["eval_method"] = eval_method

    # ── 阶段 4: 核心计算 ─────────────────────────────────────────────────
    logger.info("[阶段4] 核心工作流执行（%s → %s）...", weight_method, eval_method)

    # 4a. 权重
    weights: Optional[np.ndarray] = None
    weight_result: Optional[dict] = None
    t0 = time.time()
    try:
        if weight_method == "entropy":
            weight_result = run_entropy_weight(X_norm)
        elif weight_method == "critic":
            weight_result = run_critic_weight(X_norm)
        else:
            weight_result = {"weights": np.ones(X_norm.shape[1]) / X_norm.shape[1]}

        weights = weight_result["weights"]
        data_store["weights"] = weights.tolist()
        steps["weight_calculation"] = {"status": "ok", "time": time.time() - t0}

        logger.info("  权重计算完成:")
        for name, w in zip(indicator_cols, weights):
            logger.info("    %s: %.4f", name, w)
    except Exception as exc:
        steps["weight_calculation"] = {"status": "fail", "time": time.time() - t0,
                                       "error": str(exc)}
        logger.error("  权重计算失败: %s", exc)
        weights = np.ones(X_norm.shape[1]) / X_norm.shape[1]
        weight_result = {"weights": weights}
        warnings_list.append(f"权重计算异常，使用等权重: {exc}")

    # 4b. 综合评价
    eval_result: Optional[dict] = None
    t0 = time.time()
    try:
        if eval_method == "topsis":
            eval_result = run_topsis(X_norm, weights)
        elif eval_method == "gra":
            eval_result = run_gra(X_norm, weights)
        else:
            eval_result = run_topsis(X_norm, weights)

        data_store["scores"] = eval_result["scores"].tolist()
        data_store["ranking"] = eval_result["ranking"].tolist()
        steps["evaluation_model"] = {"status": "ok", "time": time.time() - t0}

        rdf = pd.DataFrame({
            "评价对象": objects,
            "综合得分": np.round(eval_result["scores"], 4),
            "排名": eval_result["ranking"],
        }).sort_values("排名")
        logger.info("  综合评价完成:\n%s", rdf.to_string(index=False))
    except Exception as exc:
        steps["evaluation_model"] = {"status": "fail", "time": time.time() - t0,
                                     "error": str(exc)}
        logger.error("  综合评价失败: %s", exc)

    # 4c. 灵敏度分析
    sens_result: Optional[dict] = None
    if cfg.enable_sensitivity and eval_result is not None:
        t0 = time.time()
        try:
            fn = run_topsis if eval_method == "topsis" else run_gra
            sens_result = run_sensitivity(X_norm, weights, fn, indicator_cols)
            steps["sensitivity_analysis"] = {
                "status": "ok", "time": time.time() - t0}
            tag = "稳健 ✓" if sens_result["is_robust"] else "部分不稳健 ⚠"
            logger.info("  灵敏度分析完成: %s", tag)
        except Exception as exc:
            steps["sensitivity_analysis"] = {
                "status": "fail", "time": time.time() - t0, "error": str(exc)}
            logger.warning("  灵敏度分析失败: %s", exc)
    else:
        steps["sensitivity_analysis"] = {"status": "skip", "time": 0}

    # 结果整合
    steps["result_consolidation"] = {"status": "ok", "time": 0}

    # ── 阶段 5: 生成输出 ─────────────────────────────────────────────────
    logger.info("[阶段5] 生成输出文件...")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5a. LaTeX 报告
    if cfg.generate_latex:
        t0 = time.time()
        try:
            tex_dir = out_dir / "reports"
            tex_dir.mkdir(parents=True, exist_ok=True)
            tex_path = tex_dir / f"{cfg.project_name}.tex"

            # 优先尝试 LatexBuilder 模板引擎
            try:
                from src.generators.latex_builder import LatexBuilder
                lb = LatexBuilder(
                    language=cfg.language,
                    output_dir=str(tex_dir),
                )
                # 如果 LatexBuilder 支持完整接口就调用
                content = lb.build(
                    cfg=cfg, objects=objects, indicator_cols=indicator_cols,
                    weights=weights, weight_method=weight_method,
                    eval_result=eval_result, eval_method=eval_method,
                    sens_result=sens_result, weight_result=weight_result,
                )
                tex_path.write_text(content, encoding="utf-8")
            except Exception:
                # 降级：使用内置简化生成器
                content = generate_simple_latex(
                    cfg, objects, indicator_cols, weights, weight_method,
                    eval_result, eval_method, sens_result, weight_result,
                )
                tex_path.write_text(content, encoding="utf-8")

            steps["latex_generation"] = {"status": "ok",
                                         "time": time.time() - t0}
            output_files.append(str(tex_path))
            logger.info("  LaTeX 报告: %s", tex_path)
        except Exception as exc:
            steps["latex_generation"] = {"status": "fail",
                                         "time": time.time() - t0,
                                         "error": str(exc)}
            logger.warning("  LaTeX 生成失败: %s", exc)

    # 5b. Python 脚本
    if cfg.generate_code:
        t0 = time.time()
        try:
            from src.generators.code_builder import CodeBuilder

            cb = CodeBuilder(
                language=cfg.language,
                output_dir=str(out_dir / "scripts"),
                title=f"{cfg.project_name} 综合评价模型求解脚本",
            )
            cb.set_data_config(
                data_file=cfg.data_path,
                object_col=object_col,
                indicator_cols=indicator_cols,
                indicator_types=directions,
            )
            cb.add_weight_method(weight_method)
            cb.add_evaluation_method(eval_method)
            if cfg.enable_sensitivity:
                cb.add_sensitivity_analysis()
            cb.add_save_results()
            script_path = cb.save(f"{cfg.project_name}_script.py")

            steps["code_generation"] = {"status": "ok",
                                        "time": time.time() - t0}
            output_files.append(str(script_path))
            logger.info("  Python 脚本: %s", script_path)
        except Exception as exc:
            steps["code_generation"] = {"status": "fail",
                                        "time": time.time() - t0,
                                        "error": str(exc)}
            logger.warning("  代码生成失败: %s", exc)

    # 5c. 图表
    if cfg.generate_plots and eval_result is not None:
        t0 = time.time()
        try:
            from src.generators.plot_builder import PlotBuilder

            pb = PlotBuilder(
                language=cfg.language,
                output_dir=str(out_dir / "figures"),
                fig_format=cfg.fig_format,
                dpi=cfg.fig_dpi,
            )
            pb.plot_weights(weights, indicator_cols,
                            method_name=weight_method)
            pb.plot_ranking(eval_result["scores"], objects,
                            method_name=eval_method.upper())
            pb.plot_radar(X_norm, indicator_cols, objects)
            pb.plot_correlation_heatmap(
                pd.DataFrame(X_raw, columns=indicator_cols),
                indicator_cols,
            )
            if sens_result is not None:
                pb.plot_sensitivity(sens_result, objects)
            if weight_method == "entropy" and weight_result:
                pb.plot_entropy_analysis(weight_result, indicator_cols)

            saved = pb.save_all()
            steps["plot_generation"] = {"status": "ok",
                                        "time": time.time() - t0}
            output_files.extend(str(p) for p in saved)
            logger.info("  图表生成: %d 张", len(saved))
        except Exception as exc:
            steps["plot_generation"] = {"status": "fail",
                                        "time": time.time() - t0,
                                        "error": str(exc)}
            logger.warning("  图表生成失败: %s", exc)

    # ── 执行摘要 ──────────────────────────────────────────────────────────
    total_time = time.time() - t_total
    n_ok = sum(1 for s in steps.values() if s["status"] == "ok")
    n_fail = sum(1 for s in steps.values() if s["status"] == "fail")
    status = "SUCCESS" if n_fail == 0 else ("PARTIAL" if n_ok > 0 else "FAILED")

    summary = {
        "session_id": session_id,
        "status": status,
        "total_time": total_time,
        "weight_method": weight_method,
        "eval_method": eval_method,
        "steps": steps,
        "warnings": warnings_list,
        "output_files": output_files,
        "data": data_store,
    }

    # 保存 JSON
    try:
        log_dir = out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        summary_path = log_dir / f"summary_{session_id}.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, cls=NumpyEncoder),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("执行摘要保存失败: %s", exc)

    # 终端输出
    if n_fail == 0:
        logger.info("流程全部完成 ✓")
    else:
        failed_names = [k for k, v in steps.items() if v["status"] == "fail"]
        logger.warning("流程部分完成，%d 个步骤失败: %s", n_fail, failed_names)

    print(f"\n{'='*70}")
    print(f"  AutoEval-Modeling — 执行摘要  (会话 ID: {session_id})")
    print(f"{'='*70}")
    print(f"  状态        : {status}")
    print(f"  耗时        : {total_time:.2f} 秒")
    print(f"  推荐算法    : {weight_method}-{eval_method}")
    print(f"  完成步骤    : {n_ok} / {len(steps)}")

    if warnings_list:
        print(f"\n  ⚠ 警告（共 {len(warnings_list)} 条）:")
        for w in warnings_list:
            print(f"    - {w}")

    if output_files:
        print("\n  📁 输出文件:")
        for f in output_files:
            tag = ("latex" if f.endswith(".tex") else
                   "code" if f.endswith(".py") else "figure")
            print(f"    [{tag}] {f}")

    print("\n  步骤明细:")
    for name, info in steps.items():
        icon = {"ok": "✓", "skip": "⊘", "fail": "✗"}.get(info["status"], "?")
        print(f"    {icon} {name:30s} {info.get('time', 0):.2f}s")

    failed = {k: v for k, v in steps.items() if v["status"] == "fail"}
    if failed:
        print("\n  ✗ 失败步骤详情:")
        for name, info in failed.items():
            print(f"    [{name}] {info.get('error', 'unknown')}")

    print(f"{'='*70}\n")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  10. CLI 入口                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="auto-eval",
        description="AutoEval-Modeling: 评价类数学建模自动化工具",
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    p_init = sub.add_parser("init", help="生成配置模板")
    p_init.add_argument("--template", default="basic",
                        choices=["basic", "advanced"])

    p_run = sub.add_parser("run", help="执行评价工作流")
    p_run.add_argument("--config", required=True, help="配置文件路径")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    logger.info("执行命令: %s", " ".join(sys.argv[1:]))

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    setup_logging("INFO")
    main()
