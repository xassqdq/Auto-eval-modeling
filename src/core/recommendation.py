"""
算法知识图谱推荐器 (AlgorithmRecommender)

基于数据特征、用户偏好与知识图谱规则，
自动推荐最适合的评价算法组合。

推荐策略：
    1. 数据特征分析（缺失率、相关性、样本数/指标数比）
    2. 用户偏好约束（主观/客观、是否需要动态评价）
    3. 知识图谱匹配（情境标签 → 算法路径）
    4. 多套方案评分排名

Author: AutoEval-Modeling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# 数据结构
# ======================================================================

@dataclass
class DataProfile:
    """数据集特征画像"""
    n_objects:       int   = 0
    n_indicators:    int   = 0
    missing_rate:    float = 0.0     # 总体缺失率
    max_corr:        float = 0.0     # 最大两两相关系数（绝对值）
    mean_corr:       float = 0.0     # 平均相关系数
    has_negative:    bool  = False   # 是否存在负向指标
    ratio_n_p:       float = 0.0     # n_objects / n_indicators
    data_variance:   float = 0.0     # 所有指标变异系数均值
    has_time_series: bool  = False   # 是否为多时段数据

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        direction: Optional[List[int]] = None,
        time_col: Optional[str] = None,
    ) -> "DataProfile":
        """从 DataFrame 自动生成特征画像"""
        n_objects, n_indicators = df.shape
        missing_rate = df.isnull().mean().mean()

        # 相关性分析
        try:
            corr_mat = df.corr().abs()
            np.fill_diagonal(corr_mat.values, 0)
            max_corr  = float(corr_mat.max().max())
            mean_corr = float(corr_mat.mean().mean())
        except Exception:
            max_corr = mean_corr = 0.0

        # 变异系数
        try:
            cv = (df.std() / (df.mean().abs() + 1e-10)).mean()
            data_variance = float(cv)
        except Exception:
            data_variance = 0.0

        has_negative    = bool(direction and any(d == -1 for d in direction))
        ratio_n_p       = n_objects / max(n_indicators, 1)
        has_time_series = time_col is not None

        return cls(
            n_objects       = n_objects,
            n_indicators    = n_indicators,
            missing_rate    = float(missing_rate),
            max_corr        = max_corr,
            mean_corr       = mean_corr,
            has_negative    = has_negative,
            ratio_n_p       = ratio_n_p,
            data_variance   = data_variance,
            has_time_series = has_time_series,
        )


@dataclass
class AlgorithmPlan:
    """算法推荐方案"""
    plan_id:          str
    name:             str
    description:      str
    weight_method:    str
    eval_method:      str
    preprocess_steps: List[str]        = field(default_factory=list)
    sensitivity:      bool             = True
    rank_consistency: bool             = False
    score:            float            = 0.0    # 推荐得分（越高越优先）
    reasons:          List[str]        = field(default_factory=list)
    warnings:         List[str]        = field(default_factory=list)
    workflow_config:  Dict[str, Any]   = field(default_factory=dict)

    def to_display_dict(self) -> Dict:
        """格式化为显示字典"""
        return {
            "方案":     self.name,
            "赋权方法": self.weight_method,
            "评价方法": self.eval_method,
            "推荐得分": round(self.score, 3),
            "理由":     " | ".join(self.reasons),
            "注意事项": " | ".join(self.warnings) if self.warnings else "无",
        }


@dataclass
class UserPreferences:
    """用户偏好设置"""
    prefer_objective:   bool = True    # 偏向客观赋权
    allow_subjective:   bool = False   # 是否允许主观赋权（AHP等）
    need_dynamic:       bool = False   # 是否需要动态评价
    need_classification: bool = False  # 是否需要分级（不仅排序）
    need_fuzzy:         bool = False   # 是否需要模糊处理（定性指标）
    require_dea:        bool = False   # 是否必须包含 DEA
    ahp_matrix:         Optional[List[List[float]]] = None  # 主观判断矩阵
    min_plans:          int = 2        # 至少推荐几套方案
    max_plans:          int = 4        # 最多推荐几套方案


# ======================================================================
# 推荐器主类
# ======================================================================

class AlgorithmRecommender:
    """
    基于知识图谱的算法组合推荐器

    Parameters
    ----------
    preferences : UserPreferences, optional
        用户偏好设置

    Examples
    --------
    >>> recommender = AlgorithmRecommender()
    >>> profile = DataProfile.from_dataframe(df, direction=[1,1,-1,1])
    >>> plans = recommender.recommend(profile)
    >>> for plan in plans:
    ...     print(plan.name, plan.score)
    """

    # ------------------------------------------------------------------
    # 知识图谱规则库
    # ------------------------------------------------------------------

    # 格式: (rule_name, condition_fn, action_fn, priority)
    # condition_fn(profile, prefs) -> bool
    # action_fn(profile, prefs) -> AlgorithmPlan

    _WEIGHT_RULES = [
        # 规则1: 高相关性 → CRITIC 比熵权更合适
        {
            "name": "high_correlation_critic",
            "condition": lambda p, _: p.max_corr > 0.7,
            "weight_method": "critic",
            "reason": "指标间存在强相关性(max_corr>0.7)，CRITIC法能同时衡量对比性与冲突性",
            "score_bonus": 0.15,
        },
        # 规则2: 低变异系数 → 熵权法效果弱，考虑 CRITIC
        {
            "name": "low_variance_critic",
            "condition": lambda p, _: p.data_variance < 0.1,
            "weight_method": "critic",
            "reason": "指标变异系数较小，熵权法区分度不足，建议使用CRITIC法",
            "score_bonus": 0.08,
        },
        # 规则3: 有主观判断矩阵 → AHP
        {
            "name": "has_ahp_matrix",
            "condition": lambda p, prefs: (
                prefs.allow_subjective and prefs.ahp_matrix is not None
            ),
            "weight_method": "ahp",
            "reason": "用户提供了专家判断矩阵，启用层次分析法(AHP)",
            "score_bonus": 0.12,
        },
        # 规则4: 样本少 → 熵权可能不稳定
        {
            "name": "small_sample_warning",
            "condition": lambda p, _: p.n_objects < 6,
            "weight_method": "entropy",
            "reason": "样本量较小，熵权法结果可能不稳定",
            "score_bonus": -0.1,
            "warning": "样本量 < 6，建议增加评价对象数量或改用 AHP 等主观方法",
        },
    ]

    _EVAL_RULES = [
        # 综合评价方法选择规则
        {
            "name": "standard_topsis",
            "condition": lambda p, _: p.ratio_n_p >= 1.0,
            "eval_method": "topsis",
            "reason": "样本数/指标数比≥1，TOPSIS法适用，基于理想解的综合排序",
            "score_bonus": 0.1,
        },
        {
            "name": "small_ratio_gra",
            "condition": lambda p, _: p.ratio_n_p < 1.0,
            "eval_method": "gra",
            "reason": "样本数/指标数比<1，灰色关联分析对数据量要求较低",
            "score_bonus": 0.12,
        },
        {
            "name": "fuzzy_needed",
            "condition": lambda p, prefs: prefs.need_fuzzy,
            "eval_method": "fuzzy",
            "reason": "用户指定需要模糊综合评价，适合处理定性指标",
            "score_bonus": 0.15,
        },
        {
            "name": "classification_rsr",
            "condition": lambda p, prefs: prefs.need_classification,
            "eval_method": "rsr",
            "reason": "需要分级评价，秩和比(RSR)法适合对评价对象进行分级",
            "score_bonus": 0.12,
        },
        {
            "name": "efficiency_dea",
            "condition": lambda p, prefs: prefs.require_dea,
            "eval_method": "dea",
            "reason": "用户指定需要效率评价，数据包络分析(DEA)适合投入产出效率评估",
            "score_bonus": 0.15,
        },
    ]

    def __init__(
        self,
        preferences: Optional[UserPreferences] = None,
    ) -> None:
        self.preferences = preferences or UserPreferences()

    def recommend(
        self,
        profile: DataProfile,
        top_k: Optional[int] = None,
    ) -> List[AlgorithmPlan]:
        """
        基于数据画像生成推荐方案列表

        Parameters
        ----------
        profile : DataProfile
            数据集特征画像
        top_k : int, optional
            返回前 k 个方案（None = 返回所有）

        Returns
        -------
        list of AlgorithmPlan，按推荐得分降序
        """
        prefs = self.preferences
        plans: List[AlgorithmPlan] = []

        # ---- 生成候选方案 ----
        candidate_combinations = self._get_candidate_combinations(profile, prefs)

        for combo in candidate_combinations:
            plan = self._build_plan(combo, profile, prefs)
            plans.append(plan)

        # ---- 排序 ----
        plans.sort(key=lambda p: p.score, reverse=True)

        # ---- 限制数量 ----
        n_min = prefs.min_plans
        n_max = prefs.max_plans if top_k is None else min(top_k, prefs.max_plans)
        plans = plans[:max(n_min, n_max)]

        logger.info(
            "推荐完成，共 %d 套方案 | 首选: %s (score=%.3f)",
            len(plans),
            plans[0].name if plans else "无",
            plans[0].score if plans else 0.0,
        )
        return plans

    def recommend_from_dataframe(
        self,
        df:        pd.DataFrame,
        direction: Optional[List[int]] = None,
        time_col:  Optional[str] = None,
    ) -> List[AlgorithmPlan]:
        """直接从 DataFrame 自动分析并推荐"""
        profile = DataProfile.from_dataframe(df, direction, time_col)
        logger.info(
            "数据画像: n_objects=%d, n_indicators=%d, "
            "max_corr=%.3f, missing=%.1f%%",
            profile.n_objects, profile.n_indicators,
            profile.max_corr, profile.missing_rate * 100,
        )
        return self.recommend(profile)

    def print_recommendations(self, plans: List[AlgorithmPlan]) -> None:
        """打印推荐结果表格"""
        print("\n" + "=" * 70)
        print("  算法组合推荐结果")
        print("=" * 70)
        for i, plan in enumerate(plans):
            print(f"\n  [方案 {i+1}] {plan.name}  (推荐得分: {plan.score:.3f})")
            print(f"    赋权方法: {plan.weight_method}")
            print(f"    评价方法: {plan.eval_method}")
            print(f"    预处理 : {', '.join(plan.preprocess_steps) or '标准化'}")
            print(f"    推荐理由:")
            for r in plan.reasons:
                print(f"      ✓ {r}")
            if plan.warnings:
                print(f"    注意事项:")
                for w in plan.warnings:
                    print(f"      ⚠ {w}")
        print("=" * 70 + "\n")

    def generate_workflow_config(
        self,
        plan: AlgorithmPlan,
        file_path: str,
        object_col: str = "name",
        direction: Optional[List[int]] = None,
    ) -> Dict:
        """
        根据推荐方案生成工作流配置字典

        Parameters
        ----------
        plan : AlgorithmPlan
            选定的推荐方案
        file_path : str
            数据文件路径
        object_col : str
            评价对象列名
        direction : list
            指标方向

        Returns
        -------
        dict  可直接传给 Workflow.from_config()
        """
        nodes = [
            {
                "node_id":      "load",
                "type":         "DataLoadNode",
                "dependencies": [],
                "config": {
                    "file_path":  file_path,
                    "object_col": object_col,
                    "direction":  direction or [],
                },
            },
            {
                "node_id":      "preprocess",
                "type":         "PreprocessNode",
                "dependencies": ["load"],
                "config": {
                    "method":         "minmax",
                    "handle_missing": "mean",
                    "handle_outliers": plan.preprocess_steps.__contains__("outlier"),
                },
            },
            {
                "node_id":      "weight",
                "type":         "WeightNode",
                "dependencies": ["preprocess"],
                "config": {
                    "method": plan.weight_method,
                },
            },
            {
                "node_id":      "eval",
                "type":         "EvaluationNode",
                "dependencies": ["weight"],
                "config": {
                    "method": plan.eval_method,
                },
            },
        ]

        if plan.sensitivity:
            nodes.append({
                "node_id":      "sensitivity",
                "type":         "SensitivityNode",
                "dependencies": ["eval"],
                "config": {
                    "perturbation_range": 0.3,
                    "n_steps":            21,
                },
            })

        nodes.append({
            "node_id":      "consolidate",
            "type":         "ConsolidationNode",
            "dependencies": (
                ["sensitivity"] if plan.sensitivity else ["eval"]
            ),
            "config": {},
        })

        config = {
            "workflow_name": f"AutoEval_{plan.plan_id}",
            "fail_fast":     True,
            "max_retries":   1,
            "parallel":      False,
            "nodes":         nodes,
        }
        return config

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_candidate_combinations(
        self,
        profile: DataProfile,
        prefs:   UserPreferences,
    ) -> List[Dict]:
        """生成候选的赋权-评价组合"""
        weight_methods = self._select_weight_methods(profile, prefs)
        eval_methods   = self._select_eval_methods(profile, prefs)

        combinations = []
        for wm in weight_methods:
            for em in eval_methods:
                combinations.append({
                    "weight_method": wm,
                    "eval_method":   em,
                })
        return combinations

    def _select_weight_methods(
        self,
        profile: DataProfile,
        prefs:   UserPreferences,
    ) -> List[str]:
        """基于规则选择候选赋权方法"""
        methods = []

        if prefs.prefer_objective or True:  # 默认总是包含客观方法
            if profile.max_corr > 0.6 or profile.data_variance < 0.15:
                methods.extend(["critic", "entropy"])
            else:
                methods.extend(["entropy", "critic"])

        if prefs.allow_subjective:
            if prefs.ahp_matrix is not None:
                methods.insert(0, "ahp")
            methods.append("combination")

        return methods[:3]  # 最多考虑3种赋权方法

    def _select_eval_methods(
        self,
        profile: DataProfile,
        prefs:   UserPreferences,
    ) -> List[str]:
        """基于规则选择候选评价方法"""
        methods = []

        if prefs.need_fuzzy:
            methods.append("fuzzy")
        elif prefs.need_classification:
            methods.extend(["rsr", "topsis"])
        elif prefs.require_dea:
            methods.extend(["dea", "topsis"])
        elif profile.ratio_n_p < 0.8:
            methods.extend(["gra", "topsis"])
        else:
            methods.extend(["topsis", "vikor", "gra"])

        return methods[:3]  # 最多考虑3种评价方法

    def _build_plan(
        self,
        combo:   Dict,
        profile: DataProfile,
        prefs:   UserPreferences,
    ) -> AlgorithmPlan:
        """构建单个算法方案（含评分与理由）"""
        wm = combo["weight_method"]
        em = combo["eval_method"]

        reasons  = []
        warnings = []
        score    = 0.5  # 基础分

        # ---- 赋权方法评分 ----
        for rule in self._WEIGHT_RULES:
            try:
                if rule["condition"](profile, prefs):
                    if rule.get("weight_method") == wm:
                        score += rule.get("score_bonus", 0)
                        reasons.append(rule["reason"])
                        if "warning" in rule:
                            warnings.append(rule["warning"])
            except Exception:
                pass

        # ---- 评价方法评分 ----
        for rule in self._EVAL_RULES:
            try:
                if rule["condition"](profile, prefs):
                    if rule.get("eval_method") == em:
                        score += rule.get("score_bonus", 0)
                        reasons.append(rule["reason"])
            except Exception:
                pass

        # ---- 默认理由（如果规则未触发）----
        if not reasons:
            reasons = [
                f"{wm} 是适合当前数据规模的客观赋权方法",
                f"{em} 是常用的多属性综合评价方法",
            ]

        # ---- 预处理步骤 ----
        preprocess_steps = ["normalize"]
        if profile.missing_rate > 0.05:
            preprocess_steps.insert(0, "impute")
            warnings.append(f"缺失率 {profile.missing_rate*100:.1f}%，建议检查数据质量")
        if profile.max_corr > 0.85:
            preprocess_steps.append("dimension_reduction")
            warnings.append("指标间高度相关，可考虑PCA降维或合并相关指标")

        # ---- 方案名称 ----
        wm_display = {
            "entropy": "熵权法", "critic": "CRITIC法",
            "ahp": "AHP法", "std_dev": "标准差法",
            "combination": "组合赋权",
        }.get(wm, wm)
        em_display = {
            "topsis": "TOPSIS", "vikor": "VIKOR",
            "gra": "灰色关联", "fuzzy": "模糊综合",
            "rsr": "秩和比(RSR)", "dea": "DEA",
        }.get(em, em)

        plan_name = f"{wm_display}-{em_display}综合评价模型"
        plan_id   = f"{wm}_{em}"

        return AlgorithmPlan(
            plan_id          = plan_id,
            name             = plan_name,
            description      = f"使用{wm_display}确定权重，{em_display}进行综合排序",
            weight_method    = wm,
            eval_method      = em,
            preprocess_steps = preprocess_steps,
            sensitivity      = True,
            rank_consistency = len([c for c in self._select_eval_methods(profile, prefs)]) > 1,
            score            = round(min(max(score, 0.0), 1.0), 4),
            reasons          = reasons,
            warnings         = warnings,
        )