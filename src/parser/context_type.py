"""
评价情境类型定义模块

定义所有支持的评价情境标签、情境特征描述以及匹配规则。
每个情境标签关联一组推荐算法路径和适用条件。
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple


class ContextType(Enum):
    """评价情境类型枚举"""

    # 多属性综合排序（最常见）
    MULTI_ATTRIBUTE_RANKING = auto()

    # 风险评估
    RISK_ASSESSMENT = auto()

    # 绩效基准对标
    PERFORMANCE_BENCHMARKING = auto()

    # 方案优选/决策
    SCHEME_SELECTION = auto()

    # 动态时序评价
    DYNAMIC_TEMPORAL = auto()

    # 分类分级评价
    CLASSIFICATION_GRADING = auto()

    # 模糊综合评价（含定性指标较多）
    FUZZY_COMPREHENSIVE = auto()

    # 效率评价（DEA类）
    EFFICIENCY_EVALUATION = auto()

    # 区域/空间对比评价
    REGIONAL_COMPARISON = auto()

    # 满意度/质量评价
    SATISFACTION_QUALITY = auto()


@dataclass
class ContextFeature:
    """
    情境特征描述类

    Attributes:
        context_type: 情境类型枚举
        description: 情境的自然语言描述
        keywords: 用于匹配该情境的关键词列表
        negative_keywords: 排除关键词（出现时降低匹配概率）
        typical_indicators: 该情境下常见的指标类型
        recommended_workflows: 推荐的工作流路径（按优先级排列）
        data_requirements: 对数据的基本要求
        min_confidence: 最低匹配置信度阈值
    """

    context_type: ContextType
    description: str
    keywords: List[str] = field(default_factory=list)
    negative_keywords: List[str] = field(default_factory=list)
    typical_indicators: List[str] = field(default_factory=list)
    recommended_workflows: List[str] = field(default_factory=list)
    data_requirements: Dict[str, any] = field(default_factory=dict)
    min_confidence: float = 0.3

    def match_score(self, text: str, entities: Dict[str, List[str]] = None) -> float:
        """
        计算输入文本与当前情境的匹配分数

        Args:
            text: 用户输入的问题描述文本（已分词/小写化）
            entities: 从NLP解析器提取的实体字典

        Returns:
            匹配分数 [0, 1]
        """
        score = 0.0
        max_possible = len(self.keywords) + 2  # 基础分母

        # 关键词命中计分
        keyword_hits = 0
        for kw in self.keywords:
            if kw in text:
                keyword_hits += 1
                # 核心关键词权重更高（前3个视为核心）
                if self.keywords.index(kw) < 3:
                    score += 2.0
                else:
                    score += 1.0

        # 负面关键词扣分
        for nkw in self.negative_keywords:
            if nkw in text:
                score -= 1.5

        # 实体匹配加分
        if entities:
            if entities.get("eval_objects") and len(entities["eval_objects"]) > 1:
                score += 0.5
            if entities.get("indicators") and len(entities["indicators"]) >= 3:
                score += 0.5
            # 时序关键词匹配
            if entities.get("temporal_markers") and self.context_type == ContextType.DYNAMIC_TEMPORAL:
                score += 2.0

        # 归一化到 [0, 1]
        normalized_score = min(max(score / max_possible, 0.0), 1.0)
        return normalized_score


# ==================== 情境注册表 ====================

CONTEXT_REGISTRY: List[ContextFeature] = [
    ContextFeature(
        context_type=ContextType.MULTI_ATTRIBUTE_RANKING,
        description="对多个评价对象基于多个属性指标进行综合排序",
        keywords=[
            "综合评价", "排序", "排名", "优劣", "综合得分",
            "多指标", "多属性", "评价体系", "指标体系",
            "竞争力", "发展水平", "能力评价", "对比分析",
            "ranking", "comprehensive", "evaluation"
        ],
        negative_keywords=["动态", "时序", "趋势", "效率", "DEA"],
        typical_indicators=["经济", "社会", "环境", "创新", "发展"],
        recommended_workflows=[
            "entropy_topsis",
            "critic_topsis",
            "ahp_topsis",
            "pca_entropy_topsis",
            "entropy_vikor"
        ],
        data_requirements={
            "min_objects": 3,
            "min_indicators": 3,
            "data_type": "numerical"
        },
    ),
    ContextFeature(
        context_type=ContextType.RISK_ASSESSMENT,
        description="对风险因素进行识别、量化和等级评估",
        keywords=[
            "风险", "安全", "隐患", "危险", "威胁",
            "脆弱性", "暴露度", "风险等级", "预警",
            "risk", "hazard", "vulnerability", "assessment"
        ],
        negative_keywords=["排序", "竞争力"],
        typical_indicators=["概率", "损失", "频率", "严重度", "暴露"],
        recommended_workflows=[
            "ahp_fuzzy_eval",
            "entropy_gra",
            "risk_matrix",
            "ahp_topsis"
        ],
        data_requirements={
            "min_objects": 2,
            "min_indicators": 3,
            "allows_qualitative": True
        },
    ),
    ContextFeature(
        context_type=ContextType.PERFORMANCE_BENCHMARKING,
        description="对绩效、业绩表现进行基准对标与差距分析",
        keywords=[
            "绩效", "业绩", "考核", "达标", "目标",
            "KPI", "对标", "基准", "效果评估", "performance",
            "benchmark", "assessment"
        ],
        negative_keywords=["风险", "分类"],
        typical_indicators=["完成率", "增长率", "利润", "产出", "效率"],
        recommended_workflows=[
            "entropy_topsis",
            "critic_vikor",
            "rsr_ranking",
            "entropy_gra"
        ],
        data_requirements={
            "min_objects": 3,
            "min_indicators": 3,
            "data_type": "numerical"
        },
    ),
    ContextFeature(
        context_type=ContextType.SCHEME_SELECTION,
        description="从多个候选方案中选出最优方案",
        keywords=[
            "方案", "选择", "优选", "决策", "选取",
            "最优", "比选", "备选", "候选", "替代方案",
            "selection", "alternative", "decision"
        ],
        negative_keywords=["时序", "动态", "趋势"],
        typical_indicators=["成本", "收益", "可行性", "风险", "时间"],
        recommended_workflows=[
            "ahp_topsis",
            "ahp_vikor",
            "electre_iii",
            "fuzzy_ahp_topsis"
        ],
        data_requirements={
            "min_objects": 2,
            "min_indicators": 2,
            "allows_qualitative": True
        },
    ),
    ContextFeature(
        context_type=ContextType.DYNAMIC_TEMPORAL,
        description="对评价对象进行多时间段的动态跟踪评价",
        keywords=[
            "动态", "时序", "年度", "趋势", "变化",
            "历年", "逐年", "时间序列", "演变", "发展趋势",
            "temporal", "dynamic", "trend", "time-series"
        ],
        negative_keywords=["静态", "截面"],
        typical_indicators=["年份", "季度", "月份", "增长", "变化率"],
        recommended_workflows=[
            "dynamic_entropy_topsis",
            "dynamic_gra",
            "time_weighted_topsis",
            "panel_entropy_topsis"
        ],
        data_requirements={
            "min_objects": 2,
            "min_indicators": 3,
            "requires_temporal": True,
            "min_time_periods": 3
        },
    ),
    ContextFeature(
        context_type=ContextType.CLASSIFICATION_GRADING,
        description="将评价对象分为多个等级或类别",
        keywords=[
            "分类", "分级", "等级", "聚类", "划分",
            "类别", "级别", "分档", "档次", "归类",
            "classification", "grading", "clustering"
        ],
        negative_keywords=["排序", "排名"],
        typical_indicators=["类型", "等级", "标准", "阈值"],
        recommended_workflows=[
            "clustering_topsis",
            "som_topsis",
            "fuzzy_clustering_eval",
            "rsr_grading"
        ],
        data_requirements={
            "min_objects": 5,
            "min_indicators": 3,
            "data_type": "numerical"
        },
    ),
    ContextFeature(
        context_type=ContextType.FUZZY_COMPREHENSIVE,
        description="含大量定性指标或模糊语义的综合评价",
        keywords=[
            "模糊", "定性", "语义", "等级评定", "专家评分",
            "模糊综合", "隶属度", "评语集", "模糊判断",
            "fuzzy", "qualitative", "linguistic"
        ],
        negative_keywords=["定量", "精确"],
        typical_indicators=["优", "良", "中", "差", "满意度"],
        recommended_workflows=[
            "ahp_fuzzy_eval",
            "fuzzy_topsis",
            "multilevel_fuzzy",
            "ahp_grey_fuzzy"
        ],
        data_requirements={
            "min_objects": 2,
            "min_indicators": 3,
            "allows_qualitative": True,
            "has_expert_scores": True
        },
    ),
    ContextFeature(
        context_type=ContextType.EFFICIENCY_EVALUATION,
        description="评价决策单元的投入产出效率",
        keywords=[
            "效率", "投入产出", "DEA", "包络", "生产率",
            "技术效率", "规模效率", "全要素", "产能",
            "efficiency", "productivity", "DEA"
        ],
        negative_keywords=["排序", "模糊"],
        typical_indicators=["投入", "产出", "人员", "资金", "成果"],
        recommended_workflows=[
            "dea_ccr",
            "dea_bcc",
            "malmquist_dea",
            "super_efficiency_dea"
        ],
        data_requirements={
            "min_objects": 5,
            "min_indicators": 2,
            "requires_io_separation": True
        },
    ),
    ContextFeature(
        context_type=ContextType.REGIONAL_COMPARISON,
        description="对地区、城市、国家等进行空间对比评价",
        keywords=[
            "地区", "城市", "省份", "区域", "国家",
            "空间", "区位", "东中西", "对比", "差异",
            "regional", "spatial", "city", "province"
        ],
        negative_keywords=["企业", "个人", "产品"],
        typical_indicators=["GDP", "人口", "面积", "人均", "密度"],
        recommended_workflows=[
            "entropy_topsis",
            "pca_entropy_topsis",
            "critic_gra",
            "clustering_topsis"
        ],
        data_requirements={
            "min_objects": 3,
            "min_indicators": 3,
            "data_type": "numerical"
        },
    ),
    ContextFeature(
        context_type=ContextType.SATISFACTION_QUALITY,
        description="满意度调查或质量评价",
        keywords=[
            "满意度", "质量", "服务", "体验", "评分",
            "调查", "问卷", "打分", "客户", "用户",
            "satisfaction", "quality", "survey"
        ],
        negative_keywords=["效率", "DEA"],
        typical_indicators=["满意", "一般", "不满意", "评分", "星级"],
        recommended_workflows=[
            "ahp_fuzzy_eval",
            "entropy_topsis",
            "fuzzy_comprehensive",
            "rsr_ranking"
        ],
        data_requirements={
            "min_objects": 3,
            "min_indicators": 3,
            "allows_qualitative": True
        },
    ),
]


def get_context_by_tag(context_type: ContextType) -> Optional[ContextFeature]:
    """根据情境类型枚举获取对应的情境特征对象"""
    for ctx in CONTEXT_REGISTRY:
        if ctx.context_type == context_type:
            return ctx
    return None


def get_all_context_types() -> List[ContextType]:
    """返回所有已注册的情境类型"""
    return [ctx.context_type for ctx in CONTEXT_REGISTRY]


def match_context(
    text: str,
    entities: Dict[str, List[str]] = None,
    top_k: int = 3
) -> List[Tuple[ContextType, float]]:
    """
    对输入文本进行情境匹配，返回按分数降序排列的Top-K情境

    Args:
        text: 预处理后的问题描述文本
        entities: NLP提取的实体字典
        top_k: 返回前K个匹配结果

    Returns:
        List of (ContextType, score) 元组
    """
    results = []
    for ctx_feature in CONTEXT_REGISTRY:
        score = ctx_feature.match_score(text, entities)
        if score >= ctx_feature.min_confidence:
            results.append((ctx_feature.context_type, score))

    # 按分数降序排列
    results.sort(key=lambda x: x[1], reverse=True)

    # 若无匹配，返回默认情境
    if not results:
        results = [(ContextType.MULTI_ATTRIBUTE_RANKING, 0.5)]

    return results[:top_k]