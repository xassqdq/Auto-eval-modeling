"""
自然语言问题解析器

基于关键词、正则表达式和可选的LLM接口，从用户输入的问题描述中
提取评价对象、评价指标、约束条件、偏好设置等结构化信息。
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from .context_type import (
    ContextType,
    ContextFeature,
    CONTEXT_REGISTRY,
    match_context,
)


@dataclass
class ProblemDescription:
    """
    结构化的问题描述

    从用户的自然语言输入或表单输入解析而成，
    作为下游工作流推荐与配置的输入。
    """

    # 原始文本
    raw_text: str = ""

    # 评价对象
    eval_objects: List[str] = field(default_factory=list)
    eval_object_type: str = ""  # 如 "城市", "企业", "方案"

    # 评价指标
    indicators: List[str] = field(default_factory=list)
    indicator_directions: Dict[str, str] = field(default_factory=dict)  # 指标名→"positive"/"negative"/"moderate"

    # 情境识别结果
    context_type: Optional[ContextType] = None
    context_confidence: float = 0.0
    alternative_contexts: List[Tuple[ContextType, float]] = field(default_factory=list)

    # 用户偏好
    weight_preference: str = "objective"  # "subjective", "objective", "combination"
    has_expert_data: bool = False
    has_temporal_data: bool = False
    num_time_periods: int = 1

    # 约束条件
    constraints: List[str] = field(default_factory=list)

    # 数据相关
    data_file_path: Optional[str] = None
    num_objects: int = 0
    num_indicators: int = 0

    # 额外元信息
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """生成问题描述的摘要"""
        lines = [
            "=" * 60,
            "问题解析结果摘要",
            "=" * 60,
            f"评价对象类型: {self.eval_object_type or '未识别'}",
            f"评价对象数量: {self.num_objects}",
            f"评价指标数量: {self.num_indicators}",
            f"识别情境: {self.context_type.name if self.context_type else '未识别'} "
            f"(置信度: {self.context_confidence:.2f})",
            f"权重偏好: {self.weight_preference}",
            f"是否含时序数据: {self.has_temporal_data}",
            f"是否含专家数据: {self.has_expert_data}",
        ]
        if self.alternative_contexts:
            lines.append("备选情境:")
            for ctx, score in self.alternative_contexts:
                lines.append(f"  - {ctx.name}: {score:.2f}")
        if self.constraints:
            lines.append(f"约束条件: {', '.join(self.constraints)}")
        lines.append("=" * 60)
        return "\n".join(lines)


class NLPParser:
    """
    基于规则与关键词的自然语言问题解析器

    主要功能：
    1. 从自然语言描述中提取评价对象、指标等关键实体
    2. 识别问题的评价情境类型
    3. 推断用户对权重方法的偏好
    4. 生成结构化的 ProblemDescription 对象
    """

    # 评价对象类型关键词映射
    OBJECT_TYPE_KEYWORDS = {
        "城市": ["城市", "市", "都市", "城镇", "city", "cities"],
        "省份": ["省", "省份", "自治区", "直辖市", "province"],
        "地区": ["地区", "区域", "区", "region", "area"],
        "国家": ["国家", "国", "country", "nation"],
        "企业": ["企业", "公司", "集团", "firm", "company", "enterprise"],
        "方案": ["方案", "策略", "选项", "plan", "scheme", "alternative"],
        "产品": ["产品", "商品", "服务", "product", "service"],
        "高校": ["高校", "大学", "学校", "university", "college"],
        "医院": ["医院", "医疗机构", "hospital"],
        "项目": ["项目", "工程", "project"],
    }

    # 指标方向关键词
    POSITIVE_KEYWORDS = [
        "越大越好", "越高越好", "越多越好", "正向", "效益型",
        "收入", "利润", "产出", "增长", "得分"
    ]
    NEGATIVE_KEYWORDS = [
        "越小越好", "越低越好", "越少越好", "负向", "成本型",
        "成本", "费用", "损失", "风险", "污染", "能耗"
    ]
    MODERATE_KEYWORDS = [
        "适度", "适中", "中间型", "区间型", "最优值"
    ]

    # 权重偏好关键词
    SUBJECTIVE_WEIGHT_KEYWORDS = [
        "专家", "经验", "主观", "AHP", "层次分析", "判断矩阵",
        "德尔菲", "打分", "评分"
    ]
    OBJECTIVE_WEIGHT_KEYWORDS = [
        "客观", "熵权", "数据驱动", "CRITIC", "信息熵",
        "标准离差", "变异系数"
    ]
    COMBINATION_WEIGHT_KEYWORDS = [
        "组合", "综合赋权", "主客观结合", "博弈论赋权"
    ]

    # 时序相关关键词
    TEMPORAL_KEYWORDS = [
        "年", "季度", "月", "时期", "阶段", "历年",
        "动态", "时序", "趋势", "变化", "演变",
        r"\d{4}", "2019", "2020", "2021", "2022", "2023", "2024"
    ]

    # 数量词正则
    NUM_PATTERN = re.compile(r"(\d+)\s*[个家座所项条]")
    YEAR_RANGE_PATTERN = re.compile(r"(\d{4})\s*[-至到~]\s*(\d{4})")
    YEAR_PATTERN = re.compile(r"(20\d{2}|19\d{2})")

    def __init__(self, use_llm: bool = False, llm_config: Dict = None):
        """
        初始化解析器

        Args:
            use_llm: 是否启用LLM增强解析
            llm_config: LLM配置（API地址、模型名等）
        """
        self.use_llm = use_llm
        self.llm_config = llm_config or {}

    def parse(self, text: str, **kwargs) -> ProblemDescription:
        """
        解析用户输入的问题描述

        Args:
            text: 自然语言问题描述
            **kwargs: 额外的用户输入（如 data_file, weight_preference 等）

        Returns:
            结构化的 ProblemDescription 对象
        """
        # 初始化结果
        result = ProblemDescription(raw_text=text)

        # 文本预处理
        processed_text = self._preprocess_text(text)

        # 1. 提取评价对象类型
        result.eval_object_type = self._extract_object_type(processed_text)

        # 2. 提取评价对象列表（如果文本中明确列出）
        result.eval_objects = self._extract_object_list(processed_text)

        # 3. 提取指标信息
        result.indicators = self._extract_indicators(processed_text)

        # 4. 推断指标方向
        result.indicator_directions = self._infer_indicator_directions(
            processed_text, result.indicators
        )

        # 5. 识别时序特征
        result.has_temporal_data, result.num_time_periods = self._detect_temporal(
            processed_text
        )

        # 6. 推断权重偏好
        result.weight_preference = self._infer_weight_preference(processed_text)

        # 7. 检测是否有专家数据
        result.has_expert_data = self._detect_expert_data(processed_text)

        # 8. 情境匹配
        entities = {
            "eval_objects": result.eval_objects,
            "indicators": result.indicators,
            "temporal_markers": self._extract_temporal_markers(processed_text),
        }
        context_matches = match_context(processed_text, entities, top_k=3)

        if context_matches:
            result.context_type = context_matches[0][0]
            result.context_confidence = context_matches[0][1]
            result.alternative_contexts = context_matches[1:]

        # 9. 提取约束条件
        result.constraints = self._extract_constraints(processed_text)

        # 10. 处理额外参数
        if kwargs.get("data_file"):
            result.data_file_path = kwargs["data_file"]
        if kwargs.get("weight_preference"):
            result.weight_preference = kwargs["weight_preference"]
        if kwargs.get("num_objects"):
            result.num_objects = kwargs["num_objects"]
        if kwargs.get("num_indicators"):
            result.num_indicators = kwargs["num_indicators"]

        # 估算对象与指标数量（如果未显式提供）
        if result.num_objects == 0:
            result.num_objects = self._estimate_num_objects(processed_text, result)
        if result.num_indicators == 0:
            result.num_indicators = len(result.indicators) if result.indicators else 0

        # 11. 可选：LLM 增强解析
        if self.use_llm:
            result = self._llm_enhance(result)

        return result

    def parse_from_form(self, form_data: Dict[str, Any]) -> ProblemDescription:
        """
        从结构化表单数据创建 ProblemDescription

        Args:
            form_data: 包含以下键的字典
                - description: 问题描述文本
                - eval_object_type: 评价对象类型
                - indicators: 指标列表
                - indicator_directions: 指标方向字典
                - weight_preference: 权重偏好
                - data_file: 数据文件路径
                - ...

        Returns:
            ProblemDescription 对象
        """
        result = ProblemDescription()

        result.raw_text = form_data.get("description", "")
        result.eval_object_type = form_data.get("eval_object_type", "")
        result.eval_objects = form_data.get("eval_objects", [])
        result.indicators = form_data.get("indicators", [])
        result.indicator_directions = form_data.get("indicator_directions", {})
        result.weight_preference = form_data.get("weight_preference", "objective")
        result.has_expert_data = form_data.get("has_expert_data", False)
        result.has_temporal_data = form_data.get("has_temporal_data", False)
        result.num_time_periods = form_data.get("num_time_periods", 1)
        result.data_file_path = form_data.get("data_file", None)
        result.num_objects = form_data.get("num_objects", len(result.eval_objects))
        result.num_indicators = form_data.get("num_indicators", len(result.indicators))
        result.constraints = form_data.get("constraints", [])

        # 情境匹配
        if form_data.get("context_type"):
            try:
                result.context_type = ContextType[form_data["context_type"]]
                result.context_confidence = 1.0
            except KeyError:
                pass

        if result.context_type is None and result.raw_text:
            entities = {
                "eval_objects": result.eval_objects,
                "indicators": result.indicators,
                "temporal_markers": [],
            }
            context_matches = match_context(result.raw_text, entities, top_k=3)
            if context_matches:
                result.context_type = context_matches[0][0]
                result.context_confidence = context_matches[0][1]
                result.alternative_contexts = context_matches[1:]

        return result

    # ==================== 私有方法 ====================

    def _preprocess_text(self, text: str) -> str:
        """文本预处理：统一编码、去除多余空格、转小写（英文部分）"""
        # 保留中文原样，英文转小写
        text = text.strip()
        # 统一标点
        text = text.replace("，", ",").replace("。", ".").replace("；", ";")
        text = text.replace("（", "(").replace("）", ")")
        # 去除多余空白
        text = re.sub(r"\s+", " ", text)
        return text

    def _extract_object_type(self, text: str) -> str:
        """提取评价对象类型"""
        for obj_type, keywords in self.OBJECT_TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in text.lower():
                    return obj_type
        return "对象"  # 默认

    def _extract_object_list(self, text: str) -> List[str]:
        """
        从文本中提取明确列出的评价对象

        尝试匹配如 "A、B、C和D" 或 "A,B,C,D" 等模式
        """
        objects = []

        # 模式1: 中文顿号分隔 "北京、上海、广州"
        pattern_cn = re.compile(r"([\u4e00-\u9fa5]{2,})(?:[、,，][\u4e00-\u9fa5]{2,}){2,}")
        match = pattern_cn.search(text)
        if match:
            segment = match.group(0)
            objects = re.split(r"[、,，]", segment)

        # 模式2: "包括/分别是/有 A, B, C"
        if not objects:
            pattern_list = re.compile(
                r"(?:包括|分别是|分别为|有|如|为)\s*"
                r"([\u4e00-\u9fa5A-Za-z0-9]+(?:[、,，和与及]\s*[\u4e00-\u9fa5A-Za-z0-9]+)+)"
            )
            match = pattern_list.search(text)
            if match:
                segment = match.group(1)
                objects = re.split(r"[、,，和与及]\s*", segment)

        # 清理
        objects = [obj.strip() for obj in objects if len(obj.strip()) >= 2]
        return objects

    def _extract_indicators(self, text: str) -> List[str]:
        """
        从文本中提取指标名称

        匹配 "指标包括..." 或 "从...方面" 等模式
        """
        indicators = []

        # 模式1: "指标包括/有 A、B、C"
        pattern1 = re.compile(
            r"(?:指标|因素|维度|方面)(?:包括|有|为|分别是)\s*"
            r"([\u4e00-\u9fa5A-Za-z0-9/()（）%]+(?:[、,，和与及]\s*[\u4e00-\u9fa5A-Za-z0-9/()（）%]+)+)"
        )
        match = pattern1.search(text)
        if match:
            segment = match.group(1)
            indicators = re.split(r"[、,，和与及]\s*", segment)

        # 模式2: "从A、B、C等方面/角度"
        if not indicators:
            pattern2 = re.compile(
                r"从\s*([\u4e00-\u9fa5A-Za-z0-9]+(?:[、,，]\s*[\u4e00-\u9fa5A-Za-z0-9]+)+)\s*等?(?:方面|角度|维度|层面)"
            )
            match = pattern2.search(text)
            if match:
                segment = match.group(1)
                indicators = re.split(r"[、,，]\s*", segment)

        # 模式3: 括号内列举 "(A, B, C)"
        if not indicators:
            pattern3 = re.compile(r"[（(]([\u4e00-\u9fa5A-Za-z0-9、,，]+)[）)]")
            matches = pattern3.findall(text)
            for m in matches:
                parts = re.split(r"[、,，]\s*", m)
                if len(parts) >= 3:
                    indicators = parts
                    break

        # 清理
        indicators = [ind.strip() for ind in indicators if len(ind.strip()) >= 2]
        return indicators

    def _infer_indicator_directions(
        self, text: str, indicators: List[str]
    ) -> Dict[str, str]:
        """推断各指标的方向（正向/负向/适度）"""
        directions = {}
        for indicator in indicators:
            direction = "positive"  # 默认正向

            # 检查负向关键词
            for nkw in self.NEGATIVE_KEYWORDS:
                if nkw in indicator:
                    direction = "negative"
                    break

            # 检查适度关键词
            for mkw in self.MODERATE_KEYWORDS:
                if mkw in indicator:
                    direction = "moderate"
                    break

            directions[indicator] = direction

        # 检查文本中是否有整体说明
        if "成本型" in text or "越小越好" in text:
            # 可能需要更细致的上下文分析
            pass

        return directions

    def _detect_temporal(self, text: str) -> Tuple[bool, int]:
        """
        检测是否包含时序/动态评价需求

        Returns:
            (是否时序, 时间段数量)
        """
        has_temporal = False
        num_periods = 1

        # 检查时序关键词
        for kw in self.TEMPORAL_KEYWORDS:
            if re.search(kw, text):
                has_temporal = True
                break

        # 尝试提取年份范围
        year_range_match = self.YEAR_RANGE_PATTERN.search(text)
        if year_range_match:
            start_year = int(year_range_match.group(1))
            end_year = int(year_range_match.group(2))
            num_periods = end_year - start_year + 1
            has_temporal = True
        else:
            # 计算文中出现的不同年份数
            years = set(self.YEAR_PATTERN.findall(text))
            if len(years) >= 2:
                has_temporal = True
                num_periods = len(years)

        return has_temporal, num_periods

    def _infer_weight_preference(self, text: str) -> str:
        """推断权重计算偏好"""
        subjective_score = sum(1 for kw in self.SUBJECTIVE_WEIGHT_KEYWORDS if kw in text)
        objective_score = sum(1 for kw in self.OBJECTIVE_WEIGHT_KEYWORDS if kw in text)
        combination_score = sum(1 for kw in self.COMBINATION_WEIGHT_KEYWORDS if kw in text)

        if combination_score > 0:
            return "combination"
        elif subjective_score > objective_score:
            return "subjective"
        elif objective_score > subjective_score:
            return "objective"
        else:
            return "objective"  # 默认客观赋权

    def _detect_expert_data(self, text: str) -> bool:
        """检测是否涉及专家数据"""
        expert_keywords = ["专家", "经验", "打分", "评分", "问卷", "调查", "德尔菲"]
        return any(kw in text for kw in expert_keywords)

    def _extract_temporal_markers(self, text: str) -> List[str]:
        """提取时间标记"""
        markers = []
        years = self.YEAR_PATTERN.findall(text)
        markers.extend(years)

        temporal_words = ["年度", "季度", "月度", "逐年", "历年"]
        for tw in temporal_words:
            if tw in text:
                markers.append(tw)

        return markers

    def _extract_constraints(self, text: str) -> List[str]:
        """提取约束条件"""
        constraints = []

        constraint_patterns = [
            (r"(?:要求|需要|必须|应当)\s*(.{5,30})", "requirement"),
            (r"(?:不超过|至少|最多|最少)\s*(.{3,20})", "boundary"),
            (r"(?:假设|假定)\s*(.{5,40})", "assumption"),
        ]

        for pattern, ctype in constraint_patterns:
            matches = re.findall(pattern, text)
            for m in matches:
                constraints.append(f"[{ctype}] {m.strip()}")

        return constraints

    def _estimate_num_objects(self, text: str, result: ProblemDescription) -> int:
        """估算评价对象数量"""
        # 如果已经提取到对象列表
        if result.eval_objects:
            return len(result.eval_objects)

        # 从文本中提取数字
        num_match = self.NUM_PATTERN.search(text)
        if num_match:
            return int(num_match.group(1))

        return 0

    def _llm_enhance(self, result: ProblemDescription) -> ProblemDescription:
        """
        使用LLM增强解析结果（可选功能）

        当规则引擎解析不充分时，调用LLM进行补充解析。
        """
        if not self.use_llm:
            return result

        try:
            # 构建LLM提示词
            prompt = self._build_llm_prompt(result)

            # 调用LLM（这里仅定义接口，具体实现根据配置）
            llm_response = self._call_llm(prompt)

            # 解析LLM输出并补充结果
            if llm_response:
                result = self._merge_llm_result(result, llm_response)

        except Exception as e:
            # LLM调用失败不影响主流程
            result.metadata["llm_error"] = str(e)

        return result

    def _build_llm_prompt(self, result: ProblemDescription) -> str:
        """构建发送给LLM的提示词"""
        prompt = f"""请分析以下评价类数学建模问题，提取关键信息：

问题描述：{result.raw_text}

请以JSON格式输出以下信息：
1. eval_object_type: 评价对象类型
2. indicators: 评价指标列表
3. indicator_directions: 每个指标的方向（positive/negative/moderate）
4. context_type: 最匹配的评价情境（从以下选择：MULTI_ATTRIBUTE_RANKING, RISK_ASSESSMENT, PERFORMANCE_BENCHMARKING, SCHEME_SELECTION, DYNAMIC_TEMPORAL, CLASSIFICATION_GRADING, FUZZY_COMPREHENSIVE, EFFICIENCY_EVALUATION, REGIONAL_COMPARISON, SATISFACTION_QUALITY）
5. weight_preference: 权重偏好（subjective/objective/combination）
6. key_requirements: 关键需求列表

已识别信息（可补充或修正）：
- 对象类型: {result.eval_object_type}
- 已识别指标: {result.indicators}
- 时序数据: {result.has_temporal_data}
"""
        return prompt

    def _call_llm(self, prompt: str) -> Optional[Dict]:
        """
        调用LLM API

        支持多种后端：
        - OpenAI API
        - 本地模型（如通过 llama.cpp 服务）
        - 其他兼容接口
        """
        # 此处为接口预留，实际实现需要根据配置选择后端
        # 示例实现（需要安装 openai 包）：
        """
        import openai
        client = openai.OpenAI(
            api_key=self.llm_config.get("api_key"),
            base_url=self.llm_config.get("base_url")
        )
        response = client.chat.completions.create(
            model=self.llm_config.get("model", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        import json
        return json.loads(response.choices[0].message.content)
        """
        return None

    def _merge_llm_result(
        self, result: ProblemDescription, llm_output: Dict
    ) -> ProblemDescription:
        """将LLM输出与规则引擎结果合并"""
        # 优先保留规则引擎结果，LLM结果作为补充
        if not result.eval_object_type and llm_output.get("eval_object_type"):
            result.eval_object_type = llm_output["eval_object_type"]

        if not result.indicators and llm_output.get("indicators"):
            result.indicators = llm_output["indicators"]

        if not result.indicator_directions and llm_output.get("indicator_directions"):
            result.indicator_directions = llm_output["indicator_directions"]

        # LLM的情境识别作为参考
        if llm_output.get("context_type"):
            result.metadata["llm_context_suggestion"] = llm_output["context_type"]

        return result