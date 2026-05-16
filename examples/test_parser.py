"""
使用示例：Parser模块的基本用法
"""
import pandas as pd
from src.parser import NLPParser, DataProfiler

# ========== 1. 自然语言问题解析 ==========
parser = NLPParser()

# 解析自然语言描述
problem = parser.parse(
    "请对北京、上海、广州、深圳、杭州5个城市的创新能力进行综合评价，"
    "评价指标包括R&D投入、专利申请量、高新企业数量、科技人才占比，"
    "所有指标均为正向指标，使用客观赋权法。"
)

print(problem.summary())
# 输出：
# 评价对象类型: 城市
# 评价对象数量: 5
# 评价指标数量: 4
# 识别情境: MULTI_ATTRIBUTE_RANKING (置信度: 0.75)
# 权重偏好: objective

# ========== 2. 数据特征分析 ==========
profiler = DataProfiler()

# 分析CSV文件
profile = profiler.analyze(
    "data/city_innovation.csv",
    id_col="城市"
)

print(profile.summary())
# 输出数据画像报告，包括缺失值、相关性、推荐预处理等

# ========== 3. 从表单数据创建问题描述 ==========
form_problem = parser.parse_from_form({
    "description": "城市创新能力评价",
    "eval_object_type": "城市",
    "eval_objects": ["北京", "上海", "广州", "深圳", "杭州"],
    "indicators": ["R&D投入", "专利数", "高新企业数", "科技人才占比"],
    "indicator_directions": {
        "R&D投入": "positive",
        "专利数": "positive",
        "高新企业数": "positive",
        "科技人才占比": "positive"
    },
    "weight_preference": "objective",
    "data_file": "data/city_innovation.csv"
})