# examples/test_foundation.py
"""
Part 1 基础设施验证脚本
运行: python examples/test_foundation.py
"""

import sys
from pathlib import Path

# 添加 src 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from utils.logging_config import setup_logger
from utils.config_loader import ConfigLoader, WorkflowConfig
from utils.file_handler import FileHandler
from algorithms.base import (
    BaseMethod, MethodResult, MethodCategory, IndicatorDirection
)

# ─── 1. 日志系统 ───────────────────────────────────────────
logger = setup_logger(
    name="demo",
    level="debug",
    log_to_file=True,
    log_dir="output/logs"
)
logger.info("✅ 日志系统初始化成功")

# ─── 2. 文件处理 ────────────────────────────────────────────
fh = FileHandler(output_root="output")
logger.info(f"✅ 文件处理器初始化，输出目录: {fh.output_root.resolve()}")

# 生成测试数据并保存
test_df = pd.DataFrame({
    "城市":   ["北京", "上海", "广州", "深圳", "杭州"],
    "R&D投入": [1200, 980, 720, 1100, 650],
    "专利数":  [8500, 7200, 4300, 9100, 3800],
    "GDP增长": [6.1, 5.8, 6.5, 7.2, 7.8],
})
saved = fh.save_dataframe(test_df, "test_cities", subdir="intermediate")
logger.info(f"✅ 测试数据保存至: {saved}")

# ─── 3. 配置加载 ────────────────────────────────────────────
# 构造最小化配置字典
config_dict = {
    "name": "城市创新能力评价（测试）",
    "data": {
        "file_path": str(saved),
        "object_col": "城市",
        "indicator_cols": ["R&D投入", "专利数", "GDP增长"],
        "indicator_directions": {
            "R&D投入": "positive",
            "专利数": "positive",
            "GDP增长": "positive",
        }
    },
    "weight": {"method": "entropy"},
    "evaluation": {"method": "topsis"},
}

loader = ConfigLoader()
config = loader.load_from_dict(config_dict)
logger.info(f"✅ 配置加载成功: name='{config.name}'")
logger.info(f"   权重方法: {config.weight.method}")
logger.info(f"   评价方法: {config.evaluation.method}")
logger.info(f"   输出语言: {config.output.language}")

# ─── 4. BaseMethod 接口验证 ────────────────────────────────

class SimpleScoreMethod(BaseMethod):
    """最简评价：标准化后加权求和"""
    CATEGORY = MethodCategory.EVALUATION
    METHOD_NAME_ZH = "简单加权评价"
    METHOD_NAME_EN = "SimpleWeightedScore"
    METHOD_ABBR = "SWS"

    def fit(self, data: pd.DataFrame, weights=None, **kwargs):
        self.validate_input(data)
        self._raw_data = data.copy()
        self._indicator_names = list(data.columns)
        self._object_names = list(data.index)
        n = data.shape[1]
        self._weights = (
            self._normalize_weights(weights)
            if weights is not None
            else np.ones(n) / n
        )
        self._is_fitted = True
        return self

    def compute(self) -> MethodResult:
        self._check_fitted()
        t0 = self._start_timer()

        # 简单 min-max 归一化后加权求和
        df = self._raw_data
        normalized = (df - df.min()) / (df.max() - df.min() + 1e-12)
        scores = normalized.dot(self._weights)
        scores.name = "综合得分"
        rankings = self._rank_scores(scores, ascending=False)

        weight_series = pd.Series(
            self._weights,
            index=self._indicator_names,
            name="权重"
        )

        result = self._build_result(
            scores=scores,
            rankings=rankings,
            weights=weight_series,
            scalars={"n_objects": len(scores), "n_indicators": len(self._weights)},
            elapsed=self._stop_timer(t0),
        )
        self._result = result
        return result

    def summary(self) -> str:
        r = self.get_result()
        df = r.get_summary_dataframe()
        text = f"\n{'='*45}\n{self.METHOD_NAME_ZH} 结果摘要\n{'='*45}\n{df}\n"
        print(text)
        return text

    def tex_description(self) -> str:
        return r"""
\subsection{简单加权评价模型}
本模型首先对各指标进行 Min-Max 归一化处理，
随后计算加权综合得分：
\begin{equation}
    S_i = \sum_{j=1}^{m} w_j \cdot x_{ij}^*
\end{equation}
其中 $x_{ij}^*$ 为归一化后的指标值，$w_j$ 为第 $j$ 个指标的权重。
"""


# 使用数值型测试数据
eval_data = pd.DataFrame(
    {
        "R&D投入": [1200.0, 980.0, 720.0, 1100.0, 650.0],
        "专利数":  [8500.0, 7200.0, 4300.0, 9100.0, 3800.0],
        "GDP增长": [6.1, 5.8, 6.5, 7.2, 7.8],
    },
    index=["北京", "上海", "广州", "深圳", "杭州"],
)

method = SimpleScoreMethod(language="zh")
result = method.fit_compute(eval_data)

logger.info(f"✅ BaseMethod 接口验证完成")
method.summary()
logger.info(f"   耗时: {result.elapsed_time:.4f}s")
logger.info(f"   第一名: {result.rankings.idxmin()}")
logger.info(f"   结果结构: {result}")

# ─── 5. MethodResult 序列化验证 ────────────────────────────
result_dict = result.to_dict()
logger.info(f"✅ MethodResult 序列化: keys={list(result_dict.keys())}")

# 保存中间结果
fh.save_dict(result_dict, "test_result", subdir="intermediate")
logger.info("✅ 中间结果已保存")

print("\n" + "="*50)
print("Part 1 基础设施验证全部通过 ✅")
print("="*50)