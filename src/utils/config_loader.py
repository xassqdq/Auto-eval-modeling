# src/utils/config_loader.py
"""
配置文件加载与校验模块
使用 Pydantic v2 对 YAML 配置进行强类型校验与自动补全默认值
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ConfigDict

from .logging_config import get_logger

logger = get_logger("auto_eval.config")


# ============================================================
#  Pydantic 数据模型：算法参数配置
# ============================================================

class PreprocessConfig(BaseModel):
    """数据预处理配置"""
    model_config = ConfigDict(extra="allow")

    handle_missing: Literal["drop", "mean", "median", "zero", "interpolate"] = "mean"
    handle_outlier: Literal["none", "clip", "iqr", "zscore"] = "none"
    outlier_threshold: float = Field(3.0, ge=0.5, le=10.0,
                                     description="异常值判断阈值（Z-score 倍数）")
    normalization: Literal["minmax", "zscore", "vector", "none"] = "minmax"
    enable_pca: bool = False
    pca_variance_ratio: float = Field(0.95, ge=0.5, le=1.0,
                                      description="PCA 保留方差比例")
    check_correlation: bool = True
    correlation_threshold: float = Field(0.85, ge=0.0, le=1.0,
                                         description="高相关性判断阈值")


class WeightConfig(BaseModel):
    """赋权方法配置"""
    model_config = ConfigDict(extra="allow")

    method: Literal[
        "ahp", "entropy", "critic", "std_deviation",
        "combination_multiply", "combination_game", "equal"
    ] = "entropy"
    # AHP 专属
    judgment_matrix: Optional[list[list[float]]] = None
    consistency_threshold: float = Field(0.1, ge=0.0, le=1.0)
    # 组合赋权专属
    subjective_method: Optional[str] = None
    objective_method: Optional[str] = None
    combination_alpha: float = Field(0.5, ge=0.0, le=1.0,
                                     description="主观权重占比（组合赋权）")

    @field_validator("judgment_matrix")
    @classmethod
    def validate_matrix_square(cls, v):
        if v is not None:
            n = len(v)
            for row in v:
                if len(row) != n:
                    raise ValueError("判断矩阵必须为方阵")
        return v


class EvaluationConfig(BaseModel):
    """综合评价模型配置"""
    model_config = ConfigDict(extra="allow")

    method: Literal[
        "topsis", "vikor", "gra", "fuzzy",
        "electre", "rsr", "dea", "dynamic"
    ] = "topsis"
    # VIKOR 专属
    vikor_v: float = Field(0.5, ge=0.0, le=1.0, description="VIKOR 策略权重系数")
    # 灰色关联专属
    gra_rho: float = Field(0.5, ge=0.0, le=1.0, description="灰色关联分辨系数")
    # 模糊综合专属
    fuzzy_levels: list[str] = Field(
        default_factory=lambda: ["优", "良", "中", "差"],
        description="模糊评语集"
    )
    # 动态评价专属
    time_column: Optional[str] = None
    time_weights: Optional[list[float]] = None


class SensitivityConfig(BaseModel):
    """灵敏度分析配置"""
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    method: Literal["oat", "sobol", "monte_carlo"] = "oat"
    perturbation_range: float = Field(0.1, ge=0.01, le=0.5,
                                      description="OAT 扰动范围（±比例）")
    n_simulations: int = Field(1000, ge=100, le=100000,
                               description="Monte Carlo 模拟次数")


class OutputConfig(BaseModel):
    """输出格式配置"""
    model_config = ConfigDict(extra="allow")

    output_dir: str = "output"
    language: Literal["zh", "en"] = "zh"
    generate_latex: bool = True
    generate_code: bool = True
    generate_plots: bool = True
    plot_format: Literal["png", "svg", "pdf"] = "png"
    plot_dpi: int = Field(150, ge=72, le=600)
    latex_template: Literal["mcm", "cumcm", "academic"] = "cumcm"
    report_filename: str = "report"
    code_filename: str = "solution"


class DataConfig(BaseModel):
    """数据输入配置"""
    model_config = ConfigDict(extra="allow")

    file_path: str
    sheet_name: Optional[Union[str, int]] = 0
    index_col: Optional[str] = None
    object_col: Optional[str] = None
    indicator_cols: Optional[list[str]] = None
    indicator_directions: Optional[dict[str, Literal["positive", "negative", "moderate"]]] = None
    moderate_optimal: Optional[dict[str, float]] = None  # 适度型最优值
    encoding: str = "utf-8"

    @model_validator(mode="after")
    def validate_directions(self):
        if self.indicator_directions and self.indicator_cols:
            unknown = set(self.indicator_directions.keys()) - set(self.indicator_cols)
            if unknown:
                logger.warning(f"指标方向中存在未知指标列: {unknown}，将被忽略")
        return self


class AlgorithmConfig(BaseModel):
    """单个算法节点配置（工作流节点）"""
    model_config = ConfigDict(extra="allow")

    node_id: str = Field(description="节点唯一标识")
    node_type: Literal[
        "data_loading", "preprocessing", "weight_calculation",
        "evaluation", "sensitivity", "consolidation"
    ]
    enabled: bool = True
    depends_on: list[str] = Field(default_factory=list, description="前置节点ID列表")
    params: dict[str, Any] = Field(default_factory=dict, description="节点专属参数")


class WorkflowConfig(BaseModel):
    """
    完整工作流配置
    对应 configs/*.yaml 文件的顶层结构
    """
    model_config = ConfigDict(extra="allow")

    name: str = "AutoEval Workflow"
    description: str = ""
    version: str = "1.0"

    data: DataConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    weight: WeightConfig = Field(default_factory=WeightConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    sensitivity: SensitivityConfig = Field(default_factory=SensitivityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # 可选：细粒度节点配置（高级用法）
    nodes: Optional[list[AlgorithmConfig]] = None


# ============================================================
#  配置加载器
# ============================================================

class ConfigLoader:
    """
    YAML 配置文件加载与管理器

    功能：
    - 从 YAML 文件或 dict 加载配置
    - 用 Pydantic 模型进行类型校验与默认值补全
    - 支持配置合并（基础配置 + 用户覆盖）
    - 提供便捷的配置导出方法

    Example
    -------
    >>> loader = ConfigLoader()
    >>> config = loader.load("configs/entropy_topsis.yaml")
    >>> print(config.weight.method)
    'entropy'
    """

    # 预置配置目录（相对于项目根目录）
    DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"

    def __init__(self, config_dir: Optional[str | Path] = None):
        self._config_dir = Path(config_dir) if config_dir else self.DEFAULT_CONFIG_DIR
        self._config_cache: dict[str, WorkflowConfig] = {}

    def load(
        self,
        config_path: str | Path,
        override: Optional[dict[str, Any]] = None,
    ) -> WorkflowConfig:
        """
        从 YAML 文件加载工作流配置。

        Parameters
        ----------
        config_path : str | Path
            配置文件路径（相对于项目根目录或绝对路径）
        override : dict | None
            额外覆盖参数，优先级高于文件内容

        Returns
        -------
        WorkflowConfig
            校验后的工作流配置对象
        """
        config_path = Path(config_path)

        # 尝试相对路径
        if not config_path.is_absolute():
            full_path = self._config_dir / config_path
            if not full_path.exists():
                full_path = Path(config_path)
        else:
            full_path = config_path

        if not full_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {full_path}")

        cache_key = str(full_path.resolve())
        if cache_key in self._config_cache:
            logger.debug(f"从缓存加载配置: {cache_key}")
            return self._config_cache[cache_key]

        logger.info(f"正在加载配置文件: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            raw_data: dict = yaml.safe_load(f) or {}

        # 深度合并覆盖参数
        if override:
            raw_data = self._deep_merge(raw_data, override)

        try:
            config = WorkflowConfig(**raw_data)
        except Exception as e:
            logger.error(f"配置文件校验失败: {e}")
            raise ValueError(f"配置文件 '{full_path}' 格式错误: {e}") from e

        self._config_cache[cache_key] = config
        logger.info(f"配置加载成功: '{config.name}'")
        return config

    def load_from_dict(
        self,
        config_dict: dict[str, Any],
        name: str = "inline_config",
    ) -> WorkflowConfig:
        """
        从字典直接构建配置（适合代码内动态创建）。

        Parameters
        ----------
        config_dict : dict
            配置字典
        name : str
            配置标识名称

        Returns
        -------
        WorkflowConfig
        """
        config_dict.setdefault("name", name)
        try:
            config = WorkflowConfig(**config_dict)
        except Exception as e:
            logger.error(f"配置字典校验失败: {e}")
            raise ValueError(f"配置字典格式错误: {e}") from e

        logger.info(f"从字典加载配置成功: '{config.name}'")
        return config

    def save(
        self,
        config: WorkflowConfig,
        output_path: str | Path,
        overwrite: bool = False,
    ) -> Path:
        """
        将配置对象序列化保存为 YAML 文件。

        Parameters
        ----------
        config : WorkflowConfig
            配置对象
        output_path : str | Path
            输出路径
        overwrite : bool
            是否覆盖已有文件

        Returns
        -------
        Path
            实际保存路径
        """
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"配置文件已存在: {output_path}。"
                f"如需覆盖，请设置 overwrite=True"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Pydantic v2 序列化
        config_dict = config.model_dump(exclude_none=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, allow_unicode=True,
                      default_flow_style=False, indent=2)

        logger.info(f"配置已保存至: {output_path}")
        return output_path

    def list_presets(self) -> list[str]:
        """列出所有预置配置文件名"""
        if not self._config_dir.exists():
            return []
        return [p.stem for p in self._config_dir.glob("*.yaml")]




    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """深度合并两个字典，override 优先级更高"""
        result = base.copy()
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = ConfigLoader._deep_merge(result[key], val)
            else:
                result[key] = val
        return result

    def clear_cache(self) -> None:
        """清除配置缓存"""
        self._config_cache.clear()
        logger.debug("配置缓存已清除")