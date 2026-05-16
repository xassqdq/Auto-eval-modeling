# tests/unit/test_core/test_workflow.py

import pytest
from src.core.workflow import Workflow, WorkflowConfig
from src.core.node import (
    DataLoadNode, NormalizeNode, WeightNode,
    EvaluateNode, SensitivityNode
)


class TestWorkflowConstruction:

    def test_create_from_config(self, minimal_config):
        wf = Workflow.from_config(minimal_config)
        assert wf is not None
        assert len(wf.nodes) == 3  # normalize, weight, evaluate

    def test_dag_is_acyclic(self, full_pipeline_config):
        wf = Workflow.from_config(full_pipeline_config)
        assert wf.is_acyclic()

    def test_topological_order(self, minimal_config):
        wf = Workflow.from_config(minimal_config)
        order = wf.topological_sort()
        # normalize 应在 weight 之前, weight 在 evaluate 之前
        node_names = [n.name for n in order]
        assert node_names.index("normalize") < node_names.index("weight")
        assert node_names.index("weight") < node_names.index("evaluate")

    def test_empty_pipeline_raises(self):
        config = {"data_source": "x.csv", "pipeline": []}
        with pytest.raises(ValueError, match="empty"):
            Workflow.from_config(config)

    def test_invalid_step_raises(self):
        config = {
            "data_source": "x.csv",
            "pipeline": [{"step": "nonexistent_method", "method": "foo"}]
        }
        with pytest.raises(ValueError, match="unknown"):
            Workflow.from_config(config)


class TestWorkflowExecution:

    def test_minimal_pipeline_runs(self, minimal_config, sample_3x4, tmp_output):
        wf = Workflow.from_config(minimal_config)
        result = wf.execute(data=sample_3x4, output_dir=tmp_output)
        assert result.success
        assert "scores" in result.outputs
        assert "weights" in result.outputs

    def test_full_pipeline_runs(self, full_pipeline_config, sample_10x8, tmp_output):
        wf = Workflow.from_config(full_pipeline_config)
        result = wf.execute(data=sample_10x8, output_dir=tmp_output)
        assert result.success

    def test_intermediate_results_stored(self, minimal_config, sample_3x4, tmp_output):
        wf = Workflow.from_config(minimal_config)
        result = wf.execute(data=sample_3x4, output_dir=tmp_output)
        # 每个节点的输出都应存储
        assert len(result.intermediate) == len(wf.nodes)

    def test_execution_order_correct(self, minimal_config, sample_3x4, tmp_output):
        """执行日志应反映拓扑顺序"""
        wf = Workflow.from_config(minimal_config)
        result = wf.execute(data=sample_3x4, output_dir=tmp_output)
        log_steps = [entry["step"] for entry in result.execution_log]
        assert log_steps == ["normalize", "weight", "evaluate"]

    def test_error_in_node_captured(self, sample_3x4, tmp_output):
        """节点执行出错应被捕获并记录"""
        config = {
            "data_source": "test",
            "pipeline": [
                {"step": "normalize", "method": "minmax"},
                {"step": "weight", "method": "entropy"},
                # 故意传入不存在的评价方法
                {"step": "evaluate", "method": "nonexistent"},
            ]
        }
        wf = Workflow.from_config(config)
        result = wf.execute(data=sample_3x4, output_dir=tmp_output)
        assert not result.success
        assert result.error_node == "evaluate"


# tests/unit/test_core/test_node.py

import pytest
import pandas as pd
import numpy as np
from src.core.node import NormalizeNode, WeightNode, EvaluateNode


class TestNodeInterface:

    def test_normalize_node(self, sample_3x4):
        node = NormalizeNode(method="minmax")
        output = node.run(data=sample_3x4)
        assert isinstance(output["data"], pd.DataFrame)
        assert output["data"].shape == sample_3x4.shape

    def test_weight_node(self, sample_3x4):
        # 先归一化
        norm_node = NormalizeNode(method="minmax")
        norm_out = norm_node.run(data=sample_3x4)

        weight_node = WeightNode(method="entropy")
        output = weight_node.run(data=norm_out["data"])
        assert "weights" in output
        assert abs(output["weights"].sum() - 1.0) < 1e-10

    def test_evaluate_node(self, sample_3x4):
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        node = EvaluateNode(method="topsis",
                            directions=["positive"]*4)
        output = node.run(data=sample_3x4, weights=weights)
        assert "scores" in output
        assert "ranking" in output

    def test_node_metadata(self, sample_3x4):
        node = NormalizeNode(method="minmax")
        output = node.run(data=sample_3x4)
        assert "metadata" in output
        assert "method" in output["metadata"]
        assert output["metadata"]["method"] == "minmax"


# tests/unit/test_core/test_data_bus.py

import pytest
import pandas as pd
import numpy as np
from src.core.data_bus import DataBus


class TestDataBus:

    def test_put_and_get(self):
        bus = DataBus()
        df = pd.DataFrame({"A": [1, 2, 3]})
        bus.put("node_1", "data", df)
        result = bus.get("node_1", "data")
        pd.testing.assert_frame_equal(result, df)

    def test_get_nonexistent_raises(self):
        bus = DataBus()
        with pytest.raises(KeyError):
            bus.get("missing_node", "data")

    def test_overwrite_warning(self):
        bus = DataBus()
        bus.put("node_1", "data", pd.DataFrame({"A": [1]}))
        with pytest.warns(UserWarning, match="overwrite"):
            bus.put("node_1", "data", pd.DataFrame({"A": [2]}))

    def test_list_available_keys(self):
        bus = DataBus()
        bus.put("n1", "data", "x")
        bus.put("n1", "weights", "y")
        bus.put("n2", "data", "z")
        assert set(bus.list_keys("n1")) == {"data", "weights"}


# tests/unit/test_core/test_recommendation.py

import pytest
import pandas as pd
import numpy as np
from src.core.recommendation import AlgorithmRecommender


class TestRecommender:

    def test_high_correlation_suggests_pca(self):
        """高相关性数据应推荐PCA"""
        x = np.arange(50, dtype=float)
        df = pd.DataFrame({
            "A": x, "B": x * 2, "C": x + np.random.randn(50) * 0.01
        })
        rec = AlgorithmRecommender()
        suggestions = rec.recommend(df, context="ranking")
        step_methods = [(s["step"], s["method"]) for s in suggestions]
        assert ("reduce", "pca") in step_methods

    def test_small_sample_suggests_gra(self):
        """小样本应推荐灰色关联"""
        df = pd.DataFrame(np.random.rand(3, 6),
                          columns=[f"X{i}" for i in range(6)])
        rec = AlgorithmRecommender()
        suggestions = rec.recommend(df, context="ranking")
        eval_methods = [s["method"] for s in suggestions if s["step"] == "evaluate"]
        assert "gra" in eval_methods

    def test_returns_multiple_plans(self):
        """应返回多个备选方案"""
        df = pd.DataFrame(np.random.rand(10, 4))
        rec = AlgorithmRecommender()
        plans = rec.recommend_all(df, context="ranking")
        assert len(plans) >= 2

    def test_recommendation_has_reasons(self):
        """每个推荐应附带理由"""
        df = pd.DataFrame(np.random.rand(10, 4))
        rec = AlgorithmRecommender()
        plans = rec.recommend_all(df, context="ranking")
        for plan in plans:
            assert "reason" in plan
            assert len(plan["reason"]) > 0