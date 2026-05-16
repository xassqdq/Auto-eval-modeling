"""
DAG 工作流引擎 (Workflow)

核心功能：
    - 基于 networkx 构建 DAG 拓扑图
    - 拓扑排序确定执行顺序
    - 支持顺序/并行执行
    - 节点失败时可配置为中止/跳过/重试
    - 检查点快照与断点续跑

Author: AutoEval-Modeling
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import networkx as nx

from .data_bus import DataBus
from .node import BaseNode, NodeResult, NodeStatus

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """工作流整体状态"""
    IDLE      = "idle"
    RUNNING   = "running"
    SUCCESS   = "success"
    PARTIAL   = "partial"   # 部分节点失败，其余完成
    FAILED    = "failed"    # 关键节点失败，工作流终止


class WorkflowError(Exception):
    """工作流相关异常"""
    pass


class Workflow:
    """
    DAG 工作流引擎

    Parameters
    ----------
    name : str
        工作流名称
    bus  : DataBus, optional
        共享数据总线（不提供则自动创建）
    fail_fast : bool
        True → 任何节点失败立即终止；False → 尽可能完成（默认 True）
    max_retries : int
        单节点最大重试次数（默认 0）
    parallel : bool
        是否对无依赖关系的节点并行执行（默认 False）
    max_workers : int
        并行线程数（默认 4）

    Examples
    --------
    >>> wf = Workflow("city_eval")
    >>> wf.add_node(DataLoadNode("load", config={...}))
    >>> wf.add_node(PreprocessNode("preprocess", dependencies=["load"], config={...}))
    >>> wf.add_node(WeightNode("weight", dependencies=["preprocess"], config={...}))
    >>> wf.add_node(EvaluationNode("eval", dependencies=["weight"], config={...}))
    >>> wf.add_node(ConsolidationNode("consolidate", dependencies=["eval"]))
    >>> results = wf.run()
    """

    def __init__(
        self,
        name:        str = "AutoEvalWorkflow",
        bus:         Optional[DataBus] = None,
        fail_fast:   bool = True,
        max_retries: int  = 0,
        parallel:    bool = False,
        max_workers: int  = 4,
    ) -> None:
        self.name        = name
        self.bus         = bus or DataBus()
        self.fail_fast   = fail_fast
        self.max_retries = max_retries
        self.parallel    = parallel
        self.max_workers = max_workers

        self._graph:     nx.DiGraph          = nx.DiGraph()
        self._nodes:     Dict[str, BaseNode] = {}
        self._status:    WorkflowStatus      = WorkflowStatus.IDLE
        self._results:   Dict[str, NodeResult] = {}
        self._exec_order: List[str]          = []
        self._start_time: Optional[float]    = None

    # ------------------------------------------------------------------
    # 图构建接口
    # ------------------------------------------------------------------

    def add_node(self, node: BaseNode) -> "Workflow":
        """
        向工作流添加节点

        Parameters
        ----------
        node : BaseNode
            节点实例（包含 node_id 和 dependencies）
        """
        if node.node_id in self._nodes:
            raise WorkflowError(
                f"节点 ID '{node.node_id}' 已存在，请确保 node_id 唯一"
            )
        self._nodes[node.node_id] = node
        self._graph.add_node(node.node_id)

        for dep in node.dependencies:
            if dep not in self._nodes:
                # 延迟校验：先添加边，待 validate() 检查
                logger.debug("依赖节点 '%s' 尚未注册", dep)
            self._graph.add_edge(dep, node.node_id)

        logger.debug("节点 [%s] 已添加", node.node_id)
        return self  # 支持链式调用

    def add_nodes(self, nodes: List[BaseNode]) -> "Workflow":
        """批量添加节点"""
        for node in nodes:
            self.add_node(node)
        return self

    def remove_node(self, node_id: str) -> "Workflow":
        """移除节点及其相关边"""
        if node_id not in self._nodes:
            raise WorkflowError(f"节点 '{node_id}' 不存在")
        del self._nodes[node_id]
        self._graph.remove_node(node_id)
        return self

    def add_conditional_skip(
        self,
        node_id:   str,
        condition: Any,  # callable(bus) -> bool
        reason:    str = "",
    ) -> None:
        """
        为节点添加条件跳过逻辑

        Parameters
        ----------
        node_id   : 目标节点 ID
        condition : 可调用对象，参数为 DataBus，返回 True 时跳过该节点
        reason    : 跳过原因描述
        """
        node = self._get_node(node_id)
        node._skip_condition = condition
        node._skip_reason    = reason

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """
        验证工作流合法性

        Returns
        -------
        list of str
            错误信息列表（空列表表示验证通过）
        """
        errors = []

        # 检查所有依赖节点是否存在
        for node_id, node in self._nodes.items():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    errors.append(
                        f"节点 '{node_id}' 依赖 '{dep}'，但该节点不存在"
                    )

        # 检查是否有环
        if not nx.is_directed_acyclic_graph(self._graph):
            cycles = list(nx.simple_cycles(self._graph))
            errors.append(f"工作流存在循环依赖: {cycles}")

        # 检查是否有孤立节点（无入边且无出边）
        for node_id in self._nodes:
            if (self._graph.in_degree(node_id) == 0
                    and self._graph.out_degree(node_id) == 0
                    and len(self._nodes) > 1):
                errors.append(f"节点 '{node_id}' 是孤立节点（无依赖且无后继）")

        return errors

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def run(
        self,
        start_from: Optional[str] = None,
        stop_at:    Optional[str] = None,
    ) -> Dict[str, NodeResult]:
        """
        执行整个工作流

        Parameters
        ----------
        start_from : str, optional
            从指定节点开始执行（配合检查点使用）
        stop_at : str, optional
            执行到指定节点后停止

        Returns
        -------
        dict : node_id → NodeResult
        """
        # ---- 预处理 ----
        errors = self.validate()
        if errors:
            raise WorkflowError(
                "工作流验证失败:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        self._exec_order = list(nx.topological_sort(self._graph))
        self._status     = WorkflowStatus.RUNNING
        self._start_time = time.time()
        self._results    = {}

        logger.info(
            "工作流 [%s] 启动 | 节点数=%d | 执行顺序=%s",
            self.name, len(self._nodes), self._exec_order,
        )

        # ---- 跳过 start_from 之前的节点 ----
        skip_until = None
        if start_from:
            if start_from not in self._nodes:
                raise WorkflowError(f"start_from 节点 '{start_from}' 不存在")
            skip_until = start_from

        # ---- 执行 ----
        if self.parallel:
            self._run_parallel(skip_until, stop_at)
        else:
            self._run_sequential(skip_until, stop_at)

        # ---- 汇总状态 ----
        failed_nodes  = [nid for nid, r in self._results.items() if r.status == NodeStatus.FAILED]
        success_nodes = [nid for nid, r in self._results.items() if r.status == NodeStatus.SUCCESS]
        elapsed       = time.time() - self._start_time

        if not failed_nodes:
            self._status = WorkflowStatus.SUCCESS
        elif failed_nodes and success_nodes:
            self._status = WorkflowStatus.PARTIAL
        else:
            self._status = WorkflowStatus.FAILED

        logger.info(
            "工作流 [%s] 完成 | 状态=%s | 耗时=%.2fs | "
            "成功=%d 失败=%d",
            self.name, self._status.value, elapsed,
            len(success_nodes), len(failed_nodes),
        )
        return self._results

    # ------------------------------------------------------------------
    # 内部执行逻辑
    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        skip_until: Optional[str],
        stop_at:    Optional[str],
    ) -> None:
        """顺序执行"""
        in_skip_zone = bool(skip_until)

        for node_id in self._exec_order:
            # 处理 start_from 跳过
            if in_skip_zone:
                if node_id == skip_until:
                    in_skip_zone = False
                else:
                    node = self._nodes[node_id]
                    result = node.skip("start_from 之前的节点")
                    self._results[node_id] = result
                    continue

            node   = self._nodes[node_id]
            result = self._execute_with_retry(node)
            self._results[node_id] = result

            # 失败处理
            if result.status == NodeStatus.FAILED:
                if self.fail_fast:
                    logger.error(
                        "关键节点 [%s] 失败，工作流中止", node_id
                    )
                    # 跳过后续节点
                    for remaining_id in self._exec_order[
                        self._exec_order.index(node_id) + 1:
                    ]:
                        r = self._nodes[remaining_id].skip("上游节点失败")
                        self._results[remaining_id] = r
                    return

            # stop_at 终止
            if stop_at and node_id == stop_at:
                logger.info("已到达 stop_at 节点 [%s]，提前终止", stop_at)
                return

    def _run_parallel(
        self,
        skip_until: Optional[str],
        stop_at:    Optional[str],
    ) -> None:
        """
        基于层次的并行执行：
        同一拓扑层（无互相依赖）的节点并发运行
        """
        completed:  Set[str] = set()
        failed:     Set[str] = set()
        in_skip_zone = bool(skip_until)

        # 计算拓扑层
        layers = list(nx.topological_generations(self._graph))

        for layer in layers:
            # 过滤本层需要执行的节点
            to_run = []
            for node_id in layer:
                if in_skip_zone:
                    if node_id == skip_until:
                        in_skip_zone = False
                    else:
                        result = self._nodes[node_id].skip("start_from 之前")
                        self._results[node_id] = result
                        completed.add(node_id)
                        continue

                # 检查依赖是否有失败
                node = self._nodes[node_id]
                dep_failed = any(d in failed for d in node.dependencies)
                if dep_failed:
                    result = node.skip("上游节点失败")
                    self._results[node_id] = result
                    if self.fail_fast:
                        failed.add(node_id)
                    continue

                to_run.append(node_id)

            if not to_run:
                continue

            # 并行执行本层
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                future_map = {
                    executor.submit(
                        self._execute_with_retry, self._nodes[nid]
                    ): nid
                    for nid in to_run
                }
                for future in concurrent.futures.as_completed(future_map):
                    node_id = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.error("并行节点 [%s] 异常: %s", node_id, exc)
                        result = NodeResult(
                            node_id   = node_id,
                            status    = NodeStatus.FAILED,
                            error_msg = str(exc),
                        )
                    self._results[node_id] = result
                    if result.status == NodeStatus.FAILED:
                        failed.add(node_id)
                        if self.fail_fast:
                            logger.error(
                                "并行节点 [%s] 失败，将跳过后续依赖", node_id
                            )
                    else:
                        completed.add(node_id)

            # stop_at 检查
            if stop_at and stop_at in {nid for nid in to_run}:
                return

    def _execute_with_retry(self, node: BaseNode) -> NodeResult:
        """带重试逻辑的节点执行"""
        # 条件跳过检查
        if hasattr(node, "_skip_condition"):
            try:
                should_skip = node._skip_condition(self.bus)
                if should_skip:
                    return node.skip(
                        getattr(node, "_skip_reason", "条件跳过")
                    )
            except Exception as e:
                logger.warning("条件跳过检查异常: %s", e)

        # 创建检查点（可选）
        checkpoint_name = f"before_{node.node_id}"
        self.bus.checkpoint(checkpoint_name)

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                logger.info(
                    "重试节点 [%s] 第 %d 次...", node.node_id, attempt
                )
                node.reset()
                # 回滚到节点执行前的状态
                try:
                    self.bus.rollback(checkpoint_name)
                except Exception:
                    pass

            result = node.execute(self.bus)

            if result.success:
                return result

            if attempt < self.max_retries:
                logger.warning(
                    "节点 [%s] 第 %d 次失败，准备重试...",
                    node.node_id, attempt + 1,
                )
                time.sleep(0.5 * (attempt + 1))  # 指数退避

        return result

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _get_node(self, node_id: str) -> BaseNode:
        if node_id not in self._nodes:
            raise WorkflowError(f"节点 '{node_id}' 不存在")
        return self._nodes[node_id]

    def get_result(self, node_id: str) -> Optional[NodeResult]:
        """获取指定节点的执行结果"""
        return self._results.get(node_id)

    def get_final_data(self) -> Optional[Dict]:
        """获取 ConsolidationNode 汇总的最终数据"""
        return self.bus.get("final_report_data")

    def print_summary(self) -> None:
        """打印工作流执行摘要"""
        total_time = (
            time.time() - self._start_time
            if self._start_time else 0.0
        )
        print(f"\n{'='*60}")
        print(f"  工作流: {self.name}")
        print(f"  状态  : {self._status.value}")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"{'='*60}")
        for node_id in self._exec_order:
            result = self._results.get(node_id)
            if result is None:
                status_str = "⊙ 未执行"
            elif result.status == NodeStatus.SUCCESS:
                status_str = f"✔ 成功 ({result.elapsed_sec:.2f}s)"
            elif result.status == NodeStatus.SKIPPED:
                status_str = "⊙ 跳过"
            elif result.status == NodeStatus.FAILED:
                status_str = f"✘ 失败: {result.error_msg[:60] if result.error_msg else ''}"
            else:
                status_str = result.status.value
            print(f"  [{node_id:20s}] {status_str}")
        print(f"{'='*60}\n")

    def visualize(self, save_path: Optional[str] = None) -> None:
        """可视化 DAG 拓扑图（需要 matplotlib）"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            matplotlib.use("Agg")

            pos = nx.spring_layout(self._graph, seed=42)
            color_map = {
                NodeStatus.SUCCESS: "#4CAF50",
                NodeStatus.FAILED:  "#F44336",
                NodeStatus.SKIPPED: "#9E9E9E",
                NodeStatus.RUNNING: "#2196F3",
                NodeStatus.PENDING: "#FFC107",
            }
            node_colors = []
            for nid in self._graph.nodes():
                result = self._results.get(nid)
                if result:
                    color = color_map.get(result.status, "#FFC107")
                else:
                    color = "#FFC107"
                node_colors.append(color)

            fig, ax = plt.subplots(figsize=(12, 6))
            nx.draw(
                self._graph, pos, ax=ax,
                with_labels=True, node_color=node_colors,
                node_size=2500, font_size=9,
                arrows=True, arrowsize=20,
                edge_color="gray", linewidths=1.5,
            )
            # 图例
            legend_handles = [
                mpatches.Patch(color=v, label=k.value)
                for k, v in color_map.items()
            ]
            ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
            ax.set_title(f"工作流 DAG: {self.name}", fontsize=13)
            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info("DAG 图已保存: %s", save_path)
            else:
                plt.show()
        except Exception as e:
            logger.warning("DAG 可视化失败: %s", e)

    @classmethod
    def from_config(cls, config: Dict) -> "Workflow":
        """
        从配置字典构建工作流（供 YAML 加载器调用）

        Parameters
        ----------
        config : dict
            包含 workflow_name, nodes, fail_fast 等字段的配置字典
        """
        from src.utils.config_loader import WorkflowConfig
        wf_config = WorkflowConfig(**config)

        wf = cls(
            name        = wf_config.workflow_name,
            fail_fast   = wf_config.fail_fast,
            max_retries = wf_config.max_retries,
            parallel    = wf_config.parallel,
        )

        # 节点类型映射
        from .node import (
            DataLoadNode, PreprocessNode, WeightNode,
            EvaluationNode, SensitivityNode, ConsolidationNode,
        )
        node_type_map: Dict[str, Type[BaseNode]] = {
            "DataLoadNode":     DataLoadNode,
            "PreprocessNode":   PreprocessNode,
            "WeightNode":       WeightNode,
            "EvaluationNode":   EvaluationNode,
            "SensitivityNode":  SensitivityNode,
            "ConsolidationNode": ConsolidationNode,
        }

        for node_cfg in wf_config.nodes:
            node_cls = node_type_map.get(node_cfg.type)
            if node_cls is None:
                raise WorkflowError(
                    f"未知节点类型: {node_cfg.type}. "
                    f"可用: {list(node_type_map.keys())}"
                )
            node = node_cls(
                node_id      = node_cfg.node_id,
                config       = node_cfg.config or {},
                dependencies = node_cfg.dependencies or [],
            )
            wf.add_node(node)

        return wf