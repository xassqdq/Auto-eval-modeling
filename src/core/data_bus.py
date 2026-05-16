"""
节点间数据传递总线 (DataBus)

设计原则：
    - 所有节点的中间数据通过 DataBus 共享
    - 采用键值存储 + 时间戳元数据
    - 支持快照（checkpoint）与回滚
    - 线程安全（使用 threading.Lock）

Author: AutoEval-Modeling
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class DataBusError(Exception):
    """DataBus 相关异常"""
    pass


class DataBus:
    """
    工作流节点间数据共享总线

    Usage
    -----
    >>> bus = DataBus()
    >>> bus.put("raw_data", df)
    >>> bus.put("weights", np.array([0.3, 0.4, 0.3]))
    >>> data = bus.get("raw_data")
    >>> bus.checkpoint("after_preprocessing")
    >>> bus.rollback("after_preprocessing")

    Thread Safety
    -------------
    所有读写操作均受 threading.RLock 保护。
    """

    # 保留键名（不允许外部覆盖）
    _RESERVED_KEYS: Set[str] = {
        "__metadata__",
        "__checkpoints__",
        "__access_log__",
    }

    def __init__(self, strict_mode: bool = False) -> None:
        """
        Parameters
        ----------
        strict_mode : bool
            True → 读取不存在的键时抛出异常（默认宽松：返回 None）
        """
        self._store:       Dict[str, Any]          = {}
        self._metadata:    Dict[str, Dict]          = {}
        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        self._access_log:  List[Dict]               = []
        self._lock         = threading.RLock()
        self.strict_mode   = strict_mode

    # ------------------------------------------------------------------
    # 核心读写
    # ------------------------------------------------------------------

    def put(
        self,
        key: str,
        value: Any,
        source_node: str = "unknown",
        overwrite: bool = True,
    ) -> None:
        """
        向总线写入数据

        Parameters
        ----------
        key : str
            数据键名（推荐使用 snake_case）
        value : Any
            数据对象（DataFrame、ndarray、dict 等均可）
        source_node : str
            写入方节点名称（用于追踪）
        overwrite : bool
            False 时若键已存在则抛出异常
        """
        if key in self._RESERVED_KEYS:
            raise DataBusError(f"键名 '{key}' 为保留名称，不可使用")

        with self._lock:
            if not overwrite and key in self._store:
                raise DataBusError(
                    f"键 '{key}' 已存在，若要覆盖请设置 overwrite=True"
                )
            self._store[key] = value
            self._metadata[key] = {
                "source_node": source_node,
                "timestamp":   time.time(),
                "type":        type(value).__name__,
            }
            self._log_access("write", key, source_node)
            logger.debug("DataBus PUT: [%s] ← %s (from %s)", key,
                         type(value).__name__, source_node)

    def get(
        self,
        key: str,
        default: Any = None,
        consumer_node: str = "unknown",
    ) -> Any:
        """
        从总线读取数据

        Parameters
        ----------
        key : str
        default : Any
            键不存在时的返回值（strict_mode=True 时改为抛异常）
        consumer_node : str
            读取方节点名称
        """
        with self._lock:
            if key not in self._store:
                if self.strict_mode:
                    raise DataBusError(
                        f"键 '{key}' 不存在，请检查上游节点是否已正确执行"
                    )
                logger.debug("DataBus GET: [%s] → 不存在，返回默认值", key)
                return default
            self._log_access("read", key, consumer_node)
            return self._store[key]

    def get_required(self, key: str, consumer_node: str = "unknown") -> Any:
        """必须存在的键，不存在时抛出异常"""
        with self._lock:
            if key not in self._store:
                raise DataBusError(
                    f"必要数据 '{key}' 缺失，节点 '{consumer_node}' 无法执行"
                )
            self._log_access("read_required", key, consumer_node)
            return self._store[key]

    def has(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            return key in self._store

    def delete(self, key: str) -> bool:
        """删除指定键，返回是否成功"""
        with self._lock:
            if key in self._store:
                del self._store[key]
                del self._metadata[key]
                logger.debug("DataBus DELETE: [%s]", key)
                return True
            return False

    def keys(self) -> List[str]:
        """返回所有键名列表"""
        with self._lock:
            return list(self._store.keys())

    # ------------------------------------------------------------------
    # 批量操作
    # ------------------------------------------------------------------

    def put_many(
        self,
        data_dict: Dict[str, Any],
        source_node: str = "unknown",
    ) -> None:
        """批量写入"""
        for key, value in data_dict.items():
            self.put(key, value, source_node=source_node)

    def get_many(
        self,
        keys: List[str],
        consumer_node: str = "unknown",
    ) -> Dict[str, Any]:
        """批量读取"""
        return {
            key: self.get(key, consumer_node=consumer_node)
            for key in keys
        }

    # ------------------------------------------------------------------
    # 快照与回滚
    # ------------------------------------------------------------------

    def checkpoint(self, name: str) -> None:
        """
        创建当前状态快照

        Parameters
        ----------
        name : str
            快照名称，建议使用节点名称（如 "after_preprocess"）
        """
        with self._lock:
            self._checkpoints[name] = {
                "store":    copy.deepcopy(self._store),
                "metadata": copy.deepcopy(self._metadata),
                "time":     time.time(),
            }
            logger.info("DataBus 快照已创建: '%s'（共 %d 键）",
                        name, len(self._store))

    def rollback(self, checkpoint_name: str) -> None:
        """
        回滚到指定快照状态

        Parameters
        ----------
        checkpoint_name : str
            快照名称
        """
        with self._lock:
            if checkpoint_name not in self._checkpoints:
                raise DataBusError(
                    f"快照 '{checkpoint_name}' 不存在，"
                    f"可用快照: {list(self._checkpoints.keys())}"
                )
            snap = self._checkpoints[checkpoint_name]
            self._store    = copy.deepcopy(snap["store"])
            self._metadata = copy.deepcopy(snap["metadata"])
            logger.info(
                "DataBus 已回滚至快照: '%s'", checkpoint_name
            )

    def list_checkpoints(self) -> List[str]:
        """返回所有快照名称"""
        with self._lock:
            return list(self._checkpoints.keys())

    # ------------------------------------------------------------------
    # 元数据与日志
    # ------------------------------------------------------------------

    def get_metadata(self, key: str) -> Optional[Dict]:
        """获取指定键的元数据"""
        with self._lock:
            return self._metadata.get(key)

    def get_access_log(self) -> List[Dict]:
        """获取所有访问日志"""
        with self._lock:
            return list(self._access_log)

    def describe(self) -> str:
        """打印当前总线状态摘要"""
        with self._lock:
            lines = [
                f"DataBus 状态摘要 ({len(self._store)} 个键):",
                "-" * 50,
            ]
            for key, meta in self._metadata.items():
                lines.append(
                    f"  [{key}]  type={meta['type']}"
                    f"  from={meta['source_node']}"
                    f"  time={time.strftime('%H:%M:%S', time.localtime(meta['timestamp']))}"
                )
            return "\n".join(lines)

    def clear(self, keep_checkpoints: bool = False) -> None:
        """清空总线（可选保留快照）"""
        with self._lock:
            self._store.clear()
            self._metadata.clear()
            if not keep_checkpoints:
                self._checkpoints.clear()
            self._access_log.clear()
            logger.info("DataBus 已清空")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _log_access(self, op: str, key: str, node: str) -> None:
        """记录访问日志（最多保留 1000 条）"""
        self._access_log.append({
            "op":   op,
            "key":  key,
            "node": node,
            "time": time.time(),
        })
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-500:]

    def __repr__(self) -> str:
        return (
            f"DataBus(keys={list(self._store.keys())}, "
            f"checkpoints={list(self._checkpoints.keys())})"
        )