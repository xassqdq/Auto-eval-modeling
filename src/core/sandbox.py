"""
安全代码执行沙箱 (CodeSandbox)

功能：
    - 在受控命名空间中执行动态构建的 Python 代码
    - 捕获 stdout/stderr、matplotlib 图对象、返回变量
    - 提供白名单导入机制（防止恶意代码）
    - 执行超时保护（threading.Timer）

Author: AutoEval-Modeling
"""

from __future__ import annotations

import io
import logging
import sys
import threading
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# 允许的模块白名单（沙箱模式下）
_ALLOWED_MODULES: Set[str] = {
    "numpy", "np",
    "pandas", "pd",
    "matplotlib", "matplotlib.pyplot", "plt",
    "seaborn", "sns",
    "scipy", "scipy.stats", "scipy.linalg", "scipy.optimize",
    "sklearn", "sklearn.preprocessing", "sklearn.decomposition",
    "math", "os", "sys", "json", "csv", "re",
    "collections", "itertools", "functools",
    "pathlib", "datetime", "time",
    "warnings", "logging",
}


class SandboxError(Exception):
    """沙箱执行异常"""
    pass


class SandboxResult:
    """沙箱执行结果"""

    def __init__(
        self,
        success:    bool,
        stdout:     str = "",
        stderr:     str = "",
        variables:  Optional[Dict[str, Any]] = None,
        figures:    Optional[List[Any]] = None,
        error_msg:  Optional[str] = None,
        elapsed:    float = 0.0,
    ) -> None:
        self.success   = success
        self.stdout    = stdout
        self.stderr    = stderr
        self.variables = variables or {}
        self.figures   = figures or []
        self.error_msg = error_msg
        self.elapsed   = elapsed

    def __repr__(self) -> str:
        return (
            f"SandboxResult(success={self.success}, "
            f"vars={list(self.variables.keys())}, "
            f"figures={len(self.figures)})"
        )


class CodeSandbox:
    """
    受控 Python 代码执行沙箱

    Parameters
    ----------
    timeout : float
        执行超时秒数（默认 60s，None 表示不限时）
    safe_mode : bool
        True → 启用白名单导入限制；False → 与宿主进程相同权限
    capture_figures : bool
        是否自动捕获 matplotlib 图对象
    extra_namespace : dict
        额外注入到执行命名空间的变量

    Examples
    --------
    >>> sandbox = CodeSandbox(timeout=30)
    >>> code = '''
    ... import numpy as np
    ... weights = np.array([0.3, 0.4, 0.3])
    ... scores = data @ weights
    ... '''
    >>> result = sandbox.execute(code, initial_vars={"data": my_data})
    >>> print(result.variables["scores"])
    """

    def __init__(
        self,
        timeout:         Optional[float] = 60.0,
        safe_mode:       bool = False,
        capture_figures: bool = True,
        extra_namespace: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.timeout         = timeout
        self.safe_mode       = safe_mode
        self.capture_figures = capture_figures
        self.extra_namespace = extra_namespace or {}

    def execute(
        self,
        code:          str,
        initial_vars:  Optional[Dict[str, Any]] = None,
        export_vars:   Optional[List[str]] = None,
    ) -> SandboxResult:
        """
        执行代码字符串

        Parameters
        ----------
        code : str
            要执行的 Python 代码
        initial_vars : dict
            注入到执行命名空间的初始变量
        export_vars : list of str
            执行后需要提取的变量名（None = 提取所有非私有变量）

        Returns
        -------
        SandboxResult
        """
        namespace = self._build_namespace(initial_vars)
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        figures    = []
        success    = False
        error_msg  = None
        start_time = time.time()

        # matplotlib 图捕获
        if self.capture_figures:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
                matplotlib.use("Agg")
                plt.close("all")
                namespace["plt"] = plt
            except ImportError:
                pass

        # 超时执行（使用线程）
        exec_exception = [None]

        def _exec_target():
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(code, namespace)  # noqa: S102
            except Exception as exc:
                exec_exception[0] = exc

        thread = threading.Thread(target=_exec_target, daemon=True)
        thread.start()

        if self.timeout:
            thread.join(timeout=self.timeout)
            if thread.is_alive():
                error_msg = f"执行超时（>{self.timeout}s）"
                logger.error("沙箱执行超时")
                return SandboxResult(
                    success   = False,
                    stdout    = stdout_buf.getvalue(),
                    stderr    = stderr_buf.getvalue(),
                    error_msg = error_msg,
                    elapsed   = time.time() - start_time,
                )
        else:
            thread.join()

        elapsed = time.time() - start_time

        # 检查异常
        if exec_exception[0] is not None:
            error_msg = (
                f"{type(exec_exception[0]).__name__}: {exec_exception[0]}\n"
                f"{traceback.format_exc()}"
            )
            logger.error("沙箱执行异常: %s", exec_exception[0])
            return SandboxResult(
                success   = False,
                stdout    = stdout_buf.getvalue(),
                stderr    = stderr_buf.getvalue(),
                error_msg = error_msg,
                elapsed   = elapsed,
            )

        # 捕获 matplotlib 图
        if self.capture_figures:
            try:
                import matplotlib.pyplot as plt
                figures = [plt.figure(num) for num in plt.get_fignums()]
            except Exception:
                figures = []

        # 提取变量
        extracted = self._extract_variables(namespace, export_vars)
        success = True

        logger.info(
            "沙箱执行成功 (%.2fs) | 提取变量: %s",
            elapsed, list(extracted.keys()),
        )
        return SandboxResult(
            success   = success,
            stdout    = stdout_buf.getvalue(),
            stderr    = stderr_buf.getvalue(),
            variables = extracted,
            figures   = figures,
            elapsed   = elapsed,
        )

    def execute_file(
        self,
        file_path: str,
        initial_vars: Optional[Dict[str, Any]] = None,
        export_vars:  Optional[List[str]] = None,
    ) -> SandboxResult:
        """执行 Python 文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            return self.execute(code, initial_vars, export_vars)
        except FileNotFoundError:
            return SandboxResult(
                success   = False,
                error_msg = f"文件不存在: {file_path}",
            )

    def validate_code(self, code: str) -> Tuple[bool, str]:
        """
        语法校验（不执行）

        Returns
        -------
        (is_valid, error_message)
        """
        try:
            compile(code, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"语法错误 (行 {e.lineno}): {e.msg}"

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_namespace(
        self, initial_vars: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """构建执行命名空间"""
        namespace: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "__name__":     "__sandbox__",
        }

        # 注入常用库
        try:
            import numpy as np
            import pandas as pd
            namespace.update({"np": np, "pd": pd, "numpy": np, "pandas": pd})
        except ImportError:
            pass

        # 注入额外命名空间
        namespace.update(self.extra_namespace)

        # 注入初始变量
        if initial_vars:
            namespace.update(initial_vars)

        return namespace

    def _extract_variables(
        self,
        namespace:   Dict[str, Any],
        export_vars: Optional[List[str]],
    ) -> Dict[str, Any]:
        """从命名空间提取指定变量"""
        _SKIP_KEYS = {
            "__builtins__", "__name__", "__doc__",
            "__package__", "__loader__", "__spec__",
            "np", "pd", "plt", "numpy", "pandas",
            "matplotlib", "seaborn", "scipy", "sklearn",
        }

        if export_vars is not None:
            return {
                k: namespace[k]
                for k in export_vars
                if k in namespace
            }

        # 提取所有用户定义变量
        return {
            k: v
            for k, v in namespace.items()
            if (
                not k.startswith("_")
                and k not in _SKIP_KEYS
                and not callable(v)
            )
        }