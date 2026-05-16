# src/utils/logging_config.py
"""
日志配置模块
支持控制台彩色输出（rich）与文件持久化日志
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional

# 尝试导入 rich（美化终端输出）
try:
    from rich.logging import RichHandler
    from rich.console import Console
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

# 全局日志字典，防止重复创建
_loggers: dict[str, logging.Logger] = {}

# 日志级别映射
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def setup_logger(
    name: str = "auto_eval",
    level: str = "info",
    log_dir: Optional[str | Path] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    创建并配置一个日志记录器。

    Parameters
    ----------
    name : str
        日志记录器名称，同名多次调用返回同一实例
    level : str
        日志级别：'debug', 'info', 'warning', 'error', 'critical'
    log_dir : str | Path | None
        日志文件存放目录，None 则使用 ./output/logs/
    log_to_file : bool
        是否写入文件
    log_to_console : bool
        是否输出到控制台
    max_bytes : int
        单个日志文件最大字节数（滚动写入）
    backup_count : int
        保留历史日志文件数量

    Returns
    -------
    logging.Logger
        配置好的日志记录器
    """
    # 若已存在，直接返回（避免重复添加 Handler）
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
    logger.handlers.clear()       # 清除可能的继承 Handler
    logger.propagate = False      # 不向根 Logger 传播

    # ---------- 格式化器 ----------
    detailed_fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "%(filename)s:%(lineno)d | %(message)s"
    )
    simple_fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    file_formatter = logging.Formatter(detailed_fmt, datefmt=date_fmt)
    simple_formatter = logging.Formatter(simple_fmt, datefmt=date_fmt)

    # ---------- 控制台 Handler ----------
    if log_to_console:
        if _RICH_AVAILABLE:
            console_handler = RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                show_time=True,
                show_path=False,
                markup=True,
            )
            console_handler.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
        else:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(simple_formatter)
            console_handler.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
        logger.addHandler(console_handler)

    # ---------- 文件 Handler ----------
    if log_to_file:
        if log_dir is None:
            log_dir = Path("output") / "logs"
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)   # 文件记录全量 DEBUG
        logger.addHandler(file_handler)

    _loggers[name] = logger

    logger.debug(
        f"Logger '{name}' initialized | level={level.upper()} | "
        f"file={log_to_file} | console={log_to_console}"
    )
    return logger


def get_logger(name: str = "auto_eval") -> logging.Logger:
    """
    获取已存在的 Logger，若不存在则使用默认配置创建。

    Parameters
    ----------
    name : str
        Logger 名称

    Returns
    -------
    logging.Logger
    """
    if name not in _loggers:
        return setup_logger(name=name)
    return _loggers[name]


class LoggingMixin:
    """
    为任意类提供 self.logger 属性的 Mixin 类。
    子类可直接使用 self.logger.info(...) 等方法。

    Example
    -------
    class MyClass(LoggingMixin):
        def __init__(self):
            super().__init__()
            self.logger.info("MyClass initialized")
    """

    @property
    def logger(self) -> logging.Logger:
        """返回以类名为标识的 Logger"""
        class_name = type(self).__name__
        logger_name = f"auto_eval.{class_name}"
        return get_logger(logger_name)