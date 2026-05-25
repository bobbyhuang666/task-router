"""
日志配置 — 统一的 Python logging 设置
"""

import logging
import os
import sys


def setup_logging(level: str = None) -> logging.Logger:
    """配置全局日志"""
    if level is None:
        level = os.environ.get("TASK_ROUTER_LOG_LEVEL", "INFO").upper()

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=log_format,
        datefmt=date_format,
        stream=sys.stderr,
    )

    # 降低第三方库日志级别
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    return logging.getLogger("task_router")


# 模块级 logger 供各模块使用
log = logging.getLogger("task_router")
