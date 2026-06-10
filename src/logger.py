"""日志配置模块 — 基于 loguru，同时输出控制台和滚动文件。"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(level: str = "INFO", log_file: str = "logs/news-bot.log",
                  rotation: str = "7 days", retention: str = "14 days"):
    """初始化日志系统。

    Args:
        level: 日志级别，默认 INFO
        log_file: 日志文件路径
        rotation: 日志文件轮转周期
        retention: 日志文件保留周期
    """
    # 移除默认 handler
    logger.remove()

    # 控制台输出 — 彩色、简洁格式
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # 确保日志目录存在
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # 文件输出 — 带完整上下文
    logger.add(
        log_file,
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    logger.info(f"日志系统已初始化 (level={level}, file={log_file})")
    return logger
