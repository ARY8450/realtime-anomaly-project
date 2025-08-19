import sys
import os

def setup_logging():
    # import loguru lazily to avoid circular import / stdlib shadowing issues
    from loguru import logger

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Remove the default logger
    logger.remove()

    # Add a logger for console output (level INFO)
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")

    # Add a file logger with rotation
    logger.add("logs/runtime.log", rotation="10 MB", retention="7 days", level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    return logger
