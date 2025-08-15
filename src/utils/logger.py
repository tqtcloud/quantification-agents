import logging
import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from structlog.processors import CallsiteParameter, CallsiteParameterAdder

from src.config import settings


def setup_logging() -> None:
    """Configure structured logging for the application."""
    
    # Create log directory if it doesn't exist
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            CallsiteParameterAdder(
                parameters=[
                    CallsiteParameter.FILENAME,
                    CallsiteParameter.FUNC_NAME,
                    CallsiteParameter.LINENO,
                ]
            ),
            structlog.dev.ConsoleRenderer() if settings.log_level == "DEBUG" 
            else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )
    
    # Set up file handler
    file_handler = logging.FileHandler(
        settings.log_dir / "trading.log",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    
    # Add handler to root logger
    logging.getLogger().addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capability to any class."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
    
    def log_info(self, msg: str, **kwargs: Any) -> None:
        self.logger.info(msg, **kwargs)
    
    def log_debug(self, msg: str, **kwargs: Any) -> None:
        self.logger.debug(msg, **kwargs)
    
    def log_warning(self, msg: str, **kwargs: Any) -> None:
        self.logger.warning(msg, **kwargs)
    
    def log_error(self, msg: str, **kwargs: Any) -> None:
        self.logger.error(msg, **kwargs)
    
    def log_exception(self, msg: str, **kwargs: Any) -> None:
        self.logger.exception(msg, **kwargs)