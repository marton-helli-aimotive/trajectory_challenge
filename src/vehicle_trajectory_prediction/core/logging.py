"""Logging configuration for the vehicle trajectory prediction system."""

import logging
import sys
from typing import Any, Dict, Optional

# Try to import optional dependencies
try:
    import structlog
    from structlog.stdlib import LoggerFactory
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    json_format: bool = False,
    include_timestamp: bool = True,
    include_process_id: bool = True,
    include_thread_id: bool = True,
) -> None:
    """Set up structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for console output
        json_format: Whether to use JSON format for logs
        include_timestamp: Whether to include timestamps in logs
        include_process_id: Whether to include process ID in logs
        include_thread_id: Whether to include thread ID in logs
    """
    if STRUCTLOG_AVAILABLE:
        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, level.upper()),
        )
        
        # Configure structlog
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
        ]
        
        if include_timestamp:
            processors.append(structlog.stdlib.add_log_level_number)
            processors.append(structlog.processors.TimeStamper(fmt="iso"))
        
        if include_process_id:
            processors.append(structlog.stdlib.add_log_level_number)
        
        if include_thread_id:
            processors.append(structlog.stdlib.add_log_level_number)
        
        processors.extend([
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ])
        
        if json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            if format_string:
                processors.append(structlog.dev.ConsoleRenderer(fmt=format_string))
            else:
                processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    else:
        # Fallback to basic logging
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            format=format_str,
            level=getattr(logging, level.upper()),
            stream=sys.stdout
        )


def get_logger(name: str):
    """Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structured logger or standard logger
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


def log_function_call(
    func_name: str,
    args: Optional[Dict[str, Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
):
    """Create a logger for function call logging.
    
    Args:
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments
        
    Returns:
        Logger with function call context
    """
    logger = get_logger(__name__)
    if STRUCTLOG_AVAILABLE:
        context = {"function": func_name}
        
        if args:
            context["args"] = args
        if kwargs:
            context["kwargs"] = kwargs
        
        return logger.bind(**context)
    else:
        # Simple logging for basic setup
        msg = f"Function call: {func_name}"
        if args:
            msg += f" args={args}"
        if kwargs:
            msg += f" kwargs={kwargs}"
        logger.info(msg)
        return logger


def log_performance(
    operation: str,
    duration: float,
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        additional_info: Additional performance information
    """
    logger = get_logger(__name__)
    if STRUCTLOG_AVAILABLE:
        context = {
            "operation": operation,
            "duration_seconds": duration,
            "log_type": "performance",
        }
        
        if additional_info:
            context.update(additional_info)
        
        logger.info("Performance metric", **context)
    else:
        msg = f"Performance: {operation} took {duration:.3f}s"
        if additional_info:
            msg += f" {additional_info}"
        logger.info(msg)


def log_data_quality(
    dataset_name: str,
    total_records: int,
    valid_records: int,
    quality_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Log data quality metrics.
    
    Args:
        dataset_name: Name of the dataset
        total_records: Total number of records
        valid_records: Number of valid records
        quality_metrics: Additional quality metrics
    """
    logger = get_logger(__name__)
    quality_score = valid_records / total_records if total_records > 0 else 0.0
    
    if STRUCTLOG_AVAILABLE:
        context = {
            "dataset": dataset_name,
            "total_records": total_records,
            "valid_records": valid_records,
            "quality_score": quality_score,
            "log_type": "data_quality",
        }
        
        if quality_metrics:
            context.update(quality_metrics)
        
        logger.info("Data quality report", **context)
    else:
        msg = f"Data quality: {dataset_name} - {valid_records}/{total_records} valid ({quality_score:.2%})"
        if quality_metrics:
            msg += f" {quality_metrics}"
        logger.info(msg)