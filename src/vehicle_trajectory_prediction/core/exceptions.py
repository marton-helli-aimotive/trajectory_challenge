"""Custom exceptions for the vehicle trajectory prediction system."""

from typing import Any, Dict, Optional


class TrajectoryPredictionError(Exception):
    """Base exception for trajectory prediction errors."""

    def __init__(
        self, message: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataValidationError(TrajectoryPredictionError):
    """Raised when data validation fails."""

    def __init__(
        self, message: str, field: Optional[str] = None, value: Optional[Any] = None
    ) -> None:
        super().__init__(message, {"field": field, "value": value})
        self.field = field
        self.value = value


class ModelError(TrajectoryPredictionError):
    """Raised when model operations fail."""

    def __init__(
        self, message: str, model_name: Optional[str] = None, operation: Optional[str] = None
    ) -> None:
        super().__init__(message, {"model_name": model_name, "operation": operation})
        self.model_name = model_name
        self.operation = operation


class ConfigurationError(TrajectoryPredictionError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self, message: str, config_key: Optional[str] = None, config_value: Optional[Any] = None
    ) -> None:
        super().__init__(message, {"config_key": config_key, "config_value": config_value})
        self.config_key = config_key
        self.config_value = config_value


class DataSourceError(TrajectoryPredictionError):
    """Raised when data source operations fail."""

    def __init__(
        self, message: str, source: Optional[str] = None, operation: Optional[str] = None
    ) -> None:
        super().__init__(message, {"source": source, "operation": operation})
        self.source = source
        self.operation = operation


class FeatureEngineeringError(TrajectoryPredictionError):
    """Raised when feature engineering operations fail."""

    def __init__(
        self, message: str, feature_name: Optional[str] = None, operation: Optional[str] = None
    ) -> None:
        super().__init__(message, {"feature_name": feature_name, "operation": operation})
        self.feature_name = feature_name
        self.operation = operation


class EvaluationError(TrajectoryPredictionError):
    """Raised when evaluation operations fail."""

    def __init__(
        self, message: str, metric_name: Optional[str] = None, operation: Optional[str] = None
    ) -> None:
        super().__init__(message, {"metric_name": metric_name, "operation": operation})
        self.metric_name = metric_name
        self.operation = operation