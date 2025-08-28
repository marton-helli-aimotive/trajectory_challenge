"""Model monitoring and drift detection for vehicle trajectory prediction."""

import os
import json
import time
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from evidently import DataDefinition, Report

# from ..core.config import Config, get_config  # Removed for now
from ..core.logging import get_logger, setup_logging
from ..core.models import TrajectoryPoint, Trajectory
# from ..evaluation.metrics import EvaluationMetrics  # Removed for now


class ModelMonitor:
    """Monitor model performance and health in production."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize model monitor.
        
        Args:
            config: Configuration object
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.monitoring_data: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            "accuracy_drop": 0.1,
            "latency_increase": 0.5,
            "error_rate": 0.05,
            "data_drift": 0.3
        }
        
        # Load monitoring configuration
        self._load_monitoring_config()
        
        self.logger.info("Model monitor initialized")
    
    def _load_monitoring_config(self) -> None:
        """Load monitoring configuration."""
        try:
            monitoring_config = getattr(self.config, 'monitoring', {})
            if monitoring_config:
                self.alert_thresholds.update(monitoring_config.get('alert_thresholds', {}))
        except Exception as e:
            self.logger.warning("Failed to load monitoring config", error=str(e))
    
    def log_prediction(self, 
                      model_name: str,
                      vehicle_id: str,
                      input_data: Dict[str, Any],
                      prediction: Dict[str, Any],
                      actual: Optional[Dict[str, Any]] = None,
                      inference_time: float = 0.0,
                      error: Optional[str] = None) -> None:
        """Log a prediction for monitoring.
        
        Args:
            model_name: Name of the model used
            vehicle_id: Vehicle identifier
            input_data: Input data used for prediction
            prediction: Model prediction
            actual: Actual trajectory (if available)
            inference_time: Time taken for inference
            error: Error message if prediction failed
        """
        try:
            monitoring_record = {
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name,
                "vehicle_id": vehicle_id,
                "input_data": input_data,
                "prediction": prediction,
                "actual": actual,
                "inference_time": inference_time,
                "error": error,
                "success": error is None
            }
            
            self.monitoring_data.append(monitoring_record)
            
            # Keep only recent data (last 24 hours by default)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.monitoring_data = [
                record for record in self.monitoring_data
                if datetime.fromisoformat(record["timestamp"]) > cutoff_time
            ]
            
            self.logger.debug("Logged prediction for monitoring", 
                            model_name=model_name, 
                            vehicle_id=vehicle_id,
                            success=error is None)
            
        except Exception as e:
            self.logger.error("Failed to log prediction", error=str(e))
    
    def log_performance_metrics(self, 
                               model_name: str,
                               metrics: Dict[str, float],
                               evaluation_timestamp: Optional[str] = None) -> None:
        """Log performance metrics for a model.
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics dictionary
            evaluation_timestamp: Timestamp of evaluation
        """
        try:
            performance_record = {
                "timestamp": evaluation_timestamp or datetime.now().isoformat(),
                "model_name": model_name,
                "metrics": metrics
            }
            
            self.performance_history.append(performance_record)
            
            # Keep only recent performance data (last 30 days by default)
            cutoff_time = datetime.now() - timedelta(days=30)
            self.performance_history = [
                record for record in self.performance_history
                if datetime.fromisoformat(record["timestamp"]) > cutoff_time
            ]
            
            self.logger.info("Logged performance metrics", 
                           model_name=model_name, 
                           metrics=metrics)
            
        except Exception as e:
            self.logger.error("Failed to log performance metrics", error=str(e))
    
    def get_model_performance(self, model_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics for a model.
        
        Args:
            model_name: Name of the model
            hours: Time window in hours
            
        Returns:
            Performance statistics
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter monitoring data for the model and time window
            model_data = [
                record for record in self.monitoring_data
                if (record["model_name"] == model_name and 
                    datetime.fromisoformat(record["timestamp"]) > cutoff_time)
            ]
            
            if not model_data:
                return {
                    "model_name": model_name,
                    "total_predictions": 0,
                    "success_rate": 0.0,
                    "avg_inference_time": 0.0,
                    "error_rate": 0.0
                }
            
            total_predictions = len(model_data)
            successful_predictions = sum(1 for record in model_data if record["success"])
            success_rate = successful_predictions / total_predictions
            avg_inference_time = np.mean([record["inference_time"] for record in model_data])
            error_rate = 1.0 - success_rate
            
            return {
                "model_name": model_name,
                "total_predictions": total_predictions,
                "success_rate": success_rate,
                "avg_inference_time": avg_inference_time,
                "error_rate": error_rate,
                "time_window_hours": hours
            }
            
        except Exception as e:
            self.logger.error("Failed to get model performance", error=str(e))
            return {}
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for monitoring alerts.
        
        Returns:
            List of alerts
        """
        alerts = []
        
        try:
            # Get unique model names
            model_names = set(record["model_name"] for record in self.monitoring_data)
            
            for model_name in model_names:
                performance = self.get_model_performance(model_name)
                
                if not performance:
                    continue
                
                # Check error rate
                if performance["error_rate"] > self.alert_thresholds["error_rate"]:
                    alerts.append({
                        "type": "high_error_rate",
                        "model_name": model_name,
                        "severity": "high",
                        "message": f"Error rate {performance['error_rate']:.2%} exceeds threshold {self.alert_thresholds['error_rate']:.2%}",
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Check inference time
                if performance["avg_inference_time"] > self.alert_thresholds["latency_increase"]:
                    alerts.append({
                        "type": "high_latency",
                        "model_name": model_name,
                        "severity": "medium",
                        "message": f"Average inference time {performance['avg_inference_time']:.3f}s exceeds threshold {self.alert_thresholds['latency_increase']}s",
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Check success rate
                if performance["success_rate"] < (1.0 - self.alert_thresholds["error_rate"]):
                    alerts.append({
                        "type": "low_success_rate",
                        "model_name": model_name,
                        "severity": "high",
                        "message": f"Success rate {performance['success_rate']:.2%} below threshold {(1.0 - self.alert_thresholds['error_rate']):.2%}",
                        "timestamp": datetime.now().isoformat()
                    })
            
            if alerts:
                self.logger.warning("Monitoring alerts detected", alert_count=len(alerts))
            
        except Exception as e:
            self.logger.error("Failed to check alerts", error=str(e))
        
        return alerts
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary.
        
        Returns:
            Monitoring summary
        """
        try:
            total_predictions = len(self.monitoring_data)
            total_errors = sum(1 for record in self.monitoring_data if not record["success"])
            overall_success_rate = (total_predictions - total_errors) / total_predictions if total_predictions > 0 else 0.0
            
            model_performance = {}
            model_names = set(record["model_name"] for record in self.monitoring_data)
            
            for model_name in model_names:
                model_performance[model_name] = self.get_model_performance(model_name)
            
            alerts = self.check_alerts()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_predictions": total_predictions,
                "overall_success_rate": overall_success_rate,
                "total_errors": total_errors,
                "model_performance": model_performance,
                "active_alerts": len(alerts),
                "alerts": alerts
            }
            
        except Exception as e:
            self.logger.error("Failed to get monitoring summary", error=str(e))
            return {}
    
    def save_monitoring_data(self, filepath: str) -> None:
        """Save monitoring data to file.
        
        Args:
            filepath: Path to save the data
        """
        try:
            monitoring_summary = self.get_monitoring_summary()
            
            with open(filepath, "w") as f:
                json.dump(monitoring_summary, f, indent=2, default=str)
            
            self.logger.info("Saved monitoring data", filepath=filepath)
            
        except Exception as e:
            self.logger.error("Failed to save monitoring data", error=str(e))


class DataDriftDetector:
    """Detect data drift in production data."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize data drift detector.
        
        Args:
            config: Configuration object
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.reference_data: Optional[pd.DataFrame] = None
        self.drift_history: List[Dict[str, Any]] = []
        
        self.logger.info("Data drift detector initialized")
    
    def set_reference_data(self, reference_data: pd.DataFrame) -> None:
        """Set reference data for drift detection.
        
        Args:
            reference_data: Reference dataset
        """
        try:
            self.reference_data = reference_data.copy()
            self.logger.info("Set reference data for drift detection", 
                           shape=reference_data.shape)
        except Exception as e:
            self.logger.error("Failed to set reference data", error=str(e))
    
    def load_reference_data(self, filepath: str) -> None:
        """Load reference data from file.
        
        Args:
            filepath: Path to reference data file
        """
        try:
            if filepath.endswith('.parquet'):
                self.reference_data = pd.read_parquet(filepath)
            elif filepath.endswith('.csv'):
                self.reference_data = pd.read_csv(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            self.logger.info("Loaded reference data", filepath=filepath, shape=self.reference_data.shape)
            
        except Exception as e:
            self.logger.error("Failed to load reference data", error=str(e))
    
    def detect_drift(self, current_data: pd.DataFrame, 
                    column_mapping: Optional[DataDefinition] = None) -> Dict[str, Any]:
        """Detect data drift between reference and current data.
        
        Args:
            current_data: Current production data
            column_mapping: Column mapping for Evidently
            
        Returns:
            Drift detection results
        """
        try:
            if self.reference_data is None:
                raise ValueError("Reference data not set")
            
            # Create default column mapping if not provided
            if column_mapping is None:
                column_mapping = DataDefinition(
                    target=None,  # No target for drift detection
                    numerical_features=['x', 'y', 'velocity', 'acceleration', 'heading'],
                    categorical_features=['vehicle_id']
                )
            
            # Simple drift detection using basic statistical comparison
            drift_score = 0.0
            drifted_columns = []
            detailed_metrics = {}
            
            # Compare basic statistics for numerical columns
            numerical_cols = ['x', 'y', 'velocity', 'acceleration', 'heading']
            for col in numerical_cols:
                if col in self.reference_data.columns and col in current_data.columns:
                    ref_mean = self.reference_data[col].mean()
                    ref_std = self.reference_data[col].std()
                    curr_mean = current_data[col].mean()
                    curr_std = current_data[col].std()
                    
                    # Simple drift detection based on mean/std differences
                    mean_diff = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
                    std_diff = abs(curr_std - ref_std) / (ref_std + 1e-8)
                    
                    col_drift_score = (mean_diff + std_diff) / 2
                    detailed_metrics[col] = {
                        "drift_detected": col_drift_score > 0.5,
                        "drift_score": col_drift_score,
                        "statistical_test": "mean_std_comparison"
                    }
                    
                    if col_drift_score > 0.5:
                        drifted_columns.append(col)
                        drift_score += col_drift_score
            
            if detailed_metrics:
                drift_score /= len(detailed_metrics)
            
            # Create drift result
            drift_result = {
                "timestamp": datetime.now().isoformat(),
                "drift_score": drift_score,
                "drift_detected": drift_score > 0.3,  # Threshold for drift detection
                "drifted_columns": drifted_columns,
                "total_columns": len(detailed_metrics),
                "reference_data_shape": self.reference_data.shape,
                "current_data_shape": current_data.shape,
                "detailed_metrics": detailed_metrics
            }
            
            # Store drift history
            self.drift_history.append(drift_result)
            
            # Keep only recent drift history (last 30 days)
            cutoff_time = datetime.now() - timedelta(days=30)
            self.drift_history = [
                record for record in self.drift_history
                if datetime.fromisoformat(record["timestamp"]) > cutoff_time
            ]
            
            self.logger.info("Data drift detection completed", 
                           drift_score=drift_score,
                           drift_detected=drift_result["drift_detected"],
                           drifted_columns=drifted_columns)
            
            return drift_result
            
        except Exception as e:
            self.logger.error("Failed to detect data drift", error=str(e))
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "drift_score": 0.0,
                "drift_detected": False
            }
    
    def detect_target_drift(self, reference_data: pd.DataFrame, 
                           current_data: pd.DataFrame,
                           target_column: str,
                           column_mapping: Optional[DataDefinition] = None) -> Dict[str, Any]:
        """Detect target drift between reference and current data.
        
        Args:
            reference_data: Reference dataset with target
            current_data: Current dataset with target
            target_column: Name of the target column
            column_mapping: Column mapping for Evidently
            
        Returns:
            Target drift detection results
        """
        try:
            if target_column not in reference_data.columns or target_column not in current_data.columns:
                raise ValueError(f"Target column {target_column} not found in data")
            
            # Simple target drift detection using basic statistical comparison
            ref_target = reference_data[target_column]
            curr_target = current_data[target_column]
            
            ref_mean = ref_target.mean()
            ref_std = ref_target.std()
            curr_mean = curr_target.mean()
            curr_std = curr_target.std()
            
            # Calculate drift score based on distribution differences
            mean_diff = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
            std_diff = abs(curr_std - ref_std) / (ref_std + 1e-8)
            drift_score = (mean_diff + std_diff) / 2
            
            target_drift_result = {
                "timestamp": datetime.now().isoformat(),
                "target_column": target_column,
                "drift_detected": drift_score > 0.5,
                "drift_score": drift_score,
                "reference_target_stats": {
                    "mean": ref_mean,
                    "std": ref_std,
                    "count": len(ref_target)
                },
                "current_target_stats": {
                    "mean": curr_mean,
                    "std": curr_std,
                    "count": len(curr_target)
                }
            }
            
            self.logger.info("Target drift detection completed", 
                           target_column=target_column,
                           drift_detected=target_drift_result["drift_detected"],
                           drift_score=target_drift_result["drift_score"])
            
            return target_drift_result
            
        except Exception as e:
            self.logger.error("Failed to detect target drift", error=str(e))
            return {
                "timestamp": datetime.now().isoformat(),
                "target_column": target_column,
                "error": str(e),
                "drift_detected": False,
                "drift_score": 0.0
            }
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get drift detection summary.
        
        Returns:
            Drift summary
        """
        try:
            if not self.drift_history:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "total_drift_checks": 0,
                    "drift_detected_count": 0,
                    "average_drift_score": 0.0
                }
            
            total_checks = len(self.drift_history)
            drift_detected_count = sum(1 for record in self.drift_history if record.get("drift_detected", False))
            average_drift_score = np.mean([record.get("drift_score", 0.0) for record in self.drift_history])
            
            # Get most recent drift check
            latest_drift = self.drift_history[-1] if self.drift_history else {}
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_drift_checks": total_checks,
                "drift_detected_count": drift_detected_count,
                "average_drift_score": average_drift_score,
                "latest_drift": latest_drift,
                "drift_trend": self._calculate_drift_trend()
            }
            
        except Exception as e:
            self.logger.error("Failed to get drift summary", error=str(e))
            return {}
    
    def _calculate_drift_trend(self) -> str:
        """Calculate drift trend over time.
        
        Returns:
            Trend description
        """
        try:
            if len(self.drift_history) < 2:
                return "insufficient_data"
            
            recent_scores = [record.get("drift_score", 0.0) for record in self.drift_history[-5:]]
            if len(recent_scores) < 2:
                return "insufficient_data"
            
            # Simple linear trend calculation
            x = np.arange(len(recent_scores))
            slope = np.polyfit(x, recent_scores, 1)[0]
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            self.logger.error("Failed to calculate drift trend", error=str(e))
            return "unknown"
    
    def save_drift_report(self, filepath: str) -> None:
        """Save drift detection report to file.
        
        Args:
            filepath: Path to save the report
        """
        try:
            drift_summary = self.get_drift_summary()
            
            with open(filepath, "w") as f:
                json.dump(drift_summary, f, indent=2, default=str)
            
            self.logger.info("Saved drift report", filepath=filepath)
            
        except Exception as e:
            self.logger.error("Failed to save drift report", error=str(e))