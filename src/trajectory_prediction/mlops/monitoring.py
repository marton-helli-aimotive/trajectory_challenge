"""
Model monitoring and drift detection system.

This module provides:
- Data drift detection using statistical tests
- Model performance monitoring and alerting
- Feature distribution monitoring
- Automated retraining triggers
- Real-time alerting system
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import mean_squared_error, mean_absolute_error
from omegaconf import DictConfig

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

from ..models.base import TrajectoryPredictor, PredictionResult
from ..data.schemas import TrajectoryData
from ..evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class DriftStatus(Enum):
    """Data drift status levels."""
    NO_DRIFT = "no_drift"
    WARNING = "warning" 
    CRITICAL = "critical"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    feature_name: str
    drift_score: float
    p_value: float
    drift_status: DriftStatus
    test_method: str
    reference_stats: Dict[str, float]
    current_stats: Dict[str, float]
    detected_at: str
    threshold_used: float


@dataclass
class PerformanceMetric:
    """Performance monitoring metric."""
    metric_name: str
    current_value: float
    reference_value: float
    deviation_percent: float
    status: DriftStatus
    measured_at: str
    threshold_warning: float
    threshold_critical: float


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    alert_type: str
    level: AlertLevel
    title: str
    message: str
    context: Dict[str, Any]
    created_at: str
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None


class DataDriftDetector:
    """
    Statistical data drift detection system.
    
    Monitors input data distributions and detects significant changes.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.drift_config = config.get("drift_detection", {})
        
        # Drift detection thresholds
        self.warning_threshold = self.drift_config.get("warning_threshold", 0.05)
        self.critical_threshold = self.drift_config.get("critical_threshold", 0.01)
        
        # Reference data storage
        self.reference_data = {}
        self.reference_stats = {}
        
        # Drift detection results
        self.drift_history = []
        
        logger.info("Data drift detector initialized")
    
    async def set_reference_data(self, reference_data: List[TrajectoryData]) -> None:
        """Set reference dataset for drift detection."""
        
        try:
            # Extract features from reference data
            reference_features = await self._extract_features(reference_data)
            self.reference_data = reference_features
            
            # Calculate reference statistics
            self.reference_stats = {}
            
            for feature_name, values in reference_features.items():
                if len(values) > 0:
                    self.reference_stats[feature_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "q25": np.percentile(values, 25),
                        "q50": np.percentile(values, 50),
                        "q75": np.percentile(values, 75),
                        "count": len(values),
                        "hist_bins": np.histogram(values, bins=20)[1].tolist(),
                        "hist_counts": np.histogram(values, bins=20)[0].tolist()
                    }
            
            logger.info(f"Reference data set with {len(reference_data)} samples and {len(self.reference_stats)} features")
            
        except Exception as e:
            logger.error(f"Failed to set reference data: {e}")
            raise
    
    async def detect_drift(self, current_data: List[TrajectoryData]) -> List[DriftDetectionResult]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current trajectory data to compare against reference
            
        Returns:
            List of drift detection results for each feature
        """
        
        if not self.reference_data:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        try:
            # Extract features from current data
            current_features = await self._extract_features(current_data)
            
            drift_results = []
            
            for feature_name in self.reference_data.keys():
                if feature_name in current_features:
                    
                    reference_values = self.reference_data[feature_name]
                    current_values = current_features[feature_name]
                    
                    if len(current_values) > 10:  # Minimum sample size
                        drift_result = await self._perform_drift_test(
                            feature_name, reference_values, current_values
                        )
                        drift_results.append(drift_result)
            
            # Store results in history
            self.drift_history.extend(drift_results)
            
            logger.info(f"Drift detection completed: {len(drift_results)} features analyzed")
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            raise
    
    async def _extract_features(self, trajectory_data: List[TrajectoryData]) -> Dict[str, List[float]]:
        """Extract numerical features from trajectory data."""
        
        features = {
            "velocity_mean": [],
            "velocity_std": [],
            "acceleration_mean": [],
            "acceleration_std": [],
            "trajectory_length": [],
            "curvature_mean": [],
            "lateral_position_std": [],
            "time_duration": []
        }
        
        for trajectory in trajectory_data:
            try:
                positions = np.array(trajectory.positions)
                time_steps = np.array(trajectory.time_steps)
                
                if len(positions) < 3:
                    continue
                
                # Velocity features
                velocities = np.diff(positions, axis=0) / np.diff(time_steps)[:, np.newaxis]
                velocity_magnitudes = np.linalg.norm(velocities, axis=1)
                
                features["velocity_mean"].append(np.mean(velocity_magnitudes))
                features["velocity_std"].append(np.std(velocity_magnitudes))
                
                # Acceleration features
                if len(velocities) > 1:
                    accelerations = np.diff(velocities, axis=0) / np.diff(time_steps[1:])[:, np.newaxis]
                    acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
                    
                    features["acceleration_mean"].append(np.mean(acceleration_magnitudes))
                    features["acceleration_std"].append(np.std(acceleration_magnitudes))
                else:
                    features["acceleration_mean"].append(0.0)
                    features["acceleration_std"].append(0.0)
                
                # Trajectory characteristics
                path_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                features["trajectory_length"].append(np.sum(path_lengths))
                
                # Curvature estimation (simplified)
                if len(positions) > 2:
                    dx = np.gradient(positions[:, 0])
                    dy = np.gradient(positions[:, 1])
                    d2x = np.gradient(dx)
                    d2y = np.gradient(dy)
                    
                    curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5
                    curvature = curvature[np.isfinite(curvature)]
                    
                    if len(curvature) > 0:
                        features["curvature_mean"].append(np.mean(curvature))
                    else:
                        features["curvature_mean"].append(0.0)
                else:
                    features["curvature_mean"].append(0.0)
                
                # Lateral deviation
                features["lateral_position_std"].append(np.std(positions[:, 1]))
                
                # Time characteristics
                features["time_duration"].append(time_steps[-1] - time_steps[0])
                
            except Exception as e:
                logger.warning(f"Failed to extract features from trajectory: {e}")
                continue
        
        return features
    
    async def _perform_drift_test(
        self,
        feature_name: str,
        reference_values: List[float],
        current_values: List[float]
    ) -> DriftDetectionResult:
        """Perform statistical drift test on feature."""
        
        try:
            # Clean data
            ref_clean = [v for v in reference_values if np.isfinite(v)]
            cur_clean = [v for v in current_values if np.isfinite(v)]
            
            if len(ref_clean) < 10 or len(cur_clean) < 10:
                # Insufficient data
                return DriftDetectionResult(
                    feature_name=feature_name,
                    drift_score=0.0,
                    p_value=1.0,
                    drift_status=DriftStatus.NO_DRIFT,
                    test_method="insufficient_data",
                    reference_stats={},
                    current_stats={},
                    detected_at=datetime.now().isoformat(),
                    threshold_used=self.warning_threshold
                )
            
            # Kolmogorov-Smirnov test for continuous features
            ks_statistic, ks_p_value = ks_2samp(ref_clean, cur_clean)
            
            # Calculate current statistics
            current_stats = {
                "mean": np.mean(cur_clean),
                "std": np.std(cur_clean),
                "min": np.min(cur_clean),
                "max": np.max(cur_clean),
                "q25": np.percentile(cur_clean, 25),
                "q50": np.percentile(cur_clean, 50),
                "q75": np.percentile(cur_clean, 75),
                "count": len(cur_clean)
            }
            
            reference_stats = self.reference_stats.get(feature_name, {})
            
            # Determine drift status
            if ks_p_value < self.critical_threshold:
                drift_status = DriftStatus.CRITICAL
            elif ks_p_value < self.warning_threshold:
                drift_status = DriftStatus.WARNING
            else:
                drift_status = DriftStatus.NO_DRIFT
            
            return DriftDetectionResult(
                feature_name=feature_name,
                drift_score=ks_statistic,
                p_value=ks_p_value,
                drift_status=drift_status,
                test_method="kolmogorov_smirnov",
                reference_stats=reference_stats,
                current_stats=current_stats,
                detected_at=datetime.now().isoformat(),
                threshold_used=self.warning_threshold
            )
            
        except Exception as e:
            logger.error(f"Drift test failed for {feature_name}: {e}")
            
            return DriftDetectionResult(
                feature_name=feature_name,
                drift_score=0.0,
                p_value=1.0,
                drift_status=DriftStatus.NO_DRIFT,
                test_method="error",
                reference_stats={},
                current_stats={},
                detected_at=datetime.now().isoformat(),
                threshold_used=self.warning_threshold
            )
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of recent drift detection results."""
        
        if not self.drift_history:
            return {"message": "No drift detection results available"}
        
        # Get recent results (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_results = [
            r for r in self.drift_history
            if datetime.fromisoformat(r.detected_at) > recent_cutoff
        ]
        
        if not recent_results:
            recent_results = self.drift_history[-10:]  # Last 10 results
        
        # Count by status
        status_counts = {}
        for status in DriftStatus:
            status_counts[status.value] = len([r for r in recent_results if r.drift_status == status])
        
        # Find most drifted features
        critical_features = [r.feature_name for r in recent_results if r.drift_status == DriftStatus.CRITICAL]
        warning_features = [r.feature_name for r in recent_results if r.drift_status == DriftStatus.WARNING]
        
        return {
            "total_features_monitored": len(set(r.feature_name for r in recent_results)),
            "status_counts": status_counts,
            "critical_features": critical_features,
            "warning_features": warning_features,
            "last_detection": recent_results[-1].detected_at if recent_results else None,
            "recommendation": self._generate_drift_recommendation(recent_results)
        }
    
    def _generate_drift_recommendation(self, drift_results: List[DriftDetectionResult]) -> str:
        """Generate recommendation based on drift detection results."""
        
        critical_count = len([r for r in drift_results if r.drift_status == DriftStatus.CRITICAL])
        warning_count = len([r for r in drift_results if r.drift_status == DriftStatus.WARNING])
        
        if critical_count > 0:
            return f"🚨 CRITICAL: {critical_count} features showing critical drift. Consider model retraining immediately."
        elif warning_count > len(drift_results) * 0.5:
            return f"⚠️ WARNING: {warning_count} features showing drift. Monitor closely and consider retraining."
        elif warning_count > 0:
            return f"📊 NOTICE: {warning_count} features showing minor drift. Continue monitoring."
        else:
            return "✅ No significant drift detected. Model inputs remain stable."


class PerformanceMonitor:
    """
    Model performance monitoring system.
    
    Tracks model performance metrics over time and detects degradation.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.perf_config = config.get("performance_monitoring", {})
        
        # Performance thresholds
        self.warning_threshold = self.perf_config.get("warning_threshold_percent", 10.0)  # 10% degradation
        self.critical_threshold = self.perf_config.get("critical_threshold_percent", 25.0)  # 25% degradation
        
        # Reference performance metrics
        self.reference_metrics = {}
        self.performance_history = []
        
        # Model evaluator
        self.evaluator = None
        
        logger.info("Performance monitor initialized")
    
    async def set_reference_performance(
        self,
        model: TrajectoryPredictor,
        reference_data: List[TrajectoryData],
        prediction_horizon: float = 10.0
    ) -> None:
        """Set reference performance metrics."""
        
        try:
            if self.evaluator is None:
                from ..evaluation.evaluator import ModelEvaluator
                self.evaluator = ModelEvaluator(self.config)
            
            # Evaluate model on reference data
            evaluation_results = await self.evaluator.evaluate_model(
                model, reference_data, prediction_horizon
            )
            
            # Extract key performance metrics
            self.reference_metrics = {}
            
            for category in ["trajectory_metrics", "safety_metrics", "probabilistic_metrics"]:
                if category in evaluation_results:
                    for metric_name, metric_data in evaluation_results[category].items():
                        if isinstance(metric_data, dict) and "mean" in metric_data:
                            key = f"{category}_{metric_name}"
                            self.reference_metrics[key] = metric_data["mean"]
            
            self.reference_metrics["reference_set_at"] = datetime.now().isoformat()
            self.reference_metrics["reference_data_size"] = len(reference_data)
            self.reference_metrics["prediction_horizon"] = prediction_horizon
            
            logger.info(f"Reference performance set with {len(self.reference_metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Failed to set reference performance: {e}")
            raise
    
    async def monitor_performance(
        self,
        model: TrajectoryPredictor,
        current_data: List[TrajectoryData],
        prediction_horizon: float = 10.0
    ) -> List[PerformanceMetric]:
        """
        Monitor current model performance against reference.
        
        Args:
            model: Model to evaluate
            current_data: Current data to evaluate on
            prediction_horizon: Prediction horizon to use
            
        Returns:
            List of performance metrics with status
        """
        
        if not self.reference_metrics:
            raise ValueError("Reference performance not set. Call set_reference_performance() first.")
        
        try:
            if self.evaluator is None:
                from ..evaluation.evaluator import ModelEvaluator
                self.evaluator = ModelEvaluator(self.config)
            
            # Evaluate current performance
            current_evaluation = await self.evaluator.evaluate_model(
                model, current_data, prediction_horizon
            )
            
            # Extract current metrics
            current_metrics = {}
            
            for category in ["trajectory_metrics", "safety_metrics", "probabilistic_metrics"]:
                if category in current_evaluation:
                    for metric_name, metric_data in current_evaluation[category].items():
                        if isinstance(metric_data, dict) and "mean" in metric_data:
                            key = f"{category}_{metric_name}"
                            current_metrics[key] = metric_data["mean"]
            
            # Compare with reference
            performance_metrics = []
            measured_at = datetime.now().isoformat()
            
            for metric_name in self.reference_metrics.keys():
                if (metric_name in current_metrics and 
                    not metric_name.startswith("reference_") and
                    metric_name != "prediction_horizon"):
                    
                    reference_value = self.reference_metrics[metric_name]
                    current_value = current_metrics[metric_name]
                    
                    # Calculate deviation percentage
                    if reference_value != 0:
                        deviation_percent = abs((current_value - reference_value) / reference_value) * 100
                    else:
                        deviation_percent = 0.0
                    
                    # Determine status
                    if deviation_percent > self.critical_threshold:
                        status = DriftStatus.CRITICAL
                    elif deviation_percent > self.warning_threshold:
                        status = DriftStatus.WARNING
                    else:
                        status = DriftStatus.NO_DRIFT
                    
                    perf_metric = PerformanceMetric(
                        metric_name=metric_name,
                        current_value=current_value,
                        reference_value=reference_value,
                        deviation_percent=deviation_percent,
                        status=status,
                        measured_at=measured_at,
                        threshold_warning=self.warning_threshold,
                        threshold_critical=self.critical_threshold
                    )
                    
                    performance_metrics.append(perf_metric)
            
            # Store in history
            self.performance_history.extend(performance_metrics)
            
            logger.info(f"Performance monitoring completed: {len(performance_metrics)} metrics analyzed")
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
            raise
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of recent performance monitoring."""
        
        if not self.performance_history:
            return {"message": "No performance monitoring results available"}
        
        # Get recent results (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_results = [
            r for r in self.performance_history
            if datetime.fromisoformat(r.measured_at) > recent_cutoff
        ]
        
        if not recent_results:
            recent_results = self.performance_history[-10:]  # Last 10 results
        
        # Count by status
        status_counts = {}
        for status in DriftStatus:
            status_counts[status.value] = len([r for r in recent_results if r.status == status])
        
        # Find degraded metrics
        critical_metrics = [r.metric_name for r in recent_results if r.status == DriftStatus.CRITICAL]
        warning_metrics = [r.metric_name for r in recent_results if r.status == DriftStatus.WARNING]
        
        # Average degradation
        avg_degradation = np.mean([r.deviation_percent for r in recent_results]) if recent_results else 0.0
        
        return {
            "total_metrics_monitored": len(set(r.metric_name for r in recent_results)),
            "status_counts": status_counts,
            "critical_metrics": critical_metrics,
            "warning_metrics": warning_metrics,
            "average_degradation_percent": avg_degradation,
            "last_measurement": recent_results[-1].measured_at if recent_results else None,
            "recommendation": self._generate_performance_recommendation(recent_results)
        }
    
    def _generate_performance_recommendation(self, perf_results: List[PerformanceMetric]) -> str:
        """Generate recommendation based on performance monitoring."""
        
        critical_count = len([r for r in perf_results if r.status == DriftStatus.CRITICAL])
        warning_count = len([r for r in perf_results if r.status == DriftStatus.WARNING])
        
        avg_degradation = np.mean([r.deviation_percent for r in perf_results]) if perf_results else 0.0
        
        if critical_count > 0:
            return f"🚨 CRITICAL: {critical_count} metrics critically degraded (avg: {avg_degradation:.1f}%). Immediate retraining required."
        elif warning_count > len(perf_results) * 0.5:
            return f"⚠️ WARNING: {warning_count} metrics showing degradation (avg: {avg_degradation:.1f}%). Consider retraining."
        elif avg_degradation > 5.0:
            return f"📊 NOTICE: Average degradation of {avg_degradation:.1f}%. Monitor trends closely."
        else:
            return "✅ Performance stable. No action required."


class AlertingSystem:
    """
    Comprehensive alerting system for model monitoring.
    
    Handles alert generation, notification, and resolution tracking.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.alert_config = config.get("alerting", {})
        
        # Alert storage
        self.alerts = []
        self.alert_handlers = {}
        
        # Email configuration (if available)
        if EMAIL_AVAILABLE:
            self.email_config = self.alert_config.get("email", {})
            self.smtp_server = self.email_config.get("smtp_server")
            self.smtp_port = self.email_config.get("smtp_port", 587)
            self.email_username = self.email_config.get("username")
            self.email_password = self.email_config.get("password")
            self.notification_recipients = self.email_config.get("recipients", [])
        
        # Register default alert handlers
        self._register_default_handlers()
        
        logger.info("Alerting system initialized")
    
    def _register_default_handlers(self) -> None:
        """Register default alert handlers."""
        
        # Console logging handler
        self.alert_handlers["console"] = self._console_alert_handler
        
        # File logging handler
        if self.alert_config.get("log_to_file", True):
            self.alert_handlers["file"] = self._file_alert_handler
        
        # Email handler (if configured)
        if EMAIL_AVAILABLE and self.notification_recipients:
            self.alert_handlers["email"] = self._email_alert_handler
    
    async def create_alert(
        self,
        alert_type: str,
        level: AlertLevel,
        title: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create and process new alert."""
        
        alert = Alert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alerts)}",
            alert_type=alert_type,
            level=level,
            title=title,
            message=message,
            context=context or {},
            created_at=datetime.now().isoformat()
        )
        
        self.alerts.append(alert)
        
        # Process alert through handlers
        await self._process_alert(alert)
        
        return alert
    
    async def _process_alert(self, alert: Alert) -> None:
        """Process alert through registered handlers."""
        
        for handler_name, handler_func in self.alert_handlers.items():
            try:
                await handler_func(alert)
            except Exception as e:
                logger.error(f"Alert handler {handler_name} failed: {e}")
    
    async def _console_alert_handler(self, alert: Alert) -> None:
        """Console alert handler."""
        
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️", 
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.EMERGENCY: "🔴"
        }
        
        emoji = level_emoji.get(alert.level, "📢")
        
        console_message = f"{emoji} [{alert.level.value.upper()}] {alert.title}\n{alert.message}"
        
        if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            logger.error(console_message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(console_message)
        else:
            logger.info(console_message)
    
    async def _file_alert_handler(self, alert: Alert) -> None:
        """File logging alert handler."""
        
        log_dir = Path(self.alert_config.get("log_directory", "logs/alerts"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "alerts.log"
        
        log_entry = {
            "timestamp": alert.created_at,
            "alert_id": alert.alert_id,
            "type": alert.alert_type,
            "level": alert.level.value,
            "title": alert.title,
            "message": alert.message,
            "context": alert.context
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    
    async def _email_alert_handler(self, alert: Alert) -> None:
        """Email notification alert handler."""
        
        if not EMAIL_AVAILABLE or not self.notification_recipients:
            return
        
        # Only send email for warning level and above
        if alert.level not in [AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            return
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['To'] = ', '.join(self.notification_recipients)
            msg['Subject'] = f"[{alert.level.value.upper()}] Trajectory Prediction Alert: {alert.title}"
            
            # Email body
            body = f"""
Trajectory Prediction Model Alert

Alert ID: {alert.alert_id}
Type: {alert.alert_type}
Level: {alert.level.value.upper()}
Time: {alert.created_at}

Title: {alert.title}

Message:
{alert.message}

Context:
{json.dumps(alert.context, indent=2)}

---
This is an automated alert from the Trajectory Prediction Monitoring System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            if self.smtp_server and self.email_username and self.email_password:
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.email_username, self.email_password)
                    text = msg.as_string()
                    server.sendmail(self.email_username, self.notification_recipients, text)
                
                logger.info(f"Alert email sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert."""
        
        alert = next((a for a in self.alerts if a.alert_id == alert_id), None)
        
        if alert:
            alert.resolved_at = datetime.now().isoformat()
            alert.resolved_by = resolved_by
            
            logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
            return True
        
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts."""
        return [a for a in self.alerts if a.resolved_at is None]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by severity level."""
        return [a for a in self.alerts if a.level == level]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        
        active_alerts = self.get_active_alerts()
        
        # Count by level
        level_counts = {}
        for level in AlertLevel:
            level_counts[level.value] = len([a for a in active_alerts if a.level == level])
        
        # Recent alerts (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_alerts = [
            a for a in self.alerts
            if datetime.fromisoformat(a.created_at) > recent_cutoff
        ]
        
        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "recent_alerts_24h": len(recent_alerts),
            "active_by_level": level_counts,
            "critical_active": level_counts.get("critical", 0) + level_counts.get("emergency", 0),
            "last_alert": self.alerts[-1].created_at if self.alerts else None
        }


class ModelMonitor:
    """
    High-level model monitoring orchestrator.
    
    Integrates drift detection, performance monitoring, and alerting.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # Initialize monitoring components
        self.drift_detector = DataDriftDetector(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.alerting_system = AlertingSystem(config)
        
        # Monitoring state
        self.monitoring_active = False
        self.last_monitoring_run = None
        
        logger.info("Model monitor initialized")
    
    async def setup_monitoring(
        self,
        model: TrajectoryPredictor,
        reference_data: List[TrajectoryData],
        prediction_horizon: float = 10.0
    ) -> None:
        """Setup monitoring with reference data and model."""
        
        try:
            # Set up drift detection
            await self.drift_detector.set_reference_data(reference_data)
            
            # Set up performance monitoring
            await self.performance_monitor.set_reference_performance(
                model, reference_data, prediction_horizon
            )
            
            self.monitoring_active = True
            
            await self.alerting_system.create_alert(
                alert_type="monitoring_setup",
                level=AlertLevel.INFO,
                title="Model Monitoring Activated",
                message=f"Monitoring setup complete for {model.name} with {len(reference_data)} reference samples",
                context={
                    "model_name": model.name,
                    "reference_size": len(reference_data),
                    "prediction_horizon": prediction_horizon
                }
            )
            
            logger.info(f"Model monitoring setup complete for {model.name}")
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring: {e}")
            
            await self.alerting_system.create_alert(
                alert_type="monitoring_setup_failure",
                level=AlertLevel.CRITICAL,
                title="Model Monitoring Setup Failed",
                message=f"Failed to setup monitoring: {str(e)}",
                context={"error": str(e)}
            )
            
            raise
    
    async def run_monitoring_cycle(
        self,
        model: TrajectoryPredictor,
        current_data: List[TrajectoryData],
        prediction_horizon: float = 10.0
    ) -> Dict[str, Any]:
        """Run complete monitoring cycle."""
        
        if not self.monitoring_active:
            raise ValueError("Monitoring not setup. Call setup_monitoring() first.")
        
        monitoring_results = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model.name,
            "data_size": len(current_data),
            "drift_results": [],
            "performance_results": [],
            "alerts_generated": [],
            "recommendations": []
        }
        
        try:
            # Run drift detection
            logger.info("Running drift detection...")
            drift_results = await self.drift_detector.detect_drift(current_data)
            monitoring_results["drift_results"] = drift_results
            
            # Run performance monitoring
            logger.info("Running performance monitoring...")
            perf_results = await self.performance_monitor.monitor_performance(
                model, current_data, prediction_horizon
            )
            monitoring_results["performance_results"] = perf_results
            
            # Generate alerts based on results
            alerts_generated = await self._generate_monitoring_alerts(drift_results, perf_results)
            monitoring_results["alerts_generated"] = alerts_generated
            
            # Generate recommendations
            recommendations = self._generate_monitoring_recommendations(drift_results, perf_results)
            monitoring_results["recommendations"] = recommendations
            
            self.last_monitoring_run = datetime.now().isoformat()
            
            logger.info(f"Monitoring cycle completed: {len(drift_results)} drift checks, {len(perf_results)} performance checks")
            
        except Exception as e:
            logger.error(f"Monitoring cycle failed: {e}")
            
            await self.alerting_system.create_alert(
                alert_type="monitoring_cycle_failure",
                level=AlertLevel.CRITICAL,
                title="Monitoring Cycle Failed",
                message=f"Monitoring cycle failed: {str(e)}",
                context={"error": str(e), "model_name": model.name}
            )
            
            monitoring_results["error"] = str(e)
        
        return monitoring_results
    
    async def _generate_monitoring_alerts(
        self,
        drift_results: List[DriftDetectionResult],
        perf_results: List[PerformanceMetric]
    ) -> List[Alert]:
        """Generate alerts based on monitoring results."""
        
        alerts_generated = []
        
        # Drift alerts
        critical_drift = [r for r in drift_results if r.drift_status == DriftStatus.CRITICAL]
        warning_drift = [r for r in drift_results if r.drift_status == DriftStatus.WARNING]
        
        if critical_drift:
            alert = await self.alerting_system.create_alert(
                alert_type="data_drift_critical",
                level=AlertLevel.CRITICAL,
                title=f"Critical Data Drift Detected",
                message=f"Critical drift detected in {len(critical_drift)} features: {', '.join(r.feature_name for r in critical_drift[:5])}",
                context={
                    "critical_features": [r.feature_name for r in critical_drift],
                    "drift_scores": [r.drift_score for r in critical_drift]
                }
            )
            alerts_generated.append(alert)
        
        elif len(warning_drift) > len(drift_results) * 0.5:  # More than 50% features drifting
            alert = await self.alerting_system.create_alert(
                alert_type="data_drift_warning",
                level=AlertLevel.WARNING,
                title=f"Widespread Data Drift Detected",
                message=f"Warning-level drift detected in {len(warning_drift)} features",
                context={
                    "warning_features": [r.feature_name for r in warning_drift]
                }
            )
            alerts_generated.append(alert)
        
        # Performance alerts
        critical_perf = [r for r in perf_results if r.status == DriftStatus.CRITICAL]
        warning_perf = [r for r in perf_results if r.status == DriftStatus.WARNING]
        
        if critical_perf:
            alert = await self.alerting_system.create_alert(
                alert_type="performance_degradation_critical",
                level=AlertLevel.CRITICAL,
                title=f"Critical Performance Degradation",
                message=f"Critical degradation in {len(critical_perf)} metrics: {', '.join(r.metric_name for r in critical_perf[:3])}",
                context={
                    "degraded_metrics": [
                        {"metric": r.metric_name, "degradation": r.deviation_percent}
                        for r in critical_perf
                    ]
                }
            )
            alerts_generated.append(alert)
        
        elif len(warning_perf) > len(perf_results) * 0.5:
            alert = await self.alerting_system.create_alert(
                alert_type="performance_degradation_warning",
                level=AlertLevel.WARNING,
                title=f"Performance Degradation Warning",
                message=f"Performance degradation detected in {len(warning_perf)} metrics",
                context={
                    "warning_metrics": [
                        {"metric": r.metric_name, "degradation": r.deviation_percent}
                        for r in warning_perf
                    ]
                }
            )
            alerts_generated.append(alert)
        
        return alerts_generated
    
    def _generate_monitoring_recommendations(
        self,
        drift_results: List[DriftDetectionResult],
        perf_results: List[PerformanceMetric]
    ) -> List[str]:
        """Generate actionable recommendations based on monitoring results."""
        
        recommendations = []
        
        # Drift recommendations
        critical_drift_count = len([r for r in drift_results if r.drift_status == DriftStatus.CRITICAL])
        warning_drift_count = len([r for r in drift_results if r.drift_status == DriftStatus.WARNING])
        
        if critical_drift_count > 0:
            recommendations.append(f"🚨 IMMEDIATE ACTION: {critical_drift_count} features show critical drift. Retrain model with recent data.")
        elif warning_drift_count > len(drift_results) * 0.3:
            recommendations.append(f"⚠️ MONITOR CLOSELY: {warning_drift_count} features showing drift. Consider data collection review.")
        
        # Performance recommendations
        critical_perf_count = len([r for r in perf_results if r.status == DriftStatus.CRITICAL])
        avg_degradation = np.mean([r.deviation_percent for r in perf_results]) if perf_results else 0
        
        if critical_perf_count > 0:
            recommendations.append(f"🚨 MODEL RETRAINING: {critical_perf_count} metrics critically degraded (avg: {avg_degradation:.1f}%).")
        elif avg_degradation > 10:
            recommendations.append(f"📊 PERFORMANCE REVIEW: Average degradation of {avg_degradation:.1f}%. Investigate causes.")
        
        # Combined recommendations
        if critical_drift_count > 0 and critical_perf_count > 0:
            recommendations.append("🔄 URGENT: Both data drift and performance degradation detected. Execute emergency retraining protocol.")
        
        if not recommendations:
            recommendations.append("✅ System operating normally. Continue regular monitoring.")
        
        return recommendations
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        
        return {
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "last_run": self.last_monitoring_run,
            "drift_summary": self.drift_detector.get_drift_summary(),
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "alert_summary": self.alerting_system.get_alert_summary(),
            "recommendations": self._get_overall_recommendations()
        }
    
    def _get_overall_recommendations(self) -> List[str]:
        """Get overall system recommendations."""
        
        recommendations = []
        
        # Check system health
        drift_summary = self.drift_detector.get_drift_summary()
        perf_summary = self.performance_monitor.get_performance_summary()
        alert_summary = self.alerting_system.get_alert_summary()
        
        # Critical alerts
        if alert_summary["critical_active"] > 0:
            recommendations.append(f"🚨 {alert_summary['critical_active']} critical alerts active. Address immediately.")
        
        # High drift or performance issues
        if "critical_features" in drift_summary and len(drift_summary["critical_features"]) > 0:
            recommendations.append("🔄 Data drift detected. Model retraining recommended.")
        
        if "critical_metrics" in perf_summary and len(perf_summary["critical_metrics"]) > 0:
            recommendations.append("📉 Performance degradation detected. Review model and data.")
        
        # System maintenance
        if not self.last_monitoring_run:
            recommendations.append("⚡ No recent monitoring runs. Ensure monitoring is active.")
        elif self.last_monitoring_run:
            last_run = datetime.fromisoformat(self.last_monitoring_run)
            if datetime.now() - last_run > timedelta(hours=24):
                recommendations.append("⏰ Monitoring data is stale. Run fresh monitoring cycle.")
        
        if not recommendations:
            recommendations.append("✅ All systems operational. Monitoring healthy.")
        
        return recommendations