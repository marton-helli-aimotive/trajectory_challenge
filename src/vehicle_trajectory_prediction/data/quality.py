"""Data quality validation and cleaning pipeline for trajectory data."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import warnings

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Try to import pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Use a simple base class since BaseConfig doesn't exist
class BaseConfig:
    """Base configuration class."""
    pass
from ..core.models import TrajectoryPoint, Trajectory, TrajectoryDataset
from ..core.exceptions import DataQualityError
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityConfig(BaseConfig):
    """Configuration for data quality validation."""
    
    # Validation thresholds
    min_completeness: float = 0.8
    max_missing_rate: float = 0.2
    min_trajectory_length: int = 10
    max_trajectory_length: int = 1000
    
    # Physical constraints
    max_velocity: float = 50.0  # m/s
    max_acceleration: float = 10.0  # m/s²
    max_jerk: float = 20.0  # m/s³
    min_time_gap: float = 0.05  # seconds
    
    # Spatial constraints
    max_position_jump: float = 100.0  # meters
    min_position_change: float = 0.01  # meters
    
    # Quality scoring
    enable_quality_scoring: bool = True
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        'completeness': 0.3,
        'consistency': 0.3,
        'smoothness': 0.2,
        'physics': 0.2
    })
    
    # Cleaning settings
    enable_cleaning: bool = True
    interpolation_method: str = 'linear'
    outlier_detection: bool = True
    outlier_threshold: float = 3.0  # standard deviations
    
    # Reporting
    generate_reports: bool = True
    report_format: str = 'json'  # json, html, csv
    save_reports: bool = True
    report_path: str = "reports/quality"


@dataclass
class QualityMetrics:
    """Data quality metrics for trajectory data."""
    
    # Completeness metrics
    total_trajectories: int = 0
    valid_trajectories: int = 0
    completeness_rate: float = 0.0
    missing_values: Dict[str, int] = field(default_factory=dict)
    
    # Consistency metrics
    duplicate_records: int = 0
    inconsistent_timestamps: int = 0
    invalid_vehicle_ids: int = 0
    
    # Physical constraint violations
    velocity_violations: int = 0
    acceleration_violations: int = 0
    jerk_violations: int = 0
    position_jumps: int = 0
    
    # Smoothness metrics
    trajectory_smoothness: List[float] = field(default_factory=list)
    average_smoothness: float = 0.0
    
    # Quality scores
    overall_quality_score: float = 0.0
    quality_scores: Dict[str, float] = field(default_factory=dict)
    
    # Timestamps
    validation_timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0


class DataQualityPipeline:
    """Data quality validation and cleaning pipeline."""
    
    def __init__(self, config: QualityConfig):
        """Initialize quality pipeline.
        
        Args:
            config: Quality configuration
        """
        self.config = config
        self.report_path = Path(config.report_path)
        self.report_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized data quality pipeline", 
                   config=config.dict())
    
    def validate_trajectory_dataset(self, dataset: TrajectoryDataset) -> QualityMetrics:
        """Validate a trajectory dataset.
        
        Args:
            dataset: TrajectoryDataset to validate
            
        Returns:
            Quality metrics
        """
        start_time = datetime.now()
        
        logger.info("Starting dataset validation", 
                   dataset_name=dataset.name,
                   num_trajectories=len(dataset.trajectories))
        
        metrics = QualityMetrics()
        metrics.total_trajectories = len(dataset.trajectories)
        
        # Validate each trajectory
        valid_trajectories = []
        all_smoothness_scores = []
        
        for trajectory in dataset.trajectories:
            trajectory_metrics = self._validate_trajectory(trajectory)
            
            # Aggregate metrics
            metrics.duplicate_records += trajectory_metrics.get('duplicates', 0)
            metrics.velocity_violations += trajectory_metrics.get('velocity_violations', 0)
            metrics.acceleration_violations += trajectory_metrics.get('acceleration_violations', 0)
            metrics.jerk_violations += trajectory_metrics.get('jerk_violations', 0)
            metrics.position_jumps += trajectory_metrics.get('position_jumps', 0)
            
            smoothness = trajectory_metrics.get('smoothness', 0.0)
            all_smoothness_scores.append(smoothness)
            
            # Check if trajectory is valid
            if trajectory_metrics.get('is_valid', False):
                valid_trajectories.append(trajectory)
        
        metrics.valid_trajectories = len(valid_trajectories)
        metrics.completeness_rate = metrics.valid_trajectories / metrics.total_trajectories
        metrics.trajectory_smoothness = all_smoothness_scores
        if NUMPY_AVAILABLE and all_smoothness_scores:
            metrics.average_smoothness = np.mean(all_smoothness_scores)
        else:
            metrics.average_smoothness = sum(all_smoothness_scores) / len(all_smoothness_scores) if all_smoothness_scores else 0.0
        
        # Calculate quality scores
        if self.config.enable_quality_scoring:
            metrics.quality_scores = self._calculate_quality_scores(metrics)
            metrics.overall_quality_score = np.mean(list(metrics.quality_scores.values()))
        
        # Calculate processing time
        metrics.processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("Dataset validation completed", 
                   valid_trajectories=metrics.valid_trajectories,
                   completeness_rate=metrics.completeness_rate,
                   overall_quality_score=metrics.overall_quality_score,
                   processing_time=metrics.processing_time)
        
        return metrics
    
    def _validate_trajectory(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Validate a single trajectory.
        
        Args:
            trajectory: Trajectory to validate
            
        Returns:
            Dictionary of validation metrics
        """
        metrics = {
            'is_valid': True,
            'duplicates': 0,
            'velocity_violations': 0,
            'acceleration_violations': 0,
            'jerk_violations': 0,
            'position_jumps': 0,
            'smoothness': 0.0
        }
        
        if len(trajectory.points) < self.config.min_trajectory_length:
            metrics['is_valid'] = False
            return metrics
        
        if len(trajectory.points) > self.config.max_trajectory_length:
            metrics['is_valid'] = False
            return metrics
        
        # Check for duplicates
        timestamps = [p.timestamp for p in trajectory.points]
        if len(timestamps) != len(set(timestamps)):
            metrics['duplicates'] = len(timestamps) - len(set(timestamps))
        
        # Check physical constraints
        for i in range(len(trajectory.points)):
            point = trajectory.points[i]
            
            # Velocity violations
            if abs(point.velocity) > self.config.max_velocity:
                metrics['velocity_violations'] += 1
            
            # Acceleration violations
            if abs(point.acceleration) > self.config.max_acceleration:
                metrics['acceleration_violations'] += 1
            
            # Position jumps
            if i > 0:
                prev_point = trajectory.points[i-1]
                distance = np.sqrt((point.x - prev_point.x)**2 + (point.y - prev_point.y)**2)
                time_diff = point.timestamp - prev_point.timestamp
                
                if time_diff > 0 and distance / time_diff > self.config.max_position_jump:
                    metrics['position_jumps'] += 1
        
        # Calculate jerk violations
        for i in range(2, len(trajectory.points)):
            curr_acc = trajectory.points[i].acceleration
            prev_acc = trajectory.points[i-1].acceleration
            time_diff = trajectory.points[i].timestamp - trajectory.points[i-1].timestamp
            
            if time_diff > 0:
                jerk = abs(curr_acc - prev_acc) / time_diff
                if jerk > self.config.max_jerk:
                    metrics['jerk_violations'] += 1
        
        # Calculate smoothness
        if NUMPY_AVAILABLE:
            metrics['smoothness'] = self._calculate_trajectory_smoothness(trajectory)
        else:
            metrics['smoothness'] = 0.0
        
        # Determine if trajectory is valid
        if (metrics['velocity_violations'] > len(trajectory.points) * 0.1 or
            metrics['acceleration_violations'] > len(trajectory.points) * 0.1 or
            metrics['position_jumps'] > len(trajectory.points) * 0.05):
            metrics['is_valid'] = False
        
        return metrics
    
    def _calculate_trajectory_smoothness(self, trajectory: Trajectory) -> float:
        """Calculate trajectory smoothness score.
        
        Args:
            trajectory: Trajectory to analyze
            
        Returns:
            Smoothness score (0-1, higher is smoother)
        """
        if len(trajectory.points) < 3:
            return 0.0
        
        # Calculate curvature changes
        curvatures = []
        for i in range(1, len(trajectory.points) - 1):
            p1 = trajectory.points[i-1]
            p2 = trajectory.points[i]
            p3 = trajectory.points[i+1]
            
            # Calculate curvature using three points
            dx1 = p2.x - p1.x
            dy1 = p2.y - p1.y
            dx2 = p3.x - p2.x
            dy2 = p3.y - p2.y
            
            # Cross product for curvature
            cross_product = dx1 * dy2 - dx2 * dy1
            
            # Distance factors
            dist1 = np.sqrt(dx1**2 + dy1**2)
            dist2 = np.sqrt(dx2**2 + dy2**2)
            
            if dist1 > 0 and dist2 > 0:
                curvature = abs(cross_product) / (dist1 * dist2)
                curvatures.append(curvature)
        
        if not curvatures:
            return 1.0
        
        # Smoothness is inverse of average curvature
        avg_curvature = np.mean(curvatures)
        smoothness = 1.0 / (1.0 + avg_curvature)
        
        return min(1.0, max(0.0, smoothness))
    
    def _calculate_quality_scores(self, metrics: QualityMetrics) -> Dict[str, float]:
        """Calculate quality scores for different aspects.
        
        Args:
            metrics: Quality metrics
            
        Returns:
            Dictionary of quality scores
        """
        scores = {}
        
        # Completeness score
        scores['completeness'] = metrics.completeness_rate
        
        # Consistency score
        total_records = metrics.total_trajectories
        consistency_issues = (metrics.duplicate_records + 
                            metrics.inconsistent_timestamps + 
                            metrics.invalid_vehicle_ids)
        scores['consistency'] = max(0.0, 1.0 - consistency_issues / total_records) if total_records > 0 else 0.0
        
        # Smoothness score
        scores['smoothness'] = metrics.average_smoothness
        
        # Physics score
        total_violations = (metrics.velocity_violations + 
                          metrics.acceleration_violations + 
                          metrics.jerk_violations + 
                          metrics.position_jumps)
        total_points = sum(len(t.points) for t in [])  # Would need dataset access
        scores['physics'] = max(0.0, 1.0 - total_violations / total_points) if total_points > 0 else 0.0
        
        return scores
    
    def clean_trajectory_dataset(self, dataset: TrajectoryDataset) -> TrajectoryDataset:
        """Clean trajectory dataset by removing invalid trajectories and fixing issues.
        
        Args:
            dataset: Original dataset
            
        Returns:
            Cleaned dataset
        """
        if not self.config.enable_cleaning:
            logger.info("Data cleaning disabled, returning original dataset")
            return dataset
        
        logger.info("Starting dataset cleaning", 
                   dataset_name=dataset.name,
                   num_trajectories=len(dataset.trajectories))
        
        cleaned_trajectories = []
        
        for trajectory in dataset.trajectories:
            cleaned_trajectory = self._clean_trajectory(trajectory)
            if cleaned_trajectory and len(cleaned_trajectory.points) >= self.config.min_trajectory_length:
                cleaned_trajectories.append(cleaned_trajectory)
        
        cleaned_dataset = TrajectoryDataset(
            trajectories=cleaned_trajectories,
            name=f"{dataset.name}_cleaned",
            description=f"Cleaned version of {dataset.name}",
            source=dataset.source,
            version=f"{dataset.version}_cleaned",
            metadata={
                **dataset.metadata,
                'cleaning_timestamp': datetime.now().isoformat(),
                'original_trajectories': len(dataset.trajectories),
                'cleaned_trajectories': len(cleaned_trajectories),
            }
        )
        
        logger.info("Dataset cleaning completed", 
                   original_trajectories=len(dataset.trajectories),
                   cleaned_trajectories=len(cleaned_trajectories))
        
        return cleaned_dataset
    
    def _clean_trajectory(self, trajectory: Trajectory) -> Optional[Trajectory]:
        """Clean a single trajectory.
        
        Args:
            trajectory: Trajectory to clean
            
        Returns:
            Cleaned trajectory or None if invalid
        """
        if len(trajectory.points) < 2:
            return None
        
        # Remove duplicates
        unique_points = []
        seen_timestamps = set()
        
        for point in trajectory.points:
            if point.timestamp not in seen_timestamps:
                unique_points.append(point)
                seen_timestamps.add(point.timestamp)
        
        if len(unique_points) < 2:
            return None
        
        # Sort by timestamp
        unique_points.sort(key=lambda p: p.timestamp)
        
        # Remove outliers if enabled
        if self.config.outlier_detection:
            unique_points = self._remove_outliers(unique_points)
        
        # Interpolate missing values if needed
        if self.config.enable_cleaning:
            unique_points = self._interpolate_points(unique_points)
        
        if len(unique_points) < self.config.min_trajectory_length:
            return None
        
        # Create cleaned trajectory
        cleaned_trajectory = Trajectory(
            points=unique_points,
            vehicle_id=trajectory.vehicle_id,
            metadata={
                **trajectory.metadata,
                'cleaned': True,
                'cleaning_timestamp': datetime.now().isoformat(),
            }
        )
        
        return cleaned_trajectory
    
    def _remove_outliers(self, points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """Remove outlier points from trajectory.
        
        Args:
            points: List of trajectory points
            
        Returns:
            List with outliers removed
        """
        if len(points) < 3:
            return points
        
        # Calculate velocity changes
        velocities = []
        for i in range(1, len(points)):
            p1 = points[i-1]
            p2 = points[i]
            time_diff = p2.timestamp - p1.timestamp
            
            if time_diff > 0:
                distance = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
                velocity = distance / time_diff
                velocities.append(velocity)
        
        if not velocities:
            return points
        
        # Detect outliers using z-score
        if NUMPY_AVAILABLE:
            velocities_array = np.array(velocities)
            z_scores = np.abs((velocities_array - np.mean(velocities_array)) / np.std(velocities_array))
            outlier_indices = np.where(z_scores > self.config.outlier_threshold)[0]
        else:
            # Simple outlier detection without numpy
            mean_velocity = sum(velocities) / len(velocities)
            variance = sum((v - mean_velocity) ** 2 for v in velocities) / len(velocities)
            std_velocity = variance ** 0.5
            outlier_indices = [i for i, v in enumerate(velocities) if abs(v - mean_velocity) > self.config.outlier_threshold * std_velocity]
        
        # Remove outlier points (add 1 because velocities are between points)
        outlier_point_indices = [i + 1 for i in outlier_indices]
        
        cleaned_points = [points[0]]  # Always keep first point
        for i in range(1, len(points)):
            if i not in outlier_point_indices:
                cleaned_points.append(points[i])
        
        return cleaned_points
    
    def _interpolate_points(self, points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """Interpolate missing points in trajectory.
        
        Args:
            points: List of trajectory points
            
        Returns:
            List with interpolated points
        """
        if len(points) < 2:
            return points
        
        interpolated_points = []
        
        for i in range(len(points) - 1):
            current_point = points[i]
            next_point = points[i + 1]
            
            interpolated_points.append(current_point)
            
            # Check if interpolation is needed
            time_diff = next_point.timestamp - current_point.timestamp
            
            if time_diff > self.config.min_time_gap * 2:
                # Interpolate intermediate points
                num_intermediate = int(time_diff / self.config.min_time_gap) - 1
                
                for j in range(1, num_intermediate + 1):
                    alpha = j / (num_intermediate + 1)
                    
                    # Linear interpolation
                    interpolated_point = TrajectoryPoint(
                        x=current_point.x + alpha * (next_point.x - current_point.x),
                        y=current_point.y + alpha * (next_point.y - current_point.y),
                        timestamp=current_point.timestamp + alpha * time_diff,
                        velocity=current_point.velocity + alpha * (next_point.velocity - current_point.velocity),
                        acceleration=current_point.acceleration + alpha * (next_point.acceleration - current_point.acceleration),
                        heading=current_point.heading + alpha * (next_point.heading - current_point.heading),
                        vehicle_id=current_point.vehicle_id,
                        lane_id=current_point.lane_id,
                        attributes=current_point.attributes
                    )
                    
                    interpolated_points.append(interpolated_point)
        
        # Add last point
        interpolated_points.append(points[-1])
        
        return interpolated_points
    
    def generate_quality_report(self, 
                              metrics: QualityMetrics, 
                              dataset_name: str) -> Dict[str, Any]:
        """Generate comprehensive quality report.
        
        Args:
            metrics: Quality metrics
            dataset_name: Name of the dataset
            
        Returns:
            Quality report dictionary
        """
        report = {
            'dataset_name': dataset_name,
            'validation_timestamp': metrics.validation_timestamp.isoformat(),
            'processing_time_seconds': metrics.processing_time,
            
            # Summary statistics
            'summary': {
                'total_trajectories': metrics.total_trajectories,
                'valid_trajectories': metrics.valid_trajectories,
                'completeness_rate': metrics.completeness_rate,
                'overall_quality_score': metrics.overall_quality_score,
            },
            
            # Detailed metrics
            'metrics': {
                'completeness': {
                    'missing_values': metrics.missing_values,
                },
                'consistency': {
                    'duplicate_records': metrics.duplicate_records,
                    'inconsistent_timestamps': metrics.inconsistent_timestamps,
                    'invalid_vehicle_ids': metrics.invalid_vehicle_ids,
                },
                'physics': {
                    'velocity_violations': metrics.velocity_violations,
                    'acceleration_violations': metrics.acceleration_violations,
                    'jerk_violations': metrics.jerk_violations,
                    'position_jumps': metrics.position_jumps,
                },
                'smoothness': {
                    'average_smoothness': metrics.average_smoothness,
                    'smoothness_distribution': {
                        'min': min(metrics.trajectory_smoothness) if metrics.trajectory_smoothness else 0.0,
                        'max': max(metrics.trajectory_smoothness) if metrics.trajectory_smoothness else 0.0,
                        'mean': metrics.average_smoothness,
                        'std': np.std(metrics.trajectory_smoothness) if NUMPY_AVAILABLE and metrics.trajectory_smoothness else 0.0,
                    }
                }
            },
            
            # Quality scores
            'quality_scores': metrics.quality_scores,
            
            # Recommendations
            'recommendations': self._generate_recommendations(metrics),
        }
        
        # Save report if enabled
        if self.config.save_reports:
            self._save_quality_report(report, dataset_name)
        
        return report
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate recommendations based on quality metrics.
        
        Args:
            metrics: Quality metrics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if metrics.completeness_rate < self.config.min_completeness:
            recommendations.append(
                f"Completeness rate ({metrics.completeness_rate:.2%}) is below threshold "
                f"({self.config.min_completeness:.2%}). Consider data source improvements."
            )
        
        if metrics.velocity_violations > 0:
            recommendations.append(
                f"Found {metrics.velocity_violations} velocity violations. "
                "Consider adjusting velocity thresholds or data cleaning."
            )
        
        if metrics.acceleration_violations > 0:
            recommendations.append(
                f"Found {metrics.acceleration_violations} acceleration violations. "
                "Consider adjusting acceleration thresholds or data cleaning."
            )
        
        if metrics.position_jumps > 0:
            recommendations.append(
                f"Found {metrics.position_jumps} position jumps. "
                "Consider data cleaning or sensor calibration."
            )
        
        if metrics.average_smoothness < 0.5:
            recommendations.append(
                f"Low trajectory smoothness ({metrics.average_smoothness:.2f}). "
                "Consider smoothing algorithms or data preprocessing."
            )
        
        if not recommendations:
            recommendations.append("Data quality is acceptable. No major issues detected.")
        
        return recommendations
    
    def _save_quality_report(self, report: Dict[str, Any], dataset_name: str):
        """Save quality report to file.
        
        Args:
            report: Quality report
            dataset_name: Dataset name
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if self.config.report_format == 'json':
            filename = f"{dataset_name}_quality_report_{timestamp}.json"
            filepath = self.report_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif self.config.report_format == 'html':
            filename = f"{dataset_name}_quality_report_{timestamp}.html"
            filepath = self.report_path / filename
            
            html_content = self._generate_html_report(report)
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        logger.info("Quality report saved", 
                   filepath=str(filepath),
                   format=self.config.report_format)
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML quality report.
        
        Args:
            report: Quality report dictionary
            
        Returns:
            HTML content
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quality Report - {report['dataset_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
                .score {{ font-size: 24px; font-weight: bold; color: #007bff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Report</h1>
                <p><strong>Dataset:</strong> {report['dataset_name']}</p>
                <p><strong>Validation Time:</strong> {report['validation_timestamp']}</p>
                <p><strong>Processing Time:</strong> {report['processing_time_seconds']:.2f} seconds</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="metric">
                    <strong>Total Trajectories:</strong> {report['summary']['total_trajectories']}
                </div>
                <div class="metric">
                    <strong>Valid Trajectories:</strong> {report['summary']['valid_trajectories']}
                </div>
                <div class="metric">
                    <strong>Completeness Rate:</strong> {report['summary']['completeness_rate']:.2%}
                </div>
                <div class="metric">
                    <strong>Overall Quality Score:</strong> 
                    <span class="score">{report['summary']['overall_quality_score']:.2f}</span>
                </div>
            </div>
            
            <div class="section">
                <h2>Quality Scores</h2>
                {''.join(f'<div class="metric"><strong>{k}:</strong> {v:.2f}</div>' for k, v in report['quality_scores'].items())}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {''.join(f'<div class="recommendation">{rec}</div>' for rec in report['recommendations'])}
            </div>
        </body>
        </html>
        """
        
        return html