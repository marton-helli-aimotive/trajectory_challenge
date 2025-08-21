"""Comprehensive validation framework for trajectory data quality assurance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .models import Dataset, Trajectory, TrajectoryPoint


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Individual validation issue found in trajectory data."""

    severity: ValidationSeverity
    code: str
    message: str
    trajectory_id: str | None = None
    point_index: int | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ValidationResult:
    """Results of trajectory validation with quality scores and issues."""

    is_valid: bool
    quality_score: float
    issues: list[ValidationIssue]
    metrics: dict[str, float]

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(
            1 for issue in self.issues if issue.severity == ValidationSeverity.ERROR
        )

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(
            1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING
        )


class TrajectoryValidator:
    """Comprehensive trajectory validation with physics constraints."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize validator with configuration."""
        default_config = self._default_config()
        if config:
            default_config.update(config)
        self.config = default_config

    def _default_config(self) -> dict[str, Any]:
        """Default validation configuration."""
        return {
            "max_speed": 200.0,  # m/s (720 km/h)
            "max_acceleration": 50.0,  # m/s²
            "max_jerk": 20.0,  # m/s³
            "min_trajectory_length": 2,
            "max_time_gap": 5.0,  # seconds
            "min_time_gap": 0.001,  # seconds
            "coordinate_bounds": {
                "x_min": -1000000,
                "x_max": 1000000,
                "y_min": -1000000,
                "y_max": 1000000,
            },
            "physics_tolerance": 0.1,  # tolerance for physics constraint violations
            "completeness_threshold": 0.7,
            "temporal_consistency_threshold": 0.8,
            "spatial_accuracy_threshold": 0.9,
            "smoothness_threshold": 0.7,
        }

    def validate_trajectory_point(
        self, point: TrajectoryPoint, trajectory_id: str = ""
    ) -> list[ValidationIssue]:
        """Validate individual trajectory point."""
        issues = []

        # Coordinate bounds check
        bounds = self.config["coordinate_bounds"]
        if not (bounds["x_min"] <= point.x <= bounds["x_max"]):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="COORD_OUT_OF_BOUNDS",
                    message=f"X coordinate {point.x} outside valid bounds",
                    trajectory_id=trajectory_id,
                    metadata={
                        "value": point.x,
                        "bounds": [bounds["x_min"], bounds["x_max"]],
                    },
                )
            )

        if not (bounds["y_min"] <= point.y <= bounds["y_max"]):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="COORD_OUT_OF_BOUNDS",
                    message=f"Y coordinate {point.y} outside valid bounds",
                    trajectory_id=trajectory_id,
                    metadata={
                        "value": point.y,
                        "bounds": [bounds["y_min"], bounds["y_max"]],
                    },
                )
            )

        # Speed validation
        if point.speed > self.config["max_speed"]:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="SPEED_TOO_HIGH",
                    message=f"Speed {point.speed:.2f} m/s exceeds maximum {self.config['max_speed']} m/s",
                    trajectory_id=trajectory_id,
                    metadata={
                        "speed": point.speed,
                        "max_speed": self.config["max_speed"],
                    },
                )
            )

        # Velocity consistency check
        calculated_speed = np.sqrt(point.velocity_x**2 + point.velocity_y**2)
        speed_diff = abs(calculated_speed - point.speed)
        if speed_diff > self.config["physics_tolerance"]:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="SPEED_VELOCITY_MISMATCH",
                    message=f"Speed {point.speed:.2f} doesn't match velocity components {calculated_speed:.2f}",
                    trajectory_id=trajectory_id,
                    metadata={
                        "reported_speed": point.speed,
                        "calculated_speed": calculated_speed,
                    },
                )
            )

        # Acceleration validation
        acceleration_magnitude = np.sqrt(
            point.acceleration_x**2 + point.acceleration_y**2
        )
        if acceleration_magnitude > self.config["max_acceleration"]:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="ACCELERATION_TOO_HIGH",
                    message=f"Acceleration {acceleration_magnitude:.2f} m/s² exceeds typical maximum",
                    trajectory_id=trajectory_id,
                    metadata={"acceleration": acceleration_magnitude},
                )
            )

        return issues

    def validate_trajectory_physics(
        self, trajectory: Trajectory
    ) -> list[ValidationIssue]:
        """Validate physics constraints across trajectory points."""
        issues: list[ValidationIssue] = []
        points = trajectory.points

        if len(points) < 2:
            return issues

        for i in range(1, len(points)):
            prev_point = points[i - 1]
            curr_point = points[i]
            dt = curr_point.timestamp - prev_point.timestamp

            if dt <= 0:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="NON_MONOTONIC_TIME",
                        message=f"Non-monotonic timestamp at point {i}",
                        trajectory_id=trajectory.trajectory_id,
                        point_index=i,
                    )
                )
                continue

            # Check time gap
            if dt > self.config["max_time_gap"]:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="LARGE_TIME_GAP",
                        message=f"Large time gap {dt:.2f}s between points {i - 1} and {i}",
                        trajectory_id=trajectory.trajectory_id,
                        point_index=i,
                        metadata={"time_gap": dt},
                    )
                )

            if dt < self.config["min_time_gap"]:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="SMALL_TIME_GAP",
                        message=f"Very small time gap {dt:.4f}s between points {i - 1} and {i}",
                        trajectory_id=trajectory.trajectory_id,
                        point_index=i,
                        metadata={"time_gap": dt},
                    )
                )

            # Calculate implied kinematics
            dx = curr_point.x - prev_point.x
            dy = curr_point.y - prev_point.y
            distance = np.sqrt(dx**2 + dy**2)
            implied_speed = distance / dt

            # Check for teleportation (unrealistic speed)
            if implied_speed > self.config["max_speed"]:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="TELEPORTATION",
                        message=f"Implied speed {implied_speed:.2f} m/s between points suggests teleportation",
                        trajectory_id=trajectory.trajectory_id,
                        point_index=i,
                        metadata={
                            "implied_speed": implied_speed,
                            "distance": distance,
                            "time_gap": dt,
                        },
                    )
                )

            # Validate acceleration consistency
            dvx = curr_point.velocity_x - prev_point.velocity_x
            dvy = curr_point.velocity_y - prev_point.velocity_y
            implied_acceleration = np.sqrt(dvx**2 + dvy**2) / dt

            if implied_acceleration > self.config["max_acceleration"]:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="HIGH_ACCELERATION",
                        message=f"High implied acceleration {implied_acceleration:.2f} m/s²",
                        trajectory_id=trajectory.trajectory_id,
                        point_index=i,
                        metadata={"implied_acceleration": implied_acceleration},
                    )
                )

        # Check for jerk (acceleration changes)
        if len(points) >= 3:
            for i in range(2, len(points)):
                dt1 = points[i - 1].timestamp - points[i - 2].timestamp
                dt2 = points[i].timestamp - points[i - 1].timestamp

                if dt1 > 0 and dt2 > 0:
                    # Calculate jerk
                    dax = (
                        points[i].acceleration_x - points[i - 1].acceleration_x
                    ) / dt2
                    day = (
                        points[i].acceleration_y - points[i - 1].acceleration_y
                    ) / dt2
                    jerk_magnitude = np.sqrt(dax**2 + day**2)

                    if jerk_magnitude > self.config["max_jerk"]:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.INFO,
                                code="HIGH_JERK",
                                message=f"High jerk {jerk_magnitude:.2f} m/s³ at point {i}",
                                trajectory_id=trajectory.trajectory_id,
                                point_index=i,
                                metadata={"jerk": jerk_magnitude},
                            )
                        )

        return issues

    def validate_trajectory(self, trajectory: Trajectory) -> ValidationResult:
        """Comprehensive trajectory validation."""
        issues = []

        # Basic structure validation
        if len(trajectory.points) < self.config["min_trajectory_length"]:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INSUFFICIENT_POINTS",
                    message=f"Trajectory has only {len(trajectory.points)} points, minimum is {self.config['min_trajectory_length']}",
                    trajectory_id=trajectory.trajectory_id,
                )
            )

        # Validate individual points
        for i, point in enumerate(trajectory.points):
            point_issues = self.validate_trajectory_point(
                point, trajectory.trajectory_id
            )
            for issue in point_issues:
                issue.point_index = i
            issues.extend(point_issues)

        # Validate physics constraints
        physics_issues = self.validate_trajectory_physics(trajectory)
        issues.extend(physics_issues)

        # Compute quality metrics if not present
        if trajectory.completeness_score is None:
            trajectory.compute_quality_metrics()

        # Check quality thresholds
        if (
            trajectory.completeness_score
            and trajectory.completeness_score < self.config["completeness_threshold"]
        ):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="LOW_COMPLETENESS",
                    message=f"Completeness score {trajectory.completeness_score:.2f} below threshold {self.config['completeness_threshold']}",
                    trajectory_id=trajectory.trajectory_id,
                    metadata={"score": trajectory.completeness_score},
                )
            )

        if (
            trajectory.temporal_consistency_score
            and trajectory.temporal_consistency_score
            < self.config["temporal_consistency_threshold"]
        ):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="LOW_TEMPORAL_CONSISTENCY",
                    message=f"Temporal consistency score {trajectory.temporal_consistency_score:.2f} below threshold",
                    trajectory_id=trajectory.trajectory_id,
                    metadata={"score": trajectory.temporal_consistency_score},
                )
            )

        if (
            trajectory.spatial_accuracy_score
            and trajectory.spatial_accuracy_score
            < self.config["spatial_accuracy_threshold"]
        ):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="LOW_SPATIAL_ACCURACY",
                    message=f"Spatial accuracy score {trajectory.spatial_accuracy_score:.2f} below threshold",
                    trajectory_id=trajectory.trajectory_id,
                    metadata={"score": trajectory.spatial_accuracy_score},
                )
            )

        if (
            trajectory.smoothness_score
            and trajectory.smoothness_score < self.config["smoothness_threshold"]
        ):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    code="LOW_SMOOTHNESS",
                    message=f"Smoothness score {trajectory.smoothness_score:.2f} below threshold",
                    trajectory_id=trajectory.trajectory_id,
                    metadata={"score": trajectory.smoothness_score},
                )
            )

        # Determine overall validation result
        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        # Calculate overall quality score
        quality_components = []
        if trajectory.completeness_score is not None:
            quality_components.append(trajectory.completeness_score)
        if trajectory.temporal_consistency_score is not None:
            quality_components.append(trajectory.temporal_consistency_score)
        if trajectory.spatial_accuracy_score is not None:
            quality_components.append(trajectory.spatial_accuracy_score)
        if trajectory.smoothness_score is not None:
            quality_components.append(trajectory.smoothness_score)

        quality_score = np.mean(quality_components) if quality_components else 0.0

        # Penalty for errors and warnings
        error_penalty = (
            len([i for i in issues if i.severity == ValidationSeverity.ERROR]) * 0.2
        )
        warning_penalty = (
            len([i for i in issues if i.severity == ValidationSeverity.WARNING]) * 0.1
        )
        quality_score = max(0.0, quality_score - error_penalty - warning_penalty)

        metrics = {
            "completeness_score": trajectory.completeness_score or 0.0,
            "temporal_consistency_score": trajectory.temporal_consistency_score or 0.0,
            "spatial_accuracy_score": trajectory.spatial_accuracy_score or 0.0,
            "smoothness_score": trajectory.smoothness_score or 0.0,
            "duration": trajectory.duration,
            "length": trajectory.length,
            "num_points": len(trajectory.points),
        }

        return ValidationResult(
            is_valid=not has_errors,
            quality_score=float(quality_score),
            issues=issues,
            metrics=metrics,
        )

    def validate_dataset(self, dataset: Dataset) -> dict[str, ValidationResult]:
        """Validate entire dataset and return results for each trajectory."""
        results = {}
        for trajectory in dataset.trajectories:
            results[trajectory.trajectory_id] = self.validate_trajectory(trajectory)
        return results


class DataQualityAnalyzer:
    """Advanced data quality analysis for trajectory datasets."""

    def __init__(self) -> None:
        """Initialize analyzer."""
        pass

    def analyze_dataset_quality(self, dataset: Dataset) -> dict[str, Any]:
        """Comprehensive dataset quality analysis."""
        validator = TrajectoryValidator()
        validation_results = validator.validate_dataset(dataset)

        # Aggregate statistics
        total_trajectories = len(dataset.trajectories)
        valid_trajectories = sum(
            1 for result in validation_results.values() if result.is_valid
        )

        all_quality_scores = [
            result.quality_score for result in validation_results.values()
        ]
        all_issues = []
        for result in validation_results.values():
            all_issues.extend(result.issues)

        # Issue statistics
        issue_counts: dict[str, int] = {}
        for issue in all_issues:
            key = f"{issue.severity.value}_{issue.code}"
            issue_counts[key] = issue_counts.get(key, 0) + 1

        # Quality distribution
        quality_stats = {
            "mean": np.mean(all_quality_scores) if all_quality_scores else 0.0,
            "std": np.std(all_quality_scores) if all_quality_scores else 0.0,
            "min": np.min(all_quality_scores) if all_quality_scores else 0.0,
            "max": np.max(all_quality_scores) if all_quality_scores else 0.0,
            "median": np.median(all_quality_scores) if all_quality_scores else 0.0,
        }

        return {
            "summary": {
                "total_trajectories": total_trajectories,
                "valid_trajectories": valid_trajectories,
                "validation_rate": valid_trajectories / total_trajectories
                if total_trajectories > 0
                else 0.0,
                "total_issues": len(all_issues),
            },
            "quality_statistics": quality_stats,
            "issue_breakdown": issue_counts,
            "dataset_metrics": dataset.get_quality_summary(),
            "validation_results": validation_results,
        }

    def generate_quality_report(self, dataset: Dataset) -> str:
        """Generate human-readable quality report."""
        analysis = self.analyze_dataset_quality(dataset)

        report = f"""
=== TRAJECTORY DATASET QUALITY REPORT ===

Dataset: {dataset.name} (v{dataset.version})
Total Trajectories: {analysis["summary"]["total_trajectories"]}
Valid Trajectories: {analysis["summary"]["valid_trajectories"]} ({analysis["summary"]["validation_rate"]:.1%})
Total Issues Found: {analysis["summary"]["total_issues"]}

=== QUALITY STATISTICS ===
Mean Quality Score: {analysis["quality_statistics"]["mean"]:.3f}
Quality Score Range: {analysis["quality_statistics"]["min"]:.3f} - {analysis["quality_statistics"]["max"]:.3f}
Quality Score Std Dev: {analysis["quality_statistics"]["std"]:.3f}

=== DATASET METRICS ===
Average Completeness: {analysis["dataset_metrics"]["avg_completeness"]:.3f}
Average Temporal Consistency: {analysis["dataset_metrics"]["avg_temporal_consistency"]:.3f}
Average Spatial Accuracy: {analysis["dataset_metrics"]["avg_spatial_accuracy"]:.3f}
Average Smoothness: {analysis["dataset_metrics"]["avg_smoothness"]:.3f}

=== ISSUE BREAKDOWN ===
"""
        for issue_type, count in analysis["issue_breakdown"].items():
            report += f"{issue_type}: {count}\n"

        return report.strip()
