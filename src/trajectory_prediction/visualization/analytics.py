"""Analytics and data exploration utilities for trajectory data."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..data.models import Trajectory


class TrajectoryAnalytics:
    """Analytics utilities for trajectory data exploration."""

    @staticmethod
    def compute_dataset_statistics(trajectories: list[Trajectory]) -> dict[str, float]:
        """Compute comprehensive dataset statistics."""
        if not trajectories:
            return {}

        stats: dict[str, float] = {}

        # Basic counts
        stats["total_trajectories"] = float(len(trajectories))
        stats["total_points"] = float(sum(len(traj.points) for traj in trajectories))
        stats["avg_points_per_trajectory"] = (
            stats["total_points"] / stats["total_trajectories"]
        )

        # Duration statistics
        durations = []
        for traj in trajectories:
            if len(traj.points) > 1:
                duration = max(p.timestamp for p in traj.points) - min(
                    p.timestamp for p in traj.points
                )
                durations.append(duration)

        if durations:
            stats["avg_duration"] = float(np.mean(durations))
            stats["min_duration"] = float(np.min(durations))
            stats["max_duration"] = float(np.max(durations))
            stats["std_duration"] = float(np.std(durations))

        # Distance statistics
        distances = []
        for traj in trajectories:
            if len(traj.points) > 1:
                total_distance = 0
                for i in range(1, len(traj.points)):
                    prev_pt = traj.points[i - 1]
                    curr_pt = traj.points[i]
                    distance = np.sqrt(
                        (curr_pt.x - prev_pt.x) ** 2 + (curr_pt.y - prev_pt.y) ** 2
                    )
                    total_distance += distance
                distances.append(total_distance)

        if distances:
            stats["avg_distance"] = float(np.mean(distances))
            stats["min_distance"] = float(np.min(distances))
            stats["max_distance"] = float(np.max(distances))
            stats["std_distance"] = float(np.std(distances))

        # Speed statistics
        all_speeds = []
        for traj in trajectories:
            for point in traj.points:
                all_speeds.append(point.speed)

        if all_speeds:
            stats["avg_speed"] = float(np.mean(all_speeds))
            stats["min_speed"] = float(np.min(all_speeds))
            stats["max_speed"] = float(np.max(all_speeds))
            stats["std_speed"] = float(np.std(all_speeds))

        # Vehicle type distribution
        vehicle_types = [traj.vehicle.vehicle_type for traj in trajectories]
        stats["unique_vehicle_types"] = float(len(set(vehicle_types)))

        # Quality score statistics
        quality_scores = {
            "completeness": [traj.completeness_score for traj in trajectories],
            "temporal_consistency": [
                traj.temporal_consistency_score for traj in trajectories
            ],
            "spatial_accuracy": [traj.spatial_accuracy_score for traj in trajectories],
            "smoothness": [traj.smoothness_score for traj in trajectories],
        }

        for quality_type, scores in quality_scores.items():
            # Filter out None values
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                stats[f"avg_{quality_type}"] = float(np.mean(valid_scores))
                stats[f"min_{quality_type}"] = float(np.min(valid_scores))
                stats[f"max_{quality_type}"] = float(np.max(valid_scores))

        return stats

    @staticmethod
    def create_trajectory_length_distribution(
        trajectories: list[Trajectory],
    ) -> go.Figure:
        """Create trajectory length distribution plot."""
        lengths = [len(traj.points) for traj in trajectories]

        fig = px.histogram(
            x=lengths,
            nbins=30,
            title="Trajectory Length Distribution",
            labels={"x": "Number of Points", "y": "Frequency"},
        )

        fig.add_vline(
            x=np.mean(lengths),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {np.mean(lengths):.1f}",
        )

        return fig

    @staticmethod
    def create_speed_distribution(trajectories: list[Trajectory]) -> go.Figure:
        """Create speed distribution analysis."""
        all_speeds = []
        for traj in trajectories:
            for point in traj.points:
                all_speeds.append(point.speed)

        fig = px.histogram(
            x=all_speeds,
            nbins=50,
            title="Speed Distribution",
            labels={"x": "Speed (m/s)", "y": "Frequency"},
        )

        fig.add_vline(
            x=np.mean(all_speeds),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {np.mean(all_speeds):.1f} m/s",
        )

        return fig

    @staticmethod
    def create_quality_radar_chart(trajectories: list[Trajectory]) -> go.Figure:
        """Create radar chart of data quality metrics."""

        # Calculate average quality scores
        def safe_mean(scores: list[float | None]) -> float:
            valid_scores = [s for s in scores if s is not None]
            return float(np.mean(valid_scores)) if valid_scores else 0.0

        avg_scores = {
            "Completeness": safe_mean(
                [traj.completeness_score for traj in trajectories]
            ),
            "Temporal Consistency": safe_mean(
                [traj.temporal_consistency_score for traj in trajectories]
            ),
            "Spatial Accuracy": safe_mean(
                [traj.spatial_accuracy_score for traj in trajectories]
            ),
            "Smoothness": safe_mean([traj.smoothness_score for traj in trajectories]),
        }

        categories = list(avg_scores.keys())
        values = list(avg_scores.values())

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=values, theta=categories, fill="toself", name="Average Quality Scores"
            )
        )

        fig.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            title="Data Quality Radar Chart",
        )

        return fig

    @staticmethod
    def create_vehicle_type_distribution(trajectories: list[Trajectory]) -> go.Figure:
        """Create vehicle type distribution pie chart."""
        vehicle_types = [traj.vehicle.vehicle_type.value for traj in trajectories]
        type_counts = pd.Series(vehicle_types).value_counts()

        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Vehicle Type Distribution",
        )

        return fig

    @staticmethod
    def create_trajectory_spatial_overview(trajectories: list[Trajectory]) -> go.Figure:
        """Create spatial overview of all trajectories."""
        fig = go.Figure()

        # Plot all trajectory paths
        for i, traj in enumerate(trajectories[:20]):  # Limit to first 20 for visibility
            x_coords = [p.x for p in traj.points]
            y_coords = [p.y for p in traj.points]

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    name=f"Trajectory {i + 1}",
                    line={"width": 1},
                    showlegend=False,
                )
            )

        # Add start and end points
        start_points_x = [traj.points[0].x for traj in trajectories if traj.points]
        start_points_y = [traj.points[0].y for traj in trajectories if traj.points]
        end_points_x = [traj.points[-1].x for traj in trajectories if traj.points]
        end_points_y = [traj.points[-1].y for traj in trajectories if traj.points]

        fig.add_trace(
            go.Scatter(
                x=start_points_x,
                y=start_points_y,
                mode="markers",
                name="Start Points",
                marker={"color": "green", "size": 8, "symbol": "circle"},
            )
        )

        fig.add_trace(
            go.Scatter(
                x=end_points_x,
                y=end_points_y,
                mode="markers",
                name="End Points",
                marker={"color": "red", "size": 8, "symbol": "square"},
            )
        )

        fig.update_layout(
            title="Spatial Overview of Trajectories",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            showlegend=True,
        )

        return fig

    @staticmethod
    def create_temporal_analysis(trajectories: list[Trajectory]) -> go.Figure:
        """Create temporal analysis of trajectory data."""
        # Extract temporal patterns
        hourly_counts: dict[int, int] = {}
        for traj in trajectories:
            if traj.points:
                # Use first timestamp to determine hour
                import datetime

                dt = datetime.datetime.fromtimestamp(traj.points[0].timestamp)
                hour = dt.hour
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

        hours = list(range(24))
        counts = [hourly_counts.get(hour, 0) for hour in hours]

        fig = px.bar(
            x=hours,
            y=counts,
            title="Trajectory Start Time Distribution",
            labels={"x": "Hour of Day", "y": "Number of Trajectories"},
        )

        return fig

    @staticmethod
    def create_acceleration_analysis(trajectories: list[Trajectory]) -> go.Figure:
        """Create acceleration pattern analysis."""
        all_acc_x = []
        all_acc_y = []
        all_acc_magnitude = []

        for traj in trajectories:
            for point in traj.points:
                all_acc_x.append(point.acceleration_x)
                all_acc_y.append(point.acceleration_y)
                acc_mag = np.sqrt(point.acceleration_x**2 + point.acceleration_y**2)
                all_acc_magnitude.append(acc_mag)

        fig = go.Figure()

        # Acceleration magnitude distribution
        fig.add_trace(
            go.Histogram(
                x=all_acc_magnitude,
                nbinsx=50,
                name="Acceleration Magnitude",
                opacity=0.7,
            )
        )

        fig.update_layout(
            title="Acceleration Magnitude Distribution",
            xaxis_title="Acceleration Magnitude (m/s²)",
            yaxis_title="Frequency",
        )

        return fig

    @staticmethod
    def create_lane_change_analysis(trajectories: list[Trajectory]) -> dict[str, float]:
        """Analyze lane change patterns."""
        lane_change_stats: dict[str, float] = {
            "total_lane_changes": 0.0,
            "trajectories_with_lane_changes": 0.0,
            "avg_lane_changes_per_trajectory": 0.0,
        }

        trajectories_with_changes = 0
        total_changes = 0

        for traj in trajectories:
            traj_changes = 0
            prev_lane = None

            for point in traj.points:
                if point.lane_id is not None:
                    if prev_lane is not None and point.lane_id != prev_lane:
                        traj_changes += 1
                    prev_lane = point.lane_id

            if traj_changes > 0:
                trajectories_with_changes += 1
            total_changes += traj_changes

        lane_change_stats["total_lane_changes"] = float(total_changes)
        lane_change_stats["trajectories_with_lane_changes"] = float(
            trajectories_with_changes
        )

        if len(trajectories) > 0:
            lane_change_stats["avg_lane_changes_per_trajectory"] = total_changes / len(
                trajectories
            )

        return lane_change_stats

    @staticmethod
    def create_correlation_matrix(trajectories: list[Trajectory]) -> go.Figure:
        """Create correlation matrix of trajectory features."""
        # Extract features for correlation analysis
        features_data = []

        for traj in trajectories:
            if len(traj.points) > 1:
                # Calculate trajectory-level features
                speeds = [p.speed for p in traj.points]
                acc_x = [p.acceleration_x for p in traj.points]
                acc_y = [p.acceleration_y for p in traj.points]

                duration = max(p.timestamp for p in traj.points) - min(
                    p.timestamp for p in traj.points
                )

                total_distance = 0
                for i in range(1, len(traj.points)):
                    prev_pt = traj.points[i - 1]
                    curr_pt = traj.points[i]
                    distance = np.sqrt(
                        (curr_pt.x - prev_pt.x) ** 2 + (curr_pt.y - prev_pt.y) ** 2
                    )
                    total_distance += distance

                features_data.append(
                    {
                        "avg_speed": np.mean(speeds),
                        "max_speed": np.max(speeds),
                        "std_speed": np.std(speeds),
                        "avg_acc_x": np.mean(acc_x),
                        "avg_acc_y": np.mean(acc_y),
                        "duration": duration,
                        "total_distance": total_distance,
                        "completeness": traj.completeness_score,
                        "smoothness": traj.smoothness_score,
                    }
                )

        if not features_data:
            return go.Figure()

        df = pd.DataFrame(features_data)
        corr_matrix = df.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmid=0,
            )
        )

        fig.update_layout(title="Feature Correlation Matrix", width=600, height=600)

        return fig

    @staticmethod
    def detect_anomalies(trajectories: list[Trajectory]) -> list[dict[str, Any]]:
        """Detect potential anomalies in trajectory data."""
        anomalies = []

        for i, traj in enumerate(trajectories):
            anomaly_flags = []

            # Check for speed anomalies
            speeds = [p.speed for p in traj.points]
            if speeds:
                max_speed = max(speeds)
                if max_speed > 50:  # > 180 km/h
                    anomaly_flags.append(f"High speed: {max_speed:.1f} m/s")

            # Check for acceleration anomalies
            accelerations = [
                np.sqrt(p.acceleration_x**2 + p.acceleration_y**2) for p in traj.points
            ]
            if accelerations:
                max_acc = max(accelerations)
                if max_acc > 10:  # > 10 m/s²
                    anomaly_flags.append(f"High acceleration: {max_acc:.1f} m/s²")

            # Check for data quality issues
            if traj.completeness_score is not None and traj.completeness_score < 0.5:
                anomaly_flags.append(f"Low completeness: {traj.completeness_score:.2f}")

            if traj.smoothness_score is not None and traj.smoothness_score < 0.5:
                anomaly_flags.append(f"Low smoothness: {traj.smoothness_score:.2f}")

            # Check for trajectory length anomalies
            if len(traj.points) < 10:
                anomaly_flags.append(
                    f"Very short trajectory: {len(traj.points)} points"
                )

            if anomaly_flags:
                anomalies.append(
                    {
                        "trajectory_id": traj.trajectory_id,
                        "trajectory_index": i,
                        "anomalies": anomaly_flags,
                    }
                )

        return anomalies
