"""Advanced plotting utilities for trajectory visualization."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data.models import Trajectory


class TrajectoryPlotter:
    """Advanced plotting utilities for trajectory data."""

    @staticmethod
    def create_multi_model_comparison(
        trajectories: list[Trajectory],
        predictions: dict[str, list[Trajectory]],
        model_names: list[str],
    ) -> go.Figure:
        """Create side-by-side comparison of multiple model predictions."""
        n_models = len(model_names)
        fig = make_subplots(
            rows=1, cols=n_models, subplot_titles=model_names, shared_yaxes=True
        )

        colors = px.colors.qualitative.Set3[:n_models]

        for col, (model_name, color) in enumerate(zip(model_names, colors), 1):
            # Plot actual trajectory
            if trajectories:
                traj = trajectories[0]  # Use first trajectory
                actual_x = [p.x for p in traj.points]
                actual_y = [p.y for p in traj.points]

                fig.add_trace(
                    go.Scatter(
                        x=actual_x,
                        y=actual_y,
                        mode="lines",
                        name="Actual",
                        line={"color": "black", "width": 3},
                        showlegend=(col == 1),
                    ),
                    row=1,
                    col=col,
                )

            # Plot prediction
            if model_name in predictions and predictions[model_name]:
                pred = predictions[model_name][0]
                pred_x = [p.x for p in pred.points]
                pred_y = [p.y for p in pred.points]

                fig.add_trace(
                    go.Scatter(
                        x=pred_x,
                        y=pred_y,
                        mode="lines",
                        name=f"{model_name} Prediction",
                        line={"color": color, "width": 2, "dash": "dash"},
                        showlegend=(col == 1),
                    ),
                    row=1,
                    col=col,
                )

        fig.update_layout(title="Multi-Model Prediction Comparison", height=400)

        return fig

    @staticmethod
    def create_uncertainty_bands(
        trajectory: Trajectory,
        predictions: list[Trajectory],
        confidence_level: float = 0.95,
    ) -> go.Figure:
        """Create trajectory plot with uncertainty bands."""
        fig = go.Figure()

        # Plot actual trajectory
        actual_x = [p.x for p in trajectory.points]
        actual_y = [p.y for p in trajectory.points]

        fig.add_trace(
            go.Scatter(
                x=actual_x,
                y=actual_y,
                mode="lines+markers",
                name="Actual Trajectory",
                line={"color": "black", "width": 3},
                marker={"size": 6},
            )
        )

        if predictions:
            # Calculate prediction statistics
            pred_arrays = []
            for pred in predictions:
                pred_x = [p.x for p in pred.points]
                pred_y = [p.y for p in pred.points]
                pred_arrays.append(list(zip(pred_x, pred_y)))

            # Assuming all predictions have same length
            if pred_arrays:
                min_len = min(len(arr) for arr in pred_arrays)

                mean_x, mean_y = [], []
                upper_x, upper_y = [], []
                lower_x, lower_y = [], []

                alpha = (1 - confidence_level) / 2

                for i in range(min_len):
                    x_vals = [arr[i][0] for arr in pred_arrays]
                    y_vals = [arr[i][1] for arr in pred_arrays]

                    mean_x.append(np.mean(x_vals))
                    mean_y.append(np.mean(y_vals))

                    upper_x.append(np.percentile(x_vals, (1 - alpha) * 100))
                    upper_y.append(np.percentile(y_vals, (1 - alpha) * 100))

                    lower_x.append(np.percentile(x_vals, alpha * 100))
                    lower_y.append(np.percentile(y_vals, alpha * 100))

                # Plot mean prediction
                fig.add_trace(
                    go.Scatter(
                        x=mean_x,
                        y=mean_y,
                        mode="lines",
                        name="Mean Prediction",
                        line={"color": "red", "width": 2},
                    )
                )

                # Plot uncertainty bands
                fig.add_trace(
                    go.Scatter(
                        x=upper_x + lower_x[::-1],
                        y=upper_y + lower_y[::-1],
                        fill="toself",
                        fillcolor="rgba(255,0,0,0.2)",
                        line={"color": "rgba(255,255,255,0)"},
                        name=f"{confidence_level * 100:.0f}% Confidence",
                    )
                )

        fig.update_layout(
            title="Trajectory Prediction with Uncertainty Bands",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
        )

        return fig

    @staticmethod
    def create_error_analysis_plot(
        actual_trajectories: list[Trajectory], predicted_trajectories: list[Trajectory]
    ) -> go.Figure:
        """Create error analysis visualization."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Position Error Distribution",
                "Error vs Time",
                "X Error vs Y Error",
                "Speed Error Distribution",
            ],
        )

        # Calculate errors
        position_errors = []
        time_steps = []
        x_errors = []
        y_errors = []
        speed_errors = []

        for actual, predicted in zip(actual_trajectories, predicted_trajectories):
            min_len = min(len(actual.points), len(predicted.points))

            for i in range(min_len):
                actual_pt = actual.points[i]
                pred_pt = predicted.points[i]

                x_err = pred_pt.x - actual_pt.x
                y_err = pred_pt.y - actual_pt.y
                pos_err = np.sqrt(x_err**2 + y_err**2)
                speed_err = pred_pt.speed - actual_pt.speed

                position_errors.append(pos_err)
                x_errors.append(x_err)
                y_errors.append(y_err)
                speed_errors.append(speed_err)
                time_steps.append(i * 0.1)  # Assuming 0.1s time step

        # Position error distribution
        fig.add_trace(
            go.Histogram(x=position_errors, name="Position Error", nbinsx=30),
            row=1,
            col=1,
        )

        # Error vs time
        fig.add_trace(
            go.Scatter(
                x=time_steps, y=position_errors, mode="markers", name="Error vs Time"
            ),
            row=1,
            col=2,
        )

        # X vs Y error scatter
        fig.add_trace(
            go.Scatter(x=x_errors, y=y_errors, mode="markers", name="X vs Y Error"),
            row=2,
            col=1,
        )

        # Speed error distribution
        fig.add_trace(
            go.Histogram(x=speed_errors, name="Speed Error", nbinsx=30), row=2, col=2
        )

        fig.update_layout(
            title="Comprehensive Error Analysis", height=600, showlegend=False
        )

        return fig

    @staticmethod
    def create_velocity_profile_plot(trajectories: list[Trajectory]) -> go.Figure:
        """Create velocity profile visualization."""
        fig = go.Figure()

        for i, traj in enumerate(trajectories[:5]):  # Limit to 5 trajectories
            timestamps = [p.timestamp - traj.points[0].timestamp for p in traj.points]
            speeds = [p.speed for p in traj.points]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=speeds,
                    mode="lines",
                    name=f"Trajectory {i + 1}",
                    line={"width": 2},
                )
            )

        fig.update_layout(
            title="Velocity Profiles", xaxis_title="Time (s)", yaxis_title="Speed (m/s)"
        )

        return fig

    @staticmethod
    def create_acceleration_heatmap(trajectories: list[Trajectory]) -> go.Figure:
        """Create acceleration heatmap."""
        # Collect acceleration data
        all_ax = []
        all_ay = []

        for traj in trajectories:
            for point in traj.points:
                all_ax.append(point.acceleration_x)
                all_ay.append(point.acceleration_y)

        # Create 2D histogram
        fig = go.Figure(
            data=go.Histogram2d(
                x=all_ax, y=all_ay, nbinsx=30, nbinsy=30, colorscale="Viridis"
            )
        )

        fig.update_layout(
            title="Acceleration Distribution Heatmap",
            xaxis_title="Acceleration X (m/s²)",
            yaxis_title="Acceleration Y (m/s²)",
        )

        return fig

    @staticmethod
    def create_trajectory_similarity_matrix(
        trajectories: list[Trajectory],
    ) -> go.Figure:
        """Create trajectory similarity matrix visualization."""
        n_trajectories = len(trajectories)
        similarity_matrix = np.zeros((n_trajectories, n_trajectories))

        # Calculate pairwise similarities (simplified DTW approximation)
        for i in range(n_trajectories):
            for j in range(n_trajectories):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Simple similarity based on endpoint distance
                    traj_i = trajectories[i]
                    traj_j = trajectories[j]

                    if traj_i.points and traj_j.points:
                        end_i = traj_i.points[-1]
                        end_j = traj_j.points[-1]

                        distance = np.sqrt(
                            (end_i.x - end_j.x) ** 2 + (end_i.y - end_j.y) ** 2
                        )

                        # Convert distance to similarity (0-1 scale)
                        max_distance = 1000.0  # Normalize by max expected distance
                        similarity = max(0, 1 - distance / max_distance)
                        similarity_matrix[i, j] = similarity

        fig = go.Figure(
            data=go.Heatmap(z=similarity_matrix, colorscale="Viridis", showscale=True)
        )

        fig.update_layout(
            title="Trajectory Similarity Matrix",
            xaxis_title="Trajectory Index",
            yaxis_title="Trajectory Index",
        )

        return fig

    @staticmethod
    def create_feature_importance_plot(
        feature_names: list[str], importance_values: list[float], model_name: str
    ) -> go.Figure:
        """Create feature importance visualization."""
        # Sort by importance
        sorted_data = sorted(zip(feature_names, importance_values), key=lambda x: x[1])
        sorted_names, sorted_values = zip(*sorted_data)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=sorted_values,
                    y=sorted_names,
                    orientation="h",
                    marker={"color": "steelblue"},
                )
            ]
        )

        fig.update_layout(
            title=f"Feature Importance - {model_name}",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(feature_names) * 30),
        )

        return fig

    @staticmethod
    def create_prediction_confidence_plot(
        trajectory: Trajectory,
        predictions: list[Trajectory],
        confidence_scores: list[float] | None = None,
    ) -> go.Figure:
        """Create prediction with confidence visualization."""
        fig = go.Figure()

        # Plot actual trajectory
        actual_x = [p.x for p in trajectory.points]
        actual_y = [p.y for p in trajectory.points]

        fig.add_trace(
            go.Scatter(
                x=actual_x,
                y=actual_y,
                mode="lines+markers",
                name="Actual",
                line={"color": "black", "width": 3},
                marker={"size": 8},
            )
        )

        # Plot predictions with confidence coloring
        if confidence_scores is None:
            confidence_scores = [0.5] * len(predictions)  # Default confidence

        for i, (pred, confidence) in enumerate(zip(predictions, confidence_scores)):
            pred_x = [p.x for p in pred.points]
            pred_y = [p.y for p in pred.points]

            # Color based on confidence
            color_intensity = int(255 * confidence)
            color = f"rgba(255, {255 - color_intensity}, {255 - color_intensity}, 0.7)"

            fig.add_trace(
                go.Scatter(
                    x=pred_x,
                    y=pred_y,
                    mode="lines",
                    name=f"Prediction {i + 1} (conf: {confidence:.2f})",
                    line={"color": color, "width": 2},
                )
            )

        fig.update_layout(
            title="Predictions with Confidence Scores",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
        )

        return fig
