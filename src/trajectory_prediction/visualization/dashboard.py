"""Interactive dashboard for trajectory prediction visualization and analysis."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..data.models import Trajectory, TrajectoryPoint, VehicleType
from ..evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from ..models.factory import ModelFactory


class TrajectoryDashboard:
    """Main dashboard class for trajectory prediction visualization."""

    def __init__(self) -> None:
        """Initialize the dashboard."""
        self.model_factory = ModelFactory()
        self.evaluator = ComprehensiveEvaluator()
        self.available_models = [
            "constant_velocity",
            "constant_acceleration",
            "polynomial",
            "knn",
            "gaussian_process",
            "random_forest",
        ]

    def run(self) -> None:
        """Run the main dashboard application."""
        st.set_page_config(
            page_title="Trajectory Prediction Dashboard",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🚗 Trajectory Prediction Dashboard")
        st.markdown(
            "Interactive visualization and analysis of vehicle trajectory prediction models"
        )

        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Select Dashboard Page",
            [
                "Model Comparison",
                "Trajectory Visualization",
                "Dataset Exploration",
                "Model Explainability",
                "Performance Monitoring",
                "Real-time Prediction",
            ],
        )

        # Route to appropriate page
        if page == "Model Comparison":
            self._render_model_comparison_page()
        elif page == "Trajectory Visualization":
            self._render_trajectory_visualization_page()
        elif page == "Dataset Exploration":
            self._render_dataset_exploration_page()
        elif page == "Model Explainability":
            self._render_model_explainability_page()
        elif page == "Performance Monitoring":
            self._render_performance_monitoring_page()
        elif page == "Real-time Prediction":
            self._render_realtime_prediction_page()

    def _render_model_comparison_page(self) -> None:
        """Render the model comparison page."""
        st.header("Model Comparison")
        st.markdown("Compare multiple trajectory prediction models side-by-side")

        # Model selection
        col1, col2 = st.columns(2)

        with col1:
            selected_models = st.multiselect(
                "Select Models to Compare",
                self.available_models,
                default=["constant_velocity", "polynomial", "knn"],
            )

        with col2:
            prediction_horizon = st.slider(
                "Prediction Horizon (seconds)",
                min_value=0.5,
                max_value=10.0,
                value=3.0,
                step=0.5,
            )

        if st.button("Generate Synthetic Data & Run Comparison"):
            with st.spinner("Generating data and training models..."):
                # Generate synthetic trajectories
                trajectories = self._generate_synthetic_trajectories(50)

                # Split data
                train_trajectories = trajectories[:40]
                test_trajectories = trajectories[40:]

                # Train selected models
                trained_models = {}
                training_progress = st.progress(0)

                for i, model_name in enumerate(selected_models):
                    st.text(f"Training {model_name}...")
                    try:
                        model = self.model_factory.create_model(model_name)
                        model.fit(train_trajectories)
                        trained_models[model_name] = model
                    except Exception as e:
                        st.error(f"Failed to train {model_name}: {e}")

                    training_progress.progress((i + 1) / len(selected_models))

                # Run evaluation
                if trained_models:
                    test_cases = self.evaluator.base_evaluator.prepare_test_data(
                        test_trajectories, prediction_horizon
                    )

                    comparison_results = self.evaluator.compare_models_comprehensive(
                        trained_models, test_cases, prediction_horizon
                    )

                    # Store results in session state
                    st.session_state.comparison_results = comparison_results
                    st.session_state.trained_models = trained_models
                    st.session_state.test_trajectories = test_trajectories

                    st.success("Model comparison completed!")

        # Display results if available
        if hasattr(st.session_state, "comparison_results"):
            self._display_comparison_results(st.session_state.comparison_results)

    def _render_trajectory_visualization_page(self) -> None:
        """Render the trajectory visualization page."""
        st.header("Trajectory Visualization")
        st.markdown("Interactive visualization of trajectory data and predictions")

        # Visualization options
        col1, col2, col3 = st.columns(3)

        with col1:
            viz_type = st.selectbox(
                "Visualization Type",
                [
                    "2D Trajectory Plot",
                    "3D Velocity Space",
                    "Animated Trajectory",
                    "Heatmap",
                ],
            )

        with col2:
            show_predictions = st.checkbox("Show Predictions", value=True)

        with col3:
            show_uncertainty = st.checkbox("Show Uncertainty Bands", value=False)

        # Generate sample data for visualization
        if st.button("Generate Sample Trajectories"):
            trajectories = self._generate_synthetic_trajectories(10)
            st.session_state.viz_trajectories = trajectories

        if hasattr(st.session_state, "viz_trajectories"):
            if viz_type == "2D Trajectory Plot":
                fig = self._create_2d_trajectory_plot(
                    st.session_state.viz_trajectories,
                    show_predictions,
                    show_uncertainty,
                )
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "3D Velocity Space":
                fig = self._create_3d_velocity_plot(st.session_state.viz_trajectories)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Animated Trajectory":
                fig = self._create_animated_trajectory_plot(
                    st.session_state.viz_trajectories
                )
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Heatmap":
                fig = self._create_trajectory_heatmap(st.session_state.viz_trajectories)
                st.plotly_chart(fig, use_container_width=True)

    def _render_dataset_exploration_page(self) -> None:
        """Render the dataset exploration page."""
        st.header("Dataset Exploration")
        st.markdown("Explore and analyze trajectory dataset characteristics")

        # Dataset statistics
        if st.button("Load and Analyze Dataset"):
            trajectories = self._generate_synthetic_trajectories(100)

            # Basic statistics
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Trajectories", len(trajectories))

            with col2:
                total_points = sum(len(traj.points) for traj in trajectories)
                st.metric("Total Points", total_points)

            with col3:
                avg_length = total_points / len(trajectories)
                st.metric("Avg Points/Trajectory", f"{avg_length:.1f}")

            with col4:
                vehicle_types = [traj.vehicle.vehicle_type for traj in trajectories]
                unique_types = len(set(vehicle_types))
                st.metric("Vehicle Types", unique_types)

            # Data quality visualization
            st.subheader("Data Quality Metrics")
            quality_data = []
            for traj in trajectories:
                quality_data.append(
                    {
                        "Trajectory ID": traj.trajectory_id,
                        "Completeness": traj.completeness_score,
                        "Temporal Consistency": traj.temporal_consistency_score,
                        "Spatial Accuracy": traj.spatial_accuracy_score,
                        "Smoothness": traj.smoothness_score,
                        "Length": len(traj.points),
                        "Duration": max(p.timestamp for p in traj.points)
                        - min(p.timestamp for p in traj.points),
                    }
                )

            quality_df = pd.DataFrame(quality_data)

            # Quality score distribution
            fig_quality = px.box(
                quality_df,
                y=[
                    "Completeness",
                    "Temporal Consistency",
                    "Spatial Accuracy",
                    "Smoothness",
                ],
                title="Data Quality Score Distributions",
            )
            st.plotly_chart(fig_quality, use_container_width=True)

            # Trajectory length distribution
            fig_length = px.histogram(
                quality_df, x="Length", title="Trajectory Length Distribution", nbins=20
            )
            st.plotly_chart(fig_length, use_container_width=True)

    def _render_model_explainability_page(self) -> None:
        """Render the model explainability page."""
        st.header("Model Explainability")
        st.markdown("Understand how models make predictions")

        if not hasattr(st.session_state, "trained_models"):
            st.warning("Please run model comparison first to load trained models.")
            return

        # Model selection for explainability
        model_name = st.selectbox(
            "Select Model for Analysis", list(st.session_state.trained_models.keys())
        )

        if st.button("Analyze Model"):
            model = st.session_state.trained_models[model_name]

            # Feature importance (if applicable)
            st.subheader("Feature Importance Analysis")

            if hasattr(model, "feature_importances_"):
                # For tree-based models
                feature_names = self._get_feature_names()
                importance_df = pd.DataFrame(
                    {"Feature": feature_names, "Importance": model.feature_importances_}
                ).sort_values("Importance", ascending=True)

                fig_importance = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title=f"Feature Importance - {model_name}",
                )
                st.plotly_chart(fig_importance, use_container_width=True)

            else:
                st.info(f"Feature importance not available for {model_name}")

            # Prediction breakdown for a sample trajectory
            st.subheader("Prediction Breakdown")
            if hasattr(st.session_state, "test_trajectories"):
                sample_traj = st.session_state.test_trajectories[0]

                # Show sample trajectory details
                st.write("Sample Trajectory Analysis:")
                traj_data = []
                for point in sample_traj.points[:10]:  # Show first 10 points
                    traj_data.append(
                        {
                            "Time": point.timestamp,
                            "X": point.x,
                            "Y": point.y,
                            "Speed": point.speed,
                            "Velocity X": point.velocity_x,
                            "Velocity Y": point.velocity_y,
                        }
                    )

                traj_df = pd.DataFrame(traj_data)
                st.dataframe(traj_df)

    def _render_performance_monitoring_page(self) -> None:
        """Render the performance monitoring page."""
        st.header("Performance Monitoring")
        st.markdown("Monitor model performance and evaluation metrics")

        if not hasattr(st.session_state, "comparison_results"):
            st.warning("Please run model comparison first to view performance metrics.")
            return

        results = st.session_state.comparison_results

        # Metric selection
        available_metrics = ["rmse", "ade", "fde", "minimum_distance", "lateral_error"]
        selected_metric = st.selectbox("Select Metric", available_metrics, index=0)

        # Performance comparison chart
        st.subheader("Model Performance Comparison")

        model_scores = []
        for model_name, model_results in results.get("model_results", {}).items():
            basic_metrics = model_results.get("aggregated_basic", {})
            safety_metrics = model_results.get("aggregated_safety", {})

            # Combine metrics
            all_metrics = {**basic_metrics, **safety_metrics}

            if selected_metric in all_metrics:
                model_scores.append(
                    {"Model": model_name, "Score": all_metrics[selected_metric]}
                )

        if model_scores:
            scores_df = pd.DataFrame(model_scores)
            fig_scores = px.bar(
                scores_df,
                x="Model",
                y="Score",
                title=f"{selected_metric.upper()} Comparison Across Models",
            )
            st.plotly_chart(fig_scores, use_container_width=True)

        # Detailed metrics table
        st.subheader("Detailed Performance Metrics")

        metrics_data = []
        for model_name, model_results in results.get("model_results", {}).items():
            basic_metrics = model_results.get("aggregated_basic", {})
            safety_metrics = model_results.get("aggregated_safety", {})

            row = {"Model": model_name}
            row.update(
                {
                    k: f"{v:.4f}"
                    for k, v in basic_metrics.items()
                    if isinstance(v, int | float)
                }
            )
            row.update(
                {
                    k: f"{v:.4f}"
                    for k, v in safety_metrics.items()
                    if isinstance(v, int | float)
                }
            )
            metrics_data.append(row)

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)

    def _render_realtime_prediction_page(self) -> None:
        """Render the real-time prediction page."""
        st.header("Real-time Prediction")
        st.markdown("Interactive trajectory prediction with parameter tuning")

        if not hasattr(st.session_state, "trained_models"):
            st.warning("Please run model comparison first to load trained models.")
            return

        # Model selection
        model_name = st.selectbox(
            "Select Model", list(st.session_state.trained_models.keys())
        )

        # Parameter tuning
        st.subheader("Prediction Parameters")
        col1, col2 = st.columns(2)

        with col1:
            start_x = st.slider("Initial X Position", -100.0, 100.0, 0.0)
            start_y = st.slider("Initial Y Position", -50.0, 50.0, 0.0)

        with col2:
            velocity_x = st.slider("Initial Velocity X", -30.0, 30.0, 15.0)
            velocity_y = st.slider("Initial Velocity Y", -10.0, 10.0, 0.0)

        prediction_horizon = st.slider("Prediction Horizon (s)", 0.5, 10.0, 3.0)

        if st.button("Generate Prediction"):
            # Create a simple trajectory for prediction
            trajectory = self._create_simple_trajectory(
                start_x, start_y, velocity_x, velocity_y
            )

            model = st.session_state.trained_models[model_name]

            try:
                # Make prediction
                predictions = model.predict([trajectory], prediction_horizon)

                # Visualize prediction
                fig = self._create_prediction_visualization(
                    trajectory, predictions[0], prediction_horizon
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    def _generate_synthetic_trajectories(self, n_trajectories: int) -> list[Trajectory]:
        """Generate synthetic trajectory data."""
        trajectories = []
        np.random.seed(42)

        for i in range(n_trajectories):
            # Random trajectory parameters
            start_x = np.random.uniform(0, 1000)
            start_y = np.random.uniform(0, 100)
            velocity_x = np.random.uniform(10, 30)
            velocity_y = np.random.uniform(-2, 2)
            acceleration_x = np.random.uniform(-2, 2)
            acceleration_y = np.random.uniform(-1, 1)

            # Generate trajectory points
            points = []
            duration = np.random.uniform(5, 15)
            dt = 0.1

            for t in np.arange(0, duration, dt):
                x = start_x + velocity_x * t + 0.5 * acceleration_x * t**2
                y = start_y + velocity_y * t + 0.5 * acceleration_y * t**2

                vx = velocity_x + acceleration_x * t
                vy = velocity_y + acceleration_y * t
                speed = np.sqrt(vx**2 + vy**2)

                point = TrajectoryPoint(
                    timestamp=1000000000 + t,
                    x=x,
                    y=y,
                    speed=speed,
                    velocity_x=vx,
                    velocity_y=vy,
                    acceleration_x=acceleration_x,
                    acceleration_y=acceleration_y,
                    heading=None,
                    lane_id=None,
                    frame_id=int(t / dt),
                )
                points.append(point)

            from ..data.models import Vehicle

            vehicle = Vehicle(
                vehicle_id=i, vehicle_type=VehicleType.CAR, length=4.5, width=2.0
            )

            trajectory = Trajectory(
                trajectory_id=f"traj_{i:03d}",
                vehicle=vehicle,
                points=points,
                dataset_name="synthetic",
                completeness_score=np.random.uniform(0.8, 1.0),
                temporal_consistency_score=np.random.uniform(0.8, 1.0),
                spatial_accuracy_score=np.random.uniform(0.8, 1.0),
                smoothness_score=np.random.uniform(0.8, 1.0),
            )
            trajectories.append(trajectory)

        return trajectories

    def _create_2d_trajectory_plot(
        self,
        trajectories: list[Trajectory],
        _show_predictions: bool = True,
        _show_uncertainty: bool = False,
    ) -> go.Figure:
        """Create a 2D trajectory plot."""
        fig = go.Figure()

        # Plot actual trajectories
        for i, traj in enumerate(trajectories[:5]):  # Limit to first 5 for clarity
            x_coords = [p.x for p in traj.points]
            y_coords = [p.y for p in traj.points]

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="lines+markers",
                    name=f"Trajectory {i + 1}",
                    line={"width": 2},
                    marker={"size": 4},
                )
            )

        fig.update_layout(
            title="2D Trajectory Visualization",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            showlegend=True,
            hovermode="closest",
        )

        return fig

    def _create_3d_velocity_plot(self, trajectories: list[Trajectory]) -> go.Figure:
        """Create a 3D velocity space plot."""
        fig = go.Figure()

        for i, traj in enumerate(trajectories[:3]):  # Limit for clarity
            x_coords = [p.x for p in traj.points]
            vx_coords = [p.velocity_x for p in traj.points]
            vy_coords = [p.velocity_y for p in traj.points]

            fig.add_trace(
                go.Scatter3d(
                    x=x_coords,
                    y=vx_coords,
                    z=vy_coords,
                    mode="markers+lines",
                    name=f"Trajectory {i + 1}",
                    marker={"size": 3},
                )
            )

        fig.update_layout(
            title="3D Position-Velocity Space",
            scene={
                "xaxis_title": "X Position (m)",
                "yaxis_title": "Velocity X (m/s)",
                "zaxis_title": "Velocity Y (m/s)",
            },
        )

        return fig

    def _create_animated_trajectory_plot(
        self, trajectories: list[Trajectory]
    ) -> go.Figure:
        """Create an animated trajectory plot."""
        # For simplicity, create a static plot with time color coding
        fig = go.Figure()

        traj = trajectories[0]  # Use first trajectory
        x_coords = [p.x for p in traj.points]
        y_coords = [p.y for p in traj.points]
        timestamps = [p.timestamp for p in traj.points]

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers+lines",
                marker={
                    "size": 8,
                    "color": timestamps,
                    "colorscale": "Viridis",
                    "colorbar": {"title": "Time"},
                    "showscale": True,
                },
                line={"width": 2},
                name="Trajectory Path",
            )
        )

        fig.update_layout(
            title="Time-Colored Trajectory",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
        )

        return fig

    def _create_trajectory_heatmap(self, trajectories: list[Trajectory]) -> go.Figure:
        """Create a trajectory density heatmap."""
        # Collect all points
        all_x = []
        all_y = []

        for traj in trajectories:
            all_x.extend([p.x for p in traj.points])
            all_y.extend([p.y for p in traj.points])

        # Create 2D histogram
        fig = go.Figure(
            data=go.Histogram2d(
                x=all_x, y=all_y, nbinsx=50, nbinsy=30, colorscale="Hot"
            )
        )

        fig.update_layout(
            title="Trajectory Density Heatmap",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
        )

        return fig

    def _create_simple_trajectory(
        self, start_x: float, start_y: float, velocity_x: float, velocity_y: float
    ) -> Trajectory:
        """Create a simple trajectory for real-time prediction."""
        points = []
        dt = 0.1
        duration = 2.0  # 2 seconds of history

        for t in np.arange(0, duration, dt):
            x = start_x + velocity_x * t
            y = start_y + velocity_y * t
            speed = np.sqrt(velocity_x**2 + velocity_y**2)

            point = TrajectoryPoint(
                timestamp=1000000000 + t,
                x=x,
                y=y,
                speed=speed,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                acceleration_x=0.0,
                acceleration_y=0.0,
                heading=None,
                lane_id=None,
                frame_id=int(t / dt),
            )
            points.append(point)

        from ..data.models import Vehicle

        vehicle = Vehicle(
            vehicle_id=999, vehicle_type=VehicleType.CAR, length=4.5, width=2.0
        )

        return Trajectory(
            trajectory_id="demo_trajectory",
            vehicle=vehicle,
            points=points,
            dataset_name="demo",
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )

    def _create_prediction_visualization(
        self, trajectory: Trajectory, prediction: Trajectory, horizon: float
    ) -> go.Figure:
        """Create a visualization comparing actual trajectory with prediction."""
        fig = go.Figure()

        # Plot historical trajectory
        hist_x = [p.x for p in trajectory.points]
        hist_y = [p.y for p in trajectory.points]

        fig.add_trace(
            go.Scatter(
                x=hist_x,
                y=hist_y,
                mode="lines+markers",
                name="Historical Path",
                line={"color": "blue", "width": 3},
                marker={"size": 6},
            )
        )

        # Plot prediction
        pred_x = [p.x for p in prediction.points]
        pred_y = [p.y for p in prediction.points]

        fig.add_trace(
            go.Scatter(
                x=pred_x,
                y=pred_y,
                mode="lines+markers",
                name="Predicted Path",
                line={"color": "red", "width": 3, "dash": "dash"},
                marker={"size": 6, "symbol": "diamond"},
            )
        )

        fig.update_layout(
            title=f"Real-time Trajectory Prediction ({horizon}s horizon)",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            showlegend=True,
        )

        return fig

    def _display_comparison_results(self, results: dict[str, Any]) -> None:
        """Display model comparison results."""
        st.subheader("Model Comparison Results")

        # Summary statistics
        summary = results.get("comparison_summary", {})
        best_models = summary.get("best_models", {})

        if best_models:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Best Overall Model", best_models.get("overall", "N/A"))
            with col2:
                st.metric("Best Safety Model", best_models.get("safety", "N/A"))
            with col3:
                st.metric("Best Accuracy Model", best_models.get("accuracy", "N/A"))

        # Model rankings
        rankings = summary.get("model_rankings", {})
        if rankings:
            st.subheader("Model Rankings")
            ranking_df = pd.DataFrame(
                [
                    {"Rank": i + 1, "Model": model, "Score": score}
                    for i, (model, score) in enumerate(rankings.items())
                ]
            )
            st.dataframe(ranking_df, use_container_width=True)

    def _get_feature_names(self) -> list[str]:
        """Get feature names for model explainability."""
        return [
            "position_x",
            "position_y",
            "velocity_x",
            "velocity_y",
            "acceleration_x",
            "acceleration_y",
            "speed",
            "heading",
            "curvature",
            "jerk_x",
            "jerk_y",
        ]


def main() -> None:
    """Main entry point for the dashboard."""
    dashboard = TrajectoryDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
