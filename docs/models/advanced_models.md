# Advanced Models Implementation Guide

This guide covers the implementation details of advanced trajectory prediction models that use machine learning techniques to capture complex motion patterns and provide sophisticated uncertainty quantification.

## Overview

Advanced models go beyond simple physics assumptions to learn complex patterns from data. They provide better accuracy in challenging scenarios at the cost of increased computational complexity and training requirements.

## Polynomial Trajectory Predictor

### Model Description

The Polynomial Trajectory Predictor fits polynomial curves to historical trajectory data and extrapolates them for future prediction. It uses Bayesian ridge regression to handle overfitting and provides principled uncertainty quantification.

**Mathematical Foundation**: Fits polynomials `x(t) = Σᵢ aᵢtⁱ` and `y(t) = Σⱼ bⱼtⁱ` with physics-informed regularization.

### Implementation Details

```python
from trajectory_prediction.models.advanced.polynomial import PolynomialTrajectoryPredictor

class PolynomialTrajectoryPredictor(TrajectoryPredictor):
    """
    Polynomial trajectory fitting with Bayesian regularization.
    
    Features:
    - Adaptive polynomial degree selection
    - Physics-informed constraints
    - Bayesian uncertainty quantification
    - Overfitting prevention through regularization
    """
    
    def __init__(
        self,
        max_degree: int = 4,
        regularization_strength: float = 1e-6,
        physics_constraints: bool = True
    ):
        super().__init__()
        self.model_name = "polynomial"
        self.max_degree = max_degree
        self.regularization_strength = regularization_strength
        self.physics_constraints = physics_constraints
        
        # Separate models for x and y coordinates
        self.x_model = None
        self.y_model = None
        self.degree = None
        self.feature_extractor = PolynomialFeatureExtractor()
    
    async def predict(
        self,
        trajectory: TrajectoryData,
        prediction_horizon: float = 5.0,
        time_step: float = 0.1,
        **kwargs
    ) -> TrajectoryData:
        """Predict trajectory using polynomial curve fitting."""
        
        # Prepare temporal data (relative to start time)
        times = np.array(trajectory.timestamps)
        start_time = times[0]
        relative_times = times - start_time
        
        positions_x = np.array([pos.x for pos in trajectory.positions])
        positions_y = np.array([pos.y for pos in trajectory.positions])
        
        # Select optimal polynomial degree
        optimal_degree = self._select_optimal_degree(
            relative_times, positions_x, positions_y
        )
        
        # Fit Bayesian ridge regression models
        x_model, y_model = self._fit_polynomial_models(
            relative_times, positions_x, positions_y, optimal_degree
        )
        
        # Generate future time points
        last_time = relative_times[-1]
        future_relative_times = np.arange(
            last_time + time_step,
            last_time + prediction_horizon + time_step,
            time_step
        )
        
        # Predict future positions with uncertainty
        x_predictions, x_uncertainty = self._predict_with_uncertainty(
            x_model, future_relative_times, optimal_degree
        )
        y_predictions, y_uncertainty = self._predict_with_uncertainty(
            y_model, future_relative_times, optimal_degree
        )
        
        # Apply physics constraints
        if self.physics_constraints:
            x_predictions, y_predictions = self._apply_physics_constraints(
                relative_times, positions_x, positions_y,
                future_relative_times, x_predictions, y_predictions
            )
        
        # Convert back to absolute timestamps
        future_timestamps = future_relative_times + start_time
        
        # Create prediction trajectory
        predicted_positions = [
            Position(x=x_pred, y=y_pred)
            for x_pred, y_pred in zip(x_predictions, y_predictions)
        ]
        
        # Compute velocities from polynomial derivatives
        predicted_velocities = self._compute_velocities(
            x_model, y_model, future_relative_times, optimal_degree
        )
        
        return TrajectoryData(
            trajectory_id=f"{trajectory.trajectory_id}_poly_pred",
            vehicle_id=trajectory.vehicle_id,
            positions=predicted_positions,
            velocities=predicted_velocities,
            timestamps=list(future_timestamps),
            metadata={
                "model": self.model_name,
                "degree": optimal_degree,
                "x_uncertainty": x_uncertainty.tolist(),
                "y_uncertainty": y_uncertainty.tolist(),
                "regularization": self.regularization_strength
            }
        )
    
    def _select_optimal_degree(self, times, x_pos, y_pos) -> int:
        """Select optimal polynomial degree using cross-validation."""
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import BayesianRidge
        from sklearn.pipeline import Pipeline
        
        best_degree = 1
        best_score = -np.inf
        
        for degree in range(1, min(self.max_degree + 1, len(times) - 1)):
            # Create polynomial pipeline
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('ridge', BayesianRidge(alpha_1=1e-6, alpha_2=1e-6))
            ])
            
            # Cross-validation score (negative MSE)
            try:
                cv_scores_x = cross_val_score(
                    pipeline, times.reshape(-1, 1), x_pos, cv=min(5, len(times)-1), 
                    scoring='neg_mean_squared_error'
                )
                cv_scores_y = cross_val_score(
                    pipeline, times.reshape(-1, 1), y_pos, cv=min(5, len(times)-1),
                    scoring='neg_mean_squared_error'
                )
                
                # Combined score
                combined_score = (cv_scores_x.mean() + cv_scores_y.mean()) / 2
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_degree = degree
                    
            except Exception:
                continue
        
        return best_degree
```

### Key Features

- **Adaptive Degree Selection**: Automatically chooses polynomial degree via cross-validation
- **Bayesian Regularization**: Prevents overfitting with principled uncertainty
- **Physics Constraints**: Enforces speed and acceleration limits
- **Curve Flexibility**: Captures complex trajectory shapes and maneuvers

### Use Cases

- **Curved Paths**: Excellent for highway curves and turning maneuvers
- **Smooth Trajectories**: Ideal for scenarios with gradual changes
- **Medium-term Prediction**: Works well for 3-8 second horizons
- **Feature-rich Scenarios**: Leverages temporal patterns in the data

---

## K-Nearest Neighbors Predictor

### Model Description

The KNN Trajectory Predictor finds similar historical trajectories using Dynamic Time Warping (DTW) distance and creates ensemble predictions from the most similar cases. This non-parametric approach captures complex, context-dependent patterns.

### Implementation Details

```python
from trajectory_prediction.models.advanced.knn import KNNTrajectoryPredictor

class KNNTrajectoryPredictor(TrajectoryPredictor):
    """
    K-Nearest Neighbors trajectory prediction with DTW similarity.
    
    Features:
    - Dynamic Time Warping for trajectory similarity
    - Multi-scale feature matching
    - Ensemble prediction from similar cases
    - Context-aware neighbor selection
    """
    
    def __init__(
        self,
        k: int = 5,
        similarity_metric: str = "dtw",
        feature_weights: Dict[str, float] = None,
        distance_threshold: float = np.inf
    ):
        super().__init__()
        self.model_name = "knn"
        self.k = k
        self.similarity_metric = similarity_metric
        self.distance_threshold = distance_threshold
        self.training_trajectories = []
        
        # Feature weights for multi-scale matching
        self.feature_weights = feature_weights or {
            "position": 1.0,
            "velocity": 0.7,
            "acceleration": 0.5,
            "curvature": 0.3
        }
        
        self.feature_extractor = TrajectoryFeatureExtractor()
        self.dtw_aligner = DTWAligner()
    
    async def train(
        self,
        trajectories: List[TrajectoryData],
        validation_data: Optional[List[TrajectoryData]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train by storing reference trajectories with features."""
        
        self.training_trajectories = []
        
        for trajectory in trajectories:
            # Extract multi-scale features
            features = self.feature_extractor.extract_features(trajectory)
            
            # Store trajectory with precomputed features
            self.training_trajectories.append({
                "trajectory": trajectory,
                "features": features,
                "position_sequence": self._extract_position_sequence(trajectory),
                "velocity_sequence": self._extract_velocity_sequence(trajectory),
                "temporal_features": self._extract_temporal_features(trajectory)
            })
        
        self.is_trained = True
        
        return {
            "num_training_trajectories": len(self.training_trajectories),
            "feature_dimensions": len(features),
            "training_status": "completed"
        }
    
    async def predict(
        self,
        trajectory: TrajectoryData,
        prediction_horizon: float = 5.0,
        time_step: float = 0.1,
        **kwargs
    ) -> TrajectoryData:
        """Predict using K most similar trajectories."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract query features
        query_features = self.feature_extractor.extract_features(trajectory)
        query_sequence = self._extract_position_sequence(trajectory)
        
        # Find K most similar trajectories
        similarities = []
        for i, ref_data in enumerate(self.training_trajectories):
            distance = self._compute_trajectory_distance(
                query_features, query_sequence,
                ref_data["features"], ref_data["position_sequence"]
            )
            similarities.append((distance, i))
        
        # Sort by similarity and select top-k
        similarities.sort(key=lambda x: x[0])
        top_k_indices = [idx for _, idx in similarities[:self.k] 
                        if similarities[0][0] < self.distance_threshold]
        
        if not top_k_indices:
            # Fallback to simple extrapolation if no similar trajectories
            return await self._fallback_prediction(trajectory, prediction_horizon, time_step)
        
        # Create ensemble prediction from similar trajectories
        ensemble_prediction = self._create_ensemble_prediction(
            trajectory, top_k_indices, prediction_horizon, time_step
        )
        
        return ensemble_prediction
    
    def _compute_trajectory_distance(
        self, query_features, query_sequence, ref_features, ref_sequence
    ) -> float:
        """Compute multi-scale trajectory distance."""
        
        # DTW distance on position sequences
        dtw_distance = self.dtw_aligner.distance(query_sequence, ref_sequence)
        
        # Feature-based distance
        feature_distance = 0.0
        for feature_name, weight in self.feature_weights.items():
            if feature_name in query_features and feature_name in ref_features:
                query_feat = query_features[feature_name]
                ref_feat = ref_features[feature_name]
                
                # Normalize and compute distance
                feat_dist = np.linalg.norm(query_feat - ref_feat)
                feature_distance += weight * feat_dist
        
        # Combined distance with adaptive weighting
        alpha = 0.6  # Weight for DTW vs features
        combined_distance = alpha * dtw_distance + (1 - alpha) * feature_distance
        
        return combined_distance
    
    def _create_ensemble_prediction(
        self, query_trajectory, top_k_indices, prediction_horizon, time_step
    ) -> TrajectoryData:
        """Create weighted ensemble prediction from similar trajectories."""
        
        predictions = []
        weights = []
        
        for idx in top_k_indices:
            ref_trajectory = self.training_trajectories[idx]["trajectory"]
            
            # Align reference trajectory with query
            aligned_prediction = self._align_and_extrapolate(
                query_trajectory, ref_trajectory, prediction_horizon, time_step
            )
            
            if aligned_prediction:
                predictions.append(aligned_prediction)
                # Weight inversely proportional to distance
                weight = 1.0 / (1.0 + self._compute_trajectory_distance(
                    query_trajectory, ref_trajectory
                ))
                weights.append(weight)
        
        if not predictions:
            return await self._fallback_prediction(query_trajectory, prediction_horizon, time_step)
        
        # Weighted ensemble of predictions
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        ensemble_positions = []
        ensemble_velocities = []
        
        # Combine predictions
        for i in range(len(predictions[0].positions)):
            weighted_x = sum(w * pred.positions[i].x 
                           for w, pred in zip(weights, predictions))
            weighted_y = sum(w * pred.positions[i].y 
                           for w, pred in zip(weights, predictions))
            weighted_vx = sum(w * pred.velocities[i].vx 
                            for w, pred in zip(weights, predictions))
            weighted_vy = sum(w * pred.velocities[i].vy 
                            for w, pred in zip(weights, predictions))
            
            ensemble_positions.append(Position(x=weighted_x, y=weighted_y))
            ensemble_velocities.append(Velocity(vx=weighted_vx, vy=weighted_vy))
        
        return TrajectoryData(
            trajectory_id=f"{query_trajectory.trajectory_id}_knn_pred",
            vehicle_id=query_trajectory.vehicle_id,
            positions=ensemble_positions,
            velocities=ensemble_velocities,
            timestamps=predictions[0].timestamps,
            metadata={
                "model": self.model_name,
                "k": len(predictions),
                "weights": weights.tolist(),
                "similar_trajectory_ids": [
                    self.training_trajectories[idx]["trajectory"].trajectory_id
                    for idx in top_k_indices[:len(predictions)]
                ]
            }
        )
```

### Key Features

- **Dynamic Time Warping**: Robust similarity measure for variable-length trajectories
- **Multi-scale Features**: Combines position, velocity, acceleration, and curvature
- **Ensemble Prediction**: Weighted combination of similar cases
- **Non-parametric**: No assumptions about underlying motion model

### Use Cases

- **Complex Scenarios**: Captures intricate patterns that physics models miss
- **Context-dependent Behavior**: Adapts predictions based on similar situations
- **Intersection Navigation**: Learns from similar turning patterns
- **Multi-modal Behavior**: Handles diverse trajectory types in training data

---

## Gaussian Process Predictor

### Model Description

The Gaussian Process Predictor provides principled uncertainty quantification by treating trajectory prediction as a regression problem with a prior over functions. It learns smooth trajectory patterns while providing confidence intervals.

### Implementation Details

```python
from trajectory_prediction.models.advanced.gaussian_process import GaussianProcessPredictor

class GaussianProcessPredictor(TrajectoryPredictor):
    """
    Gaussian Process trajectory prediction with principled uncertainty.
    
    Features:
    - Flexible kernel design for trajectory patterns
    - Principled uncertainty quantification
    - Non-linear relationship learning
    - Bayesian hyperparameter optimization
    """
    
    def __init__(
        self,
        kernel_type: str = "rbf_periodic",
        length_scale: float = 1.0,
        noise_level: float = 0.1,
        optimize_hyperparameters: bool = True
    ):
        super().__init__()
        self.model_name = "gaussian_process"
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.optimize_hyperparameters = optimize_hyperparameters
        
        # Separate GP models for x and y coordinates
        self.gp_x = None
        self.gp_y = None
        self.kernel = self._create_kernel()
        self.feature_extractor = TrajectoryFeatureExtractor()
    
    def _create_kernel(self):
        """Create appropriate kernel for trajectory modeling."""
        from sklearn.gaussian_process.kernels import (
            RBF, RationalQuadratic, Matern, WhiteKernel, 
            ConstantKernel, ExpSineSquared
        )
        
        if self.kernel_type == "rbf":
            kernel = ConstantKernel(1.0) * RBF(length_scale=self.length_scale)
            
        elif self.kernel_type == "rbf_periodic":
            # RBF + periodic for capturing both smooth and periodic patterns
            rbf = RBF(length_scale=self.length_scale)
            periodic = ExpSineSquared(
                length_scale=2.0, 
                periodicity=10.0,
                periodicity_bounds=(5.0, 50.0)
            )
            kernel = ConstantKernel(1.0) * (rbf + periodic)
            
        elif self.kernel_type == "matern":
            kernel = ConstantKernel(1.0) * Matern(
                length_scale=self.length_scale, nu=2.5
            )
            
        elif self.kernel_type == "rational_quadratic":
            kernel = ConstantKernel(1.0) * RationalQuadratic(
                length_scale=self.length_scale, alpha=0.1
            )
            
        else:
            # Default to RBF
            kernel = ConstantKernel(1.0) * RBF(length_scale=self.length_scale)
        
        # Add noise kernel
        kernel += WhiteKernel(noise_level=self.noise_level)
        
        return kernel
    
    async def train(
        self,
        trajectories: List[TrajectoryData],
        validation_data: Optional[List[TrajectoryData]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train GP models on trajectory data."""
        
        from sklearn.gaussian_process import GaussianProcessRegressor
        
        # Prepare training data
        X_train, y_x_train, y_y_train = self._prepare_training_data(trajectories)
        
        if len(X_train) == 0:
            raise ValueError("No valid training data available")
        
        # Create and train GP models
        self.gp_x = GaussianProcessRegressor(
            kernel=self.kernel.clone_with_theta(self.kernel.theta),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3 if self.optimize_hyperparameters else 0,
            random_state=42
        )
        
        self.gp_y = GaussianProcessRegressor(
            kernel=self.kernel.clone_with_theta(self.kernel.theta),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3 if self.optimize_hyperparameters else 0,
            random_state=42
        )
        
        # Fit models
        self.gp_x.fit(X_train, y_x_train)
        self.gp_y.fit(X_train, y_y_train)
        
        self.is_trained = True
        
        # Compute training metrics
        x_score = self.gp_x.score(X_train, y_x_train)
        y_score = self.gp_y.score(X_train, y_y_train)
        
        return {
            "training_samples": len(X_train),
            "x_model_score": x_score,
            "y_model_score": y_score,
            "x_kernel_params": dict(self.gp_x.kernel_.get_params()),
            "y_kernel_params": dict(self.gp_y.kernel_.get_params()),
            "training_status": "completed"
        }
    
    async def predict(
        self,
        trajectory: TrajectoryData,
        prediction_horizon: float = 5.0,
        time_step: float = 0.1,
        **kwargs
    ) -> TrajectoryData:
        """Predict trajectory with uncertainty quantification."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare query data
        query_features = self._prepare_query_features(trajectory)
        
        # Generate future time points
        last_timestamp = trajectory.timestamps[-1]
        future_times = np.arange(
            last_timestamp + time_step,
            last_timestamp + prediction_horizon + time_step,
            time_step
        )
        
        # Create feature matrix for future predictions
        X_pred = self._create_prediction_features(trajectory, future_times)
        
        # GP predictions with uncertainty
        x_pred_mean, x_pred_std = self.gp_x.predict(X_pred, return_std=True)
        y_pred_mean, y_pred_std = self.gp_y.predict(X_pred, return_std=True)
        
        # Create prediction trajectory
        predicted_positions = [
            Position(x=x_mean, y=y_mean)
            for x_mean, y_mean in zip(x_pred_mean, y_pred_mean)
        ]
        
        # Compute velocities from GP derivatives (approximate)
        predicted_velocities = self._compute_velocity_from_predictions(
            x_pred_mean, y_pred_mean, future_times
        )
        
        return TrajectoryData(
            trajectory_id=f"{trajectory.trajectory_id}_gp_pred",
            vehicle_id=trajectory.vehicle_id,
            positions=predicted_positions,
            velocities=predicted_velocities,
            timestamps=list(future_times),
            metadata={
                "model": self.model_name,
                "x_uncertainty": x_pred_std.tolist(),
                "y_uncertainty": y_pred_std.tolist(),
                "kernel_type": self.kernel_type,
                "total_uncertainty": np.sqrt(x_pred_std**2 + y_pred_std**2).tolist()
            }
        )
    
    def _prepare_training_data(self, trajectories):
        """Prepare training data with features and targets."""
        X_train = []
        y_x_train = []
        y_y_train = []
        
        for trajectory in trajectories:
            # Extract features for each point in trajectory
            features = self.feature_extractor.extract_features(trajectory)
            
            # Create training examples (use sliding window)
            window_size = min(5, len(trajectory.positions) - 1)
            
            for i in range(window_size, len(trajectory.positions)):
                # Features: recent trajectory points + temporal info
                feature_vector = self._create_feature_vector(
                    trajectory, i, window_size
                )
                
                X_train.append(feature_vector)
                y_x_train.append(trajectory.positions[i].x)
                y_y_train.append(trajectory.positions[i].y)
        
        return np.array(X_train), np.array(y_x_train), np.array(y_y_train)
    
    def _create_feature_vector(self, trajectory, current_idx, window_size):
        """Create feature vector for GP input."""
        features = []
        
        # Recent positions (relative to current)
        current_pos = trajectory.positions[current_idx]
        for i in range(max(0, current_idx - window_size), current_idx):
            pos = trajectory.positions[i]
            features.extend([
                pos.x - current_pos.x,
                pos.y - current_pos.y
            ])
        
        # Recent velocities
        for i in range(max(0, current_idx - window_size), current_idx):
            if i < len(trajectory.velocities):
                vel = trajectory.velocities[i]
                features.extend([vel.vx, vel.vy])
        
        # Temporal features
        current_time = trajectory.timestamps[current_idx]
        for i in range(max(0, current_idx - window_size), current_idx):
            time_diff = trajectory.timestamps[i] - current_time
            features.append(time_diff)
        
        # Pad to fixed size if necessary
        target_size = 50  # Fixed feature vector size
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        elif len(features) > target_size:
            features = features[:target_size]
        
        return features
```

### Key Features

- **Flexible Kernels**: Multiple kernel types for different trajectory patterns
- **Principled Uncertainty**: Provides confidence intervals for predictions
- **Non-linear Learning**: Captures complex relationships without explicit modeling
- **Hyperparameter Optimization**: Automatic tuning for optimal performance

### Use Cases

- **Safety-Critical Applications**: Provides uncertainty bounds for risk assessment
- **Research and Development**: Flexible framework for trajectory pattern exploration
- **Small Datasets**: Works well with limited training data
- **Interpretable Predictions**: Uncertainty quantification aids decision-making

---

## Model Comparison and Selection

### Performance Characteristics

| Model | Accuracy | Speed | Uncertainty | Training Time | Use Case |
|-------|----------|-------|-------------|---------------|----------|
| **Polynomial** | Medium-High | Fast | Bayesian | Fast | Smooth curves |
| **KNN** | High | Medium | Ensemble | Medium | Complex patterns |
| **Gaussian Process** | High | Slow | Principled | Slow | Small datasets |

### Selection Guidelines

```python
def select_advanced_model(
    training_size: int,
    scenario_complexity: str,
    uncertainty_requirements: str,
    latency_constraints: str
) -> TrajectoryPredictor:
    """
    Select appropriate advanced model based on requirements.
    """
    
    if latency_constraints == "strict" and training_size > 1000:
        # Fast inference required with large dataset
        return PolynomialTrajectoryPredictor(max_degree=3)
    
    elif scenario_complexity == "high" and training_size > 5000:
        # Complex patterns with sufficient data
        return KNNTrajectoryPredictor(k=7, similarity_metric="dtw")
    
    elif uncertainty_requirements == "critical" and training_size < 2000:
        # Need principled uncertainty with smaller dataset
        return GaussianProcessPredictor(
            kernel_type="rbf_periodic",
            optimize_hyperparameters=True
        )
    
    elif scenario_complexity == "medium":
        # Balanced approach
        return PolynomialTrajectoryPredictor(max_degree=4)
    
    else:
        # Default to KNN for general use
        return KNNTrajectoryPredictor(k=5)
```

### Ensemble Configuration

```python
class AdvancedEnsemble(TrajectoryPredictor):
    """Ensemble of advanced models with intelligent weighting."""
    
    def __init__(self):
        super().__init__()
        self.models = {
            "polynomial": PolynomialTrajectoryPredictor(),
            "knn": KNNTrajectoryPredictor(k=5),
            "gp": GaussianProcessPredictor()
        }
        self.model_name = "advanced_ensemble"
    
    async def predict(self, trajectory: TrajectoryData, **kwargs) -> TrajectoryData:
        # Get predictions from all models
        predictions = {}
        uncertainties = {}
        
        for name, model in self.models.items():
            if model.is_trained:
                pred = await model.predict(trajectory, **kwargs)
                predictions[name] = pred
                uncertainties[name] = self._extract_uncertainty(pred)
        
        # Dynamic weighting based on uncertainty and trajectory characteristics
        weights = self._compute_dynamic_weights(trajectory, uncertainties)
        
        # Create weighted ensemble prediction
        return self._weighted_ensemble(predictions, weights, trajectory)
```

## Summary

Advanced models provide sophisticated trajectory prediction capabilities:

- **Polynomial**: Best for smooth, curved trajectories with fast inference
- **KNN**: Excellent for complex, context-dependent scenarios
- **Gaussian Process**: Ideal when principled uncertainty is critical

Choose based on your specific requirements for accuracy, speed, uncertainty quantification, and training data availability. Ensemble approaches can combine the strengths of multiple models for optimal performance across diverse scenarios.