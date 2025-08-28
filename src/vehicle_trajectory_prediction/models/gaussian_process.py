"""Gaussian Process Regression for trajectory prediction with uncertainty quantification."""

from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import GPy
    GPY_AVAILABLE = True
except ImportError:
    GPY_AVAILABLE = False
    GPy = None

try:
    import torch
    import gpytorch
    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False
    torch = None
    gpytorch = None

from .base import BaseTrajectoryPredictor, PredictionResult
from ..core.models import Trajectory, TrajectoryPoint
from ..core.config import ModelConfig

logger = logging.getLogger(__name__)


class GaussianProcessPredictor(BaseTrajectoryPredictor):
    """
    Gaussian Process Regression model for trajectory prediction with uncertainty quantification.
    
    This model uses Gaussian Processes to predict vehicle trajectories while providing
    uncertainty estimates. It can use either GPy or GPyTorch as the backend.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the Gaussian Process predictor."""
        super().__init__(config)
        
        # GP-specific configuration
        self.kernel_type = getattr(self.config, 'kernel_type', 'rbf')
        self.noise_variance = getattr(self.config, 'noise_variance', 1e-6)
        self.backend = getattr(self.config, 'gp_backend', 'gpy')  # 'gpy' or 'gpytorch'
        self.optimize_kernel = getattr(self.config, 'optimize_kernel', True)
        self.n_restarts = getattr(self.config, 'n_restarts', 10)
        
        # Model components
        self.gp_model = None
        self.x_train = None
        self.y_train = None
        self.feature_names = None
        
        # Validate backend availability
        if self.backend == 'gpy' and not GPY_AVAILABLE:
            logger.warning("GPy not available, falling back to GPyTorch")
            self.backend = 'gpytorch'
        
        if self.backend == 'gpytorch' and not GPYTORCH_AVAILABLE:
            raise ImportError("Neither GPy nor GPyTorch are available. Please install one of them.")
    
    def _extract_features(self, trajectory: Trajectory) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from trajectory for GP training.
        
        Args:
            trajectory: Input trajectory
            
        Returns:
            Feature array and feature names
        """
        points = trajectory.points
        
        if len(points) < 3:
            raise ValueError("Trajectory must have at least 3 points for feature extraction")
        
        features = []
        feature_names = []
        
        # Time features
        start_time = points[0].timestamp
        times = [(p.timestamp - start_time).total_seconds() for p in points]
        features.extend(times)
        feature_names.extend([f'time_{i}' for i in range(len(times))])
        
        # Position features
        x_positions = [p.x for p in points]
        y_positions = [p.y for p in points]
        features.extend(x_positions)
        features.extend(y_positions)
        feature_names.extend([f'x_{i}' for i in range(len(x_positions))])
        feature_names.extend([f'y_{i}' for i in range(len(y_positions))])
        
        # Velocity features (if available)
        if hasattr(points[0], 'velocity') and points[0].velocity is not None:
            velocities = [p.velocity for p in points]
            features.extend(velocities)
            feature_names.extend([f'velocity_{i}' for i in range(len(velocities))])
        
        # Acceleration features (if available)
        if hasattr(points[0], 'acceleration') and points[0].acceleration is not None:
            accelerations = [p.acceleration for p in points]
            features.extend(accelerations)
            feature_names.extend([f'acceleration_{i}' for i in range(len(accelerations))])
        
        # Heading features (if available)
        if hasattr(points[0], 'heading') and points[0].heading is not None:
            headings = [p.heading for p in points]
            features.extend(headings)
            feature_names.extend([f'heading_{i}' for i in range(len(headings))])
        
        return np.array(features), feature_names
    
    def _create_gpy_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Create and train GPy model."""
        if not GPY_AVAILABLE:
            raise ImportError("GPy is not available")
        
        # Create kernel based on configuration
        if self.kernel_type == 'rbf':
            kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)
        elif self.kernel_type == 'matern32':
            kernel = GPy.kern.Matern32(input_dim=X.shape[1], ARD=True)
        elif self.kernel_type == 'matern52':
            kernel = GPy.kern.Matern52(input_dim=X.shape[1], ARD=True)
        elif self.kernel_type == 'rbf_linear':
            kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True) + GPy.kern.Linear(input_dim=X.shape[1])
        else:
            kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)
        
        # Create GP model
        model = GPy.models.GPRegression(X, y.reshape(-1, 1), kernel)
        
        # Set noise variance
        model.Gaussian_noise.variance = self.noise_variance
        
        # Optimize if requested
        if self.optimize_kernel:
            model.optimize_restarts(num_restarts=self.n_restarts, verbose=False)
        
        return model
    
    def _create_gpytorch_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Create and train GPyTorch model."""
        if not GPYTORCH_AVAILABLE:
            raise ImportError("GPyTorch is not available")
        
        # Convert to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Create kernel
        if self.kernel_type == 'rbf':
            kernel = gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1])
        elif self.kernel_type == 'matern32':
            kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=X.shape[1])
        elif self.kernel_type == 'matern52':
            kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=X.shape[1])
        else:
            kernel = gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1])
        
        # Create GP model
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
            
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X_tensor, y_tensor, likelihood)
        
        # Train model
        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        for i in range(100):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            optimizer.step()
        
        return model, likelihood
    
    def train(self, trajectories: List[Trajectory]) -> None:
        """
        Train the Gaussian Process model.
        
        Args:
            trajectories: List of training trajectories
        """
        if not trajectories:
            raise ValueError("No trajectories provided for training")
        
        logger.info(f"Training Gaussian Process model with {len(trajectories)} trajectories")
        
        # Extract features and targets
        X_list = []
        y_list = []
        
        for trajectory in trajectories:
            try:
                # Extract features from input trajectory
                features, feature_names = self._extract_features(trajectory)
                X_list.append(features)
                
                # Extract target (future positions)
                future_points = trajectory.points[-self.prediction_horizon:]
                future_x = [p.x for p in future_points]
                future_y = [p.y for p in future_points]
                y_list.extend(future_x + future_y)
                
            except Exception as e:
                logger.warning(f"Skipping trajectory due to error: {e}")
                continue
        
        if not X_list:
            raise ValueError("No valid trajectories for training")
        
        # Convert to numpy arrays
        self.x_train = np.array(X_list)
        self.y_train = np.array(y_list)
        self.feature_names = feature_names
        
        logger.info(f"Training data shape: X={self.x_train.shape}, y={self.y_train.shape}")
        
        # Create and train model based on backend
        if self.backend == 'gpy':
            self.gp_model = self._create_gpy_model(self.x_train, self.y_train)
        else:  # gpytorch
            self.gp_model, self.likelihood = self._create_gpytorch_model(self.x_train, self.y_train)
        
        self.is_trained = True
        logger.info("Gaussian Process model training completed")
    
    def predict(self, trajectory: Trajectory) -> PredictionResult:
        """
        Predict future trajectory using Gaussian Process regression.
        
        Args:
            trajectory: Input trajectory for prediction
            
        Returns:
            PredictionResult with predicted trajectory and uncertainty
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Extract features from input trajectory
        features, _ = self._extract_features(trajectory)
        X_test = features.reshape(1, -1)
        
        # Make prediction based on backend
        if self.backend == 'gpy':
            mean, variance = self.gp_model.predict(X_test)
            mean = mean.flatten()
            variance = variance.flatten()
        else:  # gpytorch
            self.gp_model.eval()
            self.likelihood.eval()
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                prediction = self.likelihood(self.gp_model(X_test_tensor))
                mean = prediction.mean.numpy().flatten()
                variance = prediction.variance.numpy().flatten()
        
        # Generate timestamps
        last_timestamp = trajectory.points[-1].timestamp
        timestamps = [
            last_timestamp + timedelta(seconds=i * self.time_step)
            for i in range(1, self.prediction_horizon + 1)
        ]
        
        # Extract x and y predictions
        n_points = self.prediction_horizon
        x_predictions = mean[:n_points]
        y_predictions = mean[n_points:2*n_points]
        
        # Extract uncertainties
        x_uncertainties = np.sqrt(variance[:n_points])
        y_uncertainties = np.sqrt(variance[n_points:2*n_points])
        
        # Create predicted trajectory points
        predicted_points = []
        for i in range(n_points):
            point = TrajectoryPoint(
                x=x_predictions[i],
                y=y_predictions[i],
                timestamp=timestamps[i],
                velocity=None,  # Could be calculated if needed
                acceleration=None,
                heading=None
            )
            predicted_points.append(point)
        
        # Create uncertainty dictionary
        uncertainty = {
            'x_std': x_uncertainties.tolist(),
            'y_std': y_uncertainties.tolist(),
            'confidence_intervals': {
                'x_95': [
                    (x_predictions[i] - 1.96 * x_uncertainties[i],
                     x_predictions[i] + 1.96 * x_uncertainties[i])
                    for i in range(n_points)
                ],
                'y_95': [
                    (y_predictions[i] - 1.96 * y_uncertainties[i],
                     y_predictions[i] + 1.96 * y_uncertainties[i])
                    for i in range(n_points)
                ]
            }
        }
        
        return PredictionResult(
            predicted_points=predicted_points,
            timestamps=timestamps,
            x_positions=x_predictions,
            y_positions=y_predictions,
            velocities=None,
            accelerations=None,
            headings=None,
            confidence_scores=None,
            uncertainty=uncertainty
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        info = super().get_model_info()
        info.update({
            'model_type': 'Gaussian Process Regression',
            'backend': self.backend,
            'kernel_type': self.kernel_type,
            'noise_variance': self.noise_variance,
            'optimize_kernel': self.optimize_kernel,
            'n_restarts': self.n_restarts,
            'feature_names': self.feature_names,
            'training_samples': len(self.x_train) if self.x_train is not None else 0,
            'feature_dimension': self.x_train.shape[1] if self.x_train is not None else 0
        })
        return info
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        import joblib
        
        model_data = {
            'config': self.config,
            'x_train': self.x_train,
            'y_train': self.y_train,
            'feature_names': self.feature_names,
            'backend': self.backend,
            'kernel_type': self.kernel_type,
            'noise_variance': self.noise_variance,
            'optimize_kernel': self.optimize_kernel,
            'n_restarts': self.n_restarts,
            'is_trained': self.is_trained
        }
        
        if self.backend == 'gpy':
            model_data['gp_model'] = self.gp_model
        else:  # gpytorch
            model_data['gp_model'] = self.gp_model.state_dict()
            model_data['likelihood'] = self.likelihood.state_dict()
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        # Restore configuration and parameters
        self.config = model_data['config']
        self.x_train = model_data['x_train']
        self.y_train = model_data['y_train']
        self.feature_names = model_data['feature_names']
        self.backend = model_data['backend']
        self.kernel_type = model_data['kernel_type']
        self.noise_variance = model_data['noise_variance']
        self.optimize_kernel = model_data['optimize_kernel']
        self.n_restarts = model_data['n_restarts']
        self.is_trained = model_data['is_trained']
        
        # Restore model based on backend
        if self.backend == 'gpy':
            self.gp_model = model_data['gp_model']
        else:  # gpytorch
            # Recreate model structure
            X_tensor = torch.tensor(self.x_train, dtype=torch.float32)
            y_tensor = torch.tensor(self.y_train, dtype=torch.float32)
            
            if self.kernel_type == 'rbf':
                kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.x_train.shape[1])
            elif self.kernel_type == 'matern32':
                kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=self.x_train.shape[1])
            elif self.kernel_type == 'matern52':
                kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.x_train.shape[1])
            else:
                kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.x_train.shape[1])
            
            class ExactGPModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super().__init__(train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
                
                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.gp_model = ExactGPModel(X_tensor, y_tensor, likelihood)
            self.likelihood = likelihood
            
            # Load state dicts
            self.gp_model.load_state_dict(model_data['gp_model'])
            self.likelihood.load_state_dict(model_data['likelihood'])
        
        logger.info(f"Model loaded from {filepath}")
    
    def predict_batch(
        self, 
        trajectories: List[Trajectory],
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> List[PredictionResult]:
        """Predict future trajectories for multiple inputs."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        results = []
        for trajectory in trajectories:
            try:
                result = self.predict(trajectory, prediction_horizon, prediction_frequency)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict trajectory {trajectory.vehicle_id}: {e}")
                # Create a default result with the original trajectory
                timestamps = self._prepare_prediction_timestamps(trajectory, prediction_horizon, prediction_frequency)
                default_points = [trajectory.points[-1]] * len(timestamps)
                results.append(PredictionResult(
                    predicted_points=default_points,
                    timestamps=timestamps,
                    x_positions=[p.x for p in default_points],
                    y_positions=[p.y for p in default_points],
                    velocities=[p.velocity for p in default_points],
                    accelerations=[p.acceleration for p in default_points],
                    headings=[p.heading for p in default_points],
                    confidence_scores=[0.0] * len(timestamps),
                    uncertainty={"std": [float('inf')] * len(timestamps)}
                ))
        
        return results