"""Mixture Density Networks for probabilistic trajectory prediction."""

from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None

from .base import BaseTrajectoryPredictor, PredictionResult
from ..core.models import Trajectory, TrajectoryPoint
from ..core.config import ModelConfig

logger = logging.getLogger(__name__)


class MixtureDensityNetwork(nn.Module):
    """
    Mixture Density Network for trajectory prediction.
    
    This network outputs parameters of a Gaussian mixture model to capture
    the uncertainty in trajectory predictions.
    """
    
    def __init__(self, input_dim: int, output_dim: int, n_components: int = 5, hidden_dim: int = 128):
        """
        Initialize the Mixture Density Network.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output (trajectory points)
            n_components: Number of Gaussian mixture components
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components
        
        # Shared feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Output layers for mixture parameters
        self.pi_net = nn.Linear(hidden_dim // 2, n_components)  # Mixing coefficients
        self.mu_net = nn.Linear(hidden_dim // 2, n_components * output_dim)  # Means
        self.sigma_net = nn.Linear(hidden_dim // 2, n_components * output_dim)  # Standard deviations
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (pi, mu, sigma) where:
            - pi: Mixing coefficients of shape (batch_size, n_components)
            - mu: Means of shape (batch_size, n_components, output_dim)
            - sigma: Standard deviations of shape (batch_size, n_components, output_dim)
        """
        features = self.feature_net(x)
        
        # Mixing coefficients (must sum to 1)
        pi = F.softmax(self.pi_net(features), dim=1)
        
        # Means
        mu = self.mu_net(features).view(-1, self.n_components, self.output_dim)
        
        # Standard deviations (must be positive)
        sigma = torch.exp(self.sigma_net(features)).view(-1, self.n_components, self.output_dim)
        
        return pi, mu, sigma


class MixtureDensityPredictor(BaseTrajectoryPredictor):
    """
    Mixture Density Network for trajectory prediction with uncertainty quantification.
    
    This model uses a neural network to predict the parameters of a Gaussian mixture model,
    allowing for multi-modal trajectory predictions with uncertainty estimates.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the Mixture Density Network predictor."""
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Mixture Density Networks")
        
        # Model-specific configuration
        self.n_components = getattr(self.config, 'n_components', 5)
        self.hidden_dim = getattr(self.config, 'hidden_dim', 128)
        self.learning_rate = getattr(self.config, 'learning_rate', 0.001)
        self.batch_size = getattr(self.config, 'batch_size', 32)
        self.n_epochs = getattr(self.config, 'n_epochs', 100)
        self.early_stopping_patience = getattr(self.config, 'early_stopping_patience', 10)
        self.weight_decay = getattr(self.config, 'weight_decay', 1e-5)
        
        # Model components
        self.model = None
        self.optimizer = None
        self.feature_names = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
    
    def _extract_features(self, trajectory: Trajectory) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from trajectory for MDN training.
        
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
        
        # Velocity features (if available or calculated)
        velocities = []
        if hasattr(points[0], 'velocity') and points[0].velocity is not None:
            velocities = [p.velocity for p in points]
        else:
            # Calculate velocities from positions
            for i in range(1, len(points)):
                dt = (points[i].timestamp - points[i-1].timestamp).total_seconds()
                if dt > 0:
                    dx = points[i].x - points[i-1].x
                    dy = points[i].y - points[i-1].y
                    velocity = np.sqrt(dx**2 + dy**2) / dt
                    velocities.append(velocity)
                else:
                    velocities.append(0.0)
            velocities.insert(0, velocities[0] if velocities else 0.0)
        
        features.extend(velocities)
        feature_names.extend([f'velocity_{i}' for i in range(len(velocities))])
        
        # Acceleration features (if available or calculated)
        accelerations = []
        if hasattr(points[0], 'acceleration') and points[0].acceleration is not None:
            accelerations = [p.acceleration for p in points]
        else:
            # Calculate accelerations from velocities
            for i in range(1, len(velocities)):
                dt = (points[i].timestamp - points[i-1].timestamp).total_seconds()
                if dt > 0:
                    acceleration = (velocities[i] - velocities[i-1]) / dt
                    accelerations.append(acceleration)
                else:
                    accelerations.append(0.0)
            accelerations.insert(0, accelerations[0] if accelerations else 0.0)
        
        features.extend(accelerations)
        feature_names.extend([f'acceleration_{i}' for i in range(len(accelerations))])
        
        # Heading features (if available or calculated)
        headings = []
        if hasattr(points[0], 'heading') and points[0].heading is not None:
            headings = [p.heading for p in points]
        else:
            # Calculate headings from positions
            for i in range(1, len(points)):
                dx = points[i].x - points[i-1].x
                dy = points[i].y - points[i-1].y
                heading = np.arctan2(dy, dx)
                headings.append(heading)
            headings.insert(0, headings[0] if headings else 0.0)
        
        features.extend(headings)
        feature_names.extend([f'heading_{i}' for i in range(len(headings))])
        
        # Statistical features
        features.extend([
            np.mean(x_positions), np.std(x_positions),
            np.mean(y_positions), np.std(y_positions),
            np.mean(velocities), np.std(velocities),
            np.mean(accelerations), np.std(accelerations),
            np.mean(headings), np.std(headings)
        ])
        feature_names.extend([
            'mean_x', 'std_x', 'mean_y', 'std_y',
            'mean_velocity', 'std_velocity', 'mean_acceleration', 'std_acceleration',
            'mean_heading', 'std_heading'
        ])
        
        return np.array(features), feature_names
    
    def _gaussian_mixture_loss(self, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, 
                              target: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative log-likelihood loss for Gaussian mixture model.
        
        Args:
            pi: Mixing coefficients of shape (batch_size, n_components)
            mu: Means of shape (batch_size, n_components, output_dim)
            sigma: Standard deviations of shape (batch_size, n_components, output_dim)
            target: Target values of shape (batch_size, output_dim)
            
        Returns:
            Negative log-likelihood loss
        """
        # Expand target for broadcasting
        target = target.unsqueeze(1).expand(-1, self.n_components, -1)
        
        # Compute Gaussian probabilities
        prob = torch.exp(-0.5 * ((target - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        
        # Product over output dimensions
        prob = torch.prod(prob, dim=2)
        
        # Weighted sum over components
        weighted_prob = pi * prob
        
        # Sum over components and take negative log
        loss = -torch.log(torch.sum(weighted_prob, dim=1) + 1e-8)
        
        return torch.mean(loss)
    
    def train(self, trajectories: List[Trajectory]) -> None:
        """
        Train the Mixture Density Network.
        
        Args:
            trajectories: List of training trajectories
        """
        if not trajectories:
            raise ValueError("No trajectories provided for training")
        
        logger.info(f"Training Mixture Density Network with {len(trajectories)} trajectories")
        
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
        X = np.array(X_list)
        y = np.array(y_list)
        self.feature_names = feature_names
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Feature scaling
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X.shape[1]
        output_dim = self.prediction_horizon * 2  # x and y coordinates
        self.model = MixtureDensityNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            n_components=self.n_components,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                pi, mu, sigma = self.model(batch_X)
                
                # Compute loss
                loss = self._gaussian_mixture_loss(pi, mu, sigma, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.n_epochs}, Loss: {avg_loss:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        self.is_trained = True
        logger.info("Mixture Density Network training completed")
    
    def predict(self, trajectory: Trajectory) -> PredictionResult:
        """
        Predict future trajectory using Mixture Density Network.
        
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
        
        # Apply scaling
        X_test = self.scaler.transform(X_test)
        
        # Convert to tensor
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            pi, mu, sigma = self.model(X_tensor)
            
            # Convert to numpy
            pi = pi.cpu().numpy().flatten()
            mu = mu.cpu().numpy().squeeze(0)  # (n_components, output_dim)
            sigma = sigma.cpu().numpy().squeeze(0)  # (n_components, output_dim)
        
        # Generate timestamps
        last_timestamp = trajectory.points[-1].timestamp
        timestamps = [
            last_timestamp + timedelta(seconds=i * self.time_step)
            for i in range(1, self.prediction_horizon + 1)
        ]
        
        # Extract predictions for each component
        n_points = self.prediction_horizon
        x_predictions = mu[:, :n_points]  # (n_components, n_points)
        y_predictions = mu[:, n_points:]  # (n_components, n_points)
        
        # Use the component with highest mixing coefficient for main prediction
        best_component = np.argmax(pi)
        x_pred = x_predictions[best_component]
        y_pred = y_predictions[best_component]
        
        # Calculate uncertainty using all components
        x_uncertainty = np.sqrt(np.sum(pi[:, np.newaxis] * (x_predictions**2 + sigma[:, :n_points]**2), axis=0) - 
                               np.sum(pi[:, np.newaxis] * x_predictions, axis=0)**2)
        y_uncertainty = np.sqrt(np.sum(pi[:, np.newaxis] * (y_predictions**2 + sigma[:, n_points:]**2), axis=0) - 
                               np.sum(pi[:, np.newaxis] * y_predictions, axis=0)**2)
        
        # Create predicted trajectory points
        predicted_points = []
        for i in range(n_points):
            point = TrajectoryPoint(
                x=x_pred[i],
                y=y_pred[i],
                timestamp=timestamps[i],
                velocity=None,
                acceleration=None,
                heading=None
            )
            predicted_points.append(point)
        
        # Create uncertainty dictionary
        uncertainty = {
            'x_std': x_uncertainty.tolist(),
            'y_std': y_uncertainty.tolist(),
            'mixing_coefficients': pi.tolist(),
            'component_means': {
                'x': x_predictions.tolist(),
                'y': y_predictions.tolist()
            },
            'component_stds': {
                'x': sigma[:, :n_points].tolist(),
                'y': sigma[:, n_points:].tolist()
            }
        }
        
        return PredictionResult(
            predicted_points=predicted_points,
            timestamps=timestamps,
            x_positions=x_pred,
            y_positions=y_pred,
            velocities=None,
            accelerations=None,
            headings=None,
            confidence_scores=pi.tolist(),
            uncertainty=uncertainty
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        info = super().get_model_info()
        info.update({
            'model_type': 'Mixture Density Network',
            'n_components': self.n_components,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'weight_decay': self.weight_decay,
            'device': str(self.device),
            'feature_names': self.feature_names,
            'input_dimension': len(self.feature_names) if self.feature_names else 0,
            'output_dimension': self.prediction_horizon * 2
        })
        return info
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        import joblib
        
        model_data = {
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'n_components': self.n_components,
            'hidden_dim': self.hidden_dim,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        # Restore configuration and parameters
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.scaler = model_data['scaler']
        self.n_components = model_data['n_components']
        self.hidden_dim = model_data['hidden_dim']
        self.is_trained = model_data['is_trained']
        
        # Recreate and load model
        input_dim = len(self.feature_names) if self.feature_names else 0
        output_dim = self.prediction_horizon * 2
        
        self.model = MixtureDensityNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            n_components=self.n_components,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")