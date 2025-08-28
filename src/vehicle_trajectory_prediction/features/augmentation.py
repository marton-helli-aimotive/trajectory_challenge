"""Data augmentation techniques for vehicle trajectory prediction."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from scipy import interpolate
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
import logging
import random

from ..core.config import BaseConfig
from ..core.types import TrajectoryData, TrajectoryPoint

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig(BaseConfig):
    """Configuration for trajectory data augmentation."""
    
    # Noise injection parameters
    noise_std_position: float = 0.1
    noise_std_velocity: float = 0.05
    noise_std_timestamp: float = 0.01
    noise_type: str = "gaussian"  # "gaussian", "uniform", "laplace"
    
    # Interpolation parameters
    interpolation_method: str = "cubic"  # "linear", "cubic", "spline"
    max_interpolation_gap: float = 2.0
    interpolation_density: float = 2.0  # points per original point
    
    # Synthetic scenario parameters
    synthetic_scenario_count: int = 100
    scenario_complexity_range: Tuple[float, float] = (0.1, 1.0)
    scenario_duration_range: Tuple[float, float] = (10.0, 60.0)
    
    # Adversarial example parameters
    adversarial_perturbation_scale: float = 0.1
    adversarial_max_iterations: int = 10
    adversarial_epsilon: float = 0.01
    
    # General augmentation parameters
    enable_noise_injection: bool = True
    enable_interpolation: bool = True
    enable_synthetic_scenarios: bool = True
    enable_adversarial_examples: bool = True
    preserve_physics_constraints: bool = True


class BaseAugmentor(ABC):
    """Base class for trajectory augmentors."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        
    @abstractmethod
    def augment_trajectory(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Augment a single trajectory."""
        pass
    
    @abstractmethod
    def augment_trajectories(self, trajectories: List[TrajectoryData]) -> List[TrajectoryData]:
        """Augment multiple trajectories."""
        pass


class NoiseInjectionAugmentor(BaseAugmentor):
    """Inject noise into trajectory data."""
    
    def __init__(self, config: AugmentationConfig):
        super().__init__(config)
        self.noise_generators = {
            "gaussian": self._gaussian_noise,
            "uniform": self._uniform_noise,
            "laplace": self._laplace_noise
        }
    
    def augment_trajectory(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Inject noise into a single trajectory."""
        if not self.config.enable_noise_injection:
            return trajectory
        
        # Generate noise
        noise_generator = self.noise_generators.get(self.config.noise_type, self._gaussian_noise)
        
        # Add noise to positions
        x_noise = noise_generator(len(trajectory.x_positions), self.config.noise_std_position)
        y_noise = noise_generator(len(trajectory.y_positions), self.config.noise_std_position)
        
        # Add noise to timestamps
        t_noise = noise_generator(len(trajectory.timestamps), self.config.noise_std_timestamp)
        
        # Create augmented trajectory
        augmented_trajectory = TrajectoryData(
            vehicle_id=trajectory.vehicle_id,
            timestamps=trajectory.timestamps + t_noise,
            x_positions=trajectory.x_positions + x_noise,
            y_positions=trajectory.y_positions + y_noise
        )
        
        # Apply physics constraints if enabled
        if self.config.preserve_physics_constraints:
            augmented_trajectory = self._apply_physics_constraints(augmented_trajectory)
        
        return augmented_trajectory
    
    def augment_trajectories(self, trajectories: List[TrajectoryData]) -> List[TrajectoryData]:
        """Inject noise into multiple trajectories."""
        augmented_trajectories = []
        
        for trajectory in trajectories:
            try:
                augmented = self.augment_trajectory(trajectory)
                augmented_trajectories.append(augmented)
            except Exception as e:
                logger.error(f"Error augmenting trajectory {trajectory.vehicle_id}: {e}")
                augmented_trajectories.append(trajectory)  # Keep original
        
        return augmented_trajectories
    
    def _gaussian_noise(self, size: int, std: float) -> np.ndarray:
        """Generate Gaussian noise."""
        return np.random.normal(0, std, size)
    
    def _uniform_noise(self, size: int, std: float) -> np.ndarray:
        """Generate uniform noise."""
        # Scale uniform noise to have similar variance as Gaussian
        scale = std * np.sqrt(3)
        return np.random.uniform(-scale, scale, size)
    
    def _laplace_noise(self, size: int, std: float) -> np.ndarray:
        """Generate Laplace noise."""
        # Scale Laplace noise to have similar variance as Gaussian
        scale = std / np.sqrt(2)
        return np.random.laplace(0, scale, size)
    
    def _apply_physics_constraints(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Apply physics constraints to ensure realistic trajectory."""
        # Ensure timestamps are monotonically increasing
        timestamps = np.sort(trajectory.timestamps)
        
        # Ensure minimum time step
        dt = np.diff(timestamps)
        min_dt = 0.01  # Minimum 10ms time step
        dt = np.maximum(dt, min_dt)
        
        # Reconstruct timestamps
        timestamps = np.concatenate([[timestamps[0]], timestamps[0] + np.cumsum(dt)])
        
        return TrajectoryData(
            vehicle_id=trajectory.vehicle_id,
            timestamps=timestamps,
            x_positions=trajectory.x_positions,
            y_positions=trajectory.y_positions
        )


class InterpolationAugmentor(BaseAugmentor):
    """Augment trajectories through interpolation."""
    
    def __init__(self, config: AugmentationConfig):
        super().__init__(config)
        self.interpolators = {
            "linear": self._linear_interpolation,
            "cubic": self._cubic_interpolation,
            "spline": self._spline_interpolation
        }
    
    def augment_trajectory(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Augment trajectory through interpolation."""
        if not self.config.enable_interpolation or len(trajectory.timestamps) < 3:
            return trajectory
        
        # Choose interpolation method
        interpolator = self.interpolators.get(self.config.interpolation_method, self._cubic_interpolation)
        
        # Create denser time grid
        original_dt = np.diff(trajectory.timestamps)
        avg_dt = np.mean(original_dt)
        new_dt = avg_dt / self.config.interpolation_density
        
        # Generate new timestamps
        new_timestamps = np.arange(
            trajectory.timestamps[0],
            trajectory.timestamps[-1] + new_dt,
            new_dt
        )
        
        # Interpolate positions
        new_x_positions = interpolator(trajectory.timestamps, trajectory.x_positions, new_timestamps)
        new_y_positions = interpolator(trajectory.timestamps, trajectory.y_positions, new_timestamps)
        
        # Create augmented trajectory
        augmented_trajectory = TrajectoryData(
            vehicle_id=trajectory.vehicle_id,
            timestamps=new_timestamps,
            x_positions=new_x_positions,
            y_positions=new_y_positions
        )
        
        return augmented_trajectory
    
    def augment_trajectories(self, trajectories: List[TrajectoryData]) -> List[TrajectoryData]:
        """Augment multiple trajectories through interpolation."""
        augmented_trajectories = []
        
        for trajectory in trajectories:
            try:
                augmented = self.augment_trajectory(trajectory)
                augmented_trajectories.append(augmented)
            except Exception as e:
                logger.error(f"Error interpolating trajectory {trajectory.vehicle_id}: {e}")
                augmented_trajectories.append(trajectory)  # Keep original
        
        return augmented_trajectories
    
    def _linear_interpolation(self, x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        """Linear interpolation."""
        return np.interp(x_new, x_old, y_old)
    
    def _cubic_interpolation(self, x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        """Cubic interpolation."""
        if len(x_old) < 4:
            return self._linear_interpolation(x_old, y_old, x_new)
        
        # Use scipy's cubic interpolation
        f = interpolate.interp1d(x_old, y_old, kind='cubic', bounds_error=False, fill_value='extrapolate')
        return f(x_new)
    
    def _spline_interpolation(self, x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        """Spline interpolation."""
        if len(x_old) < 4:
            return self._linear_interpolation(x_old, y_old, x_new)
        
        # Use scipy's spline interpolation
        tck = interpolate.splrep(x_old, y_old, s=0)
        return interpolate.splev(x_new, tck, der=0)


class SyntheticScenarioGenerator(BaseAugmentor):
    """Generate synthetic trajectory scenarios."""
    
    def __init__(self, config: AugmentationConfig):
        super().__init__(config)
        self.scenario_templates = {
            "straight_line": self._generate_straight_line,
            "curve": self._generate_curve,
            "lane_change": self._generate_lane_change,
            "acceleration": self._generate_acceleration,
            "deceleration": self._generate_deceleration,
            "stop_and_go": self._generate_stop_and_go
        }
    
    def augment_trajectory(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Generate synthetic scenario based on trajectory characteristics."""
        if not self.config.enable_synthetic_scenarios:
            return trajectory
        
        # Analyze trajectory to determine scenario type
        scenario_type = self._analyze_trajectory(trajectory)
        
        # Generate synthetic scenario
        generator = self.scenario_templates.get(scenario_type, self._generate_straight_line)
        synthetic_trajectory = generator(trajectory)
        
        return synthetic_trajectory
    
    def augment_trajectories(self, trajectories: List[TrajectoryData]) -> List[TrajectoryData]:
        """Generate synthetic scenarios for multiple trajectories."""
        synthetic_trajectories = []
        
        for trajectory in trajectories:
            try:
                synthetic = self.augment_trajectory(trajectory)
                synthetic_trajectories.append(synthetic)
            except Exception as e:
                logger.error(f"Error generating synthetic scenario for trajectory {trajectory.vehicle_id}: {e}")
                synthetic_trajectories.append(trajectory)  # Keep original
        
        return synthetic_trajectories
    
    def _analyze_trajectory(self, trajectory: TrajectoryData) -> str:
        """Analyze trajectory to determine scenario type."""
        if len(trajectory.timestamps) < 3:
            return "straight_line"
        
        # Calculate basic characteristics
        dx = np.diff(trajectory.x_positions)
        dy = np.diff(trajectory.y_positions)
        dt = np.diff(trajectory.timestamps)
        
        # Calculate velocity
        velocity = np.sqrt(dx**2 + dy**2) / dt
        velocity = np.where(dt > 0, velocity, 0)
        
        # Calculate curvature
        curvature = self._calculate_curvature(trajectory)
        
        # Determine scenario type based on characteristics
        if np.std(curvature) > 0.1:
            return "curve"
        elif np.std(dy) > np.std(dx) * 0.5:
            return "lane_change"
        elif np.std(velocity) > np.mean(velocity) * 0.3:
            if np.min(velocity) < np.mean(velocity) * 0.5:
                return "stop_and_go"
            elif np.max(velocity) > np.mean(velocity) * 1.5:
                return "acceleration"
            else:
                return "deceleration"
        else:
            return "straight_line"
    
    def _calculate_curvature(self, trajectory: TrajectoryData) -> np.ndarray:
        """Calculate curvature of trajectory."""
        x = trajectory.x_positions
        y = trajectory.y_positions
        
        if len(x) < 3:
            return np.array([0.0])
        
        curvature = np.zeros(len(x))
        for i in range(1, len(x) - 1):
            p1 = np.array([x[i-1], y[i-1]])
            p2 = np.array([x[i], y[i]])
            p3 = np.array([x[i+1], y[i+1]])
            
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)
            
            if a * b * c > 0:
                s = (a + b + c) / 2
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                if area > 0:
                    radius = (a * b * c) / (4 * area)
                    curvature[i] = 1 / radius if radius > 0 else 0
        
        return curvature
    
    def _generate_straight_line(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Generate straight line trajectory."""
        duration = random.uniform(*self.config.scenario_duration_range)
        complexity = random.uniform(*self.config.scenario_complexity_range)
        
        # Generate timestamps
        timestamps = np.linspace(0, duration, int(duration * 10))
        
        # Generate straight line path
        start_x, start_y = 0, 0
        end_x = duration * 20  # 20 m/s average speed
        end_y = 0
        
        x_positions = np.linspace(start_x, end_x, len(timestamps))
        y_positions = np.linspace(start_y, end_y, len(timestamps))
        
        # Add some complexity
        if complexity > 0.5:
            noise = np.random.normal(0, complexity * 2, len(timestamps))
            y_positions += noise
        
        return TrajectoryData(
            vehicle_id=f"synthetic_straight_{len(timestamps)}",
            timestamps=timestamps,
            x_positions=x_positions,
            y_positions=y_positions
        )
    
    def _generate_curve(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Generate curved trajectory."""
        duration = random.uniform(*self.config.scenario_duration_range)
        complexity = random.uniform(*self.config.scenario_complexity_range)
        
        # Generate timestamps
        timestamps = np.linspace(0, duration, int(duration * 10))
        
        # Generate curved path
        t = timestamps
        radius = 50 + complexity * 100  # Variable radius
        angular_velocity = 0.5 + complexity * 1.0  # Variable angular velocity
        
        x_positions = radius * np.cos(angular_velocity * t)
        y_positions = radius * np.sin(angular_velocity * t)
        
        return TrajectoryData(
            vehicle_id=f"synthetic_curve_{len(timestamps)}",
            timestamps=timestamps,
            x_positions=x_positions,
            y_positions=y_positions
        )
    
    def _generate_lane_change(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Generate lane change trajectory."""
        duration = random.uniform(*self.config.scenario_duration_range)
        complexity = random.uniform(*self.config.scenario_complexity_range)
        
        # Generate timestamps
        timestamps = np.linspace(0, duration, int(duration * 10))
        
        # Generate lane change path
        t = timestamps
        lane_width = 3.5  # Standard lane width
        change_duration = duration * 0.3  # 30% of total time for lane change
        
        # Sigmoid function for smooth lane change
        sigmoid = 1 / (1 + np.exp(-10 * (t - duration/2) / change_duration))
        y_positions = lane_width * (sigmoid - 0.5)
        x_positions = 20 * t  # Constant forward motion
        
        return TrajectoryData(
            vehicle_id=f"synthetic_lane_change_{len(timestamps)}",
            timestamps=timestamps,
            x_positions=x_positions,
            y_positions=y_positions
        )
    
    def _generate_acceleration(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Generate acceleration trajectory."""
        duration = random.uniform(*self.config.scenario_duration_range)
        complexity = random.uniform(*self.config.scenario_complexity_range)
        
        # Generate timestamps
        timestamps = np.linspace(0, duration, int(duration * 10))
        
        # Generate acceleration path
        t = timestamps
        initial_velocity = 10  # m/s
        acceleration = 2 + complexity * 3  # m/s²
        
        # Position with constant acceleration
        x_positions = initial_velocity * t + 0.5 * acceleration * t**2
        y_positions = np.zeros_like(x_positions)
        
        return TrajectoryData(
            vehicle_id=f"synthetic_acceleration_{len(timestamps)}",
            timestamps=timestamps,
            x_positions=x_positions,
            y_positions=y_positions
        )
    
    def _generate_deceleration(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Generate deceleration trajectory."""
        duration = random.uniform(*self.config.scenario_duration_range)
        complexity = random.uniform(*self.config.scenario_complexity_range)
        
        # Generate timestamps
        timestamps = np.linspace(0, duration, int(duration * 10))
        
        # Generate deceleration path
        t = timestamps
        initial_velocity = 30  # m/s
        deceleration = 2 + complexity * 3  # m/s²
        
        # Position with constant deceleration
        x_positions = initial_velocity * t - 0.5 * deceleration * t**2
        x_positions = np.maximum(x_positions, 0)  # Don't go backwards
        y_positions = np.zeros_like(x_positions)
        
        return TrajectoryData(
            vehicle_id=f"synthetic_deceleration_{len(timestamps)}",
            timestamps=timestamps,
            x_positions=x_positions,
            y_positions=y_positions
        )
    
    def _generate_stop_and_go(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Generate stop-and-go trajectory."""
        duration = random.uniform(*self.config.scenario_duration_range)
        complexity = random.uniform(*self.config.scenario_complexity_range)
        
        # Generate timestamps
        timestamps = np.linspace(0, duration, int(duration * 10))
        
        # Generate stop-and-go path
        t = timestamps
        base_velocity = 15  # m/s
        
        # Create velocity profile with stops
        velocity_profile = base_velocity * np.ones_like(t)
        stop_times = [duration * 0.3, duration * 0.7]  # Stop at 30% and 70%
        
        for stop_time in stop_times:
            stop_duration = 2.0  # 2 seconds stop
            stop_mask = (t >= stop_time) & (t <= stop_time + stop_duration)
            velocity_profile[stop_mask] = 0
        
        # Integrate velocity to get position
        x_positions = np.cumsum(velocity_profile) * (t[1] - t[0])
        y_positions = np.zeros_like(x_positions)
        
        return TrajectoryData(
            vehicle_id=f"synthetic_stop_and_go_{len(timestamps)}",
            timestamps=timestamps,
            x_positions=x_positions,
            y_positions=y_positions
        )


class AdversarialExampleGenerator(BaseAugmentor):
    """Generate adversarial examples for trajectory prediction."""
    
    def __init__(self, config: AugmentationConfig):
        super().__init__(config)
    
    def augment_trajectory(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Generate adversarial example for trajectory."""
        if not self.config.enable_adversarial_examples or len(trajectory.timestamps) < 3:
            return trajectory
        
        # Create adversarial perturbation
        adversarial_trajectory = self._create_adversarial_perturbation(trajectory)
        
        return adversarial_trajectory
    
    def augment_trajectories(self, trajectories: List[TrajectoryData]) -> List[TrajectoryData]:
        """Generate adversarial examples for multiple trajectories."""
        adversarial_trajectories = []
        
        for trajectory in trajectories:
            try:
                adversarial = self.augment_trajectory(trajectory)
                adversarial_trajectories.append(adversarial)
            except Exception as e:
                logger.error(f"Error generating adversarial example for trajectory {trajectory.vehicle_id}: {e}")
                adversarial_trajectories.append(trajectory)  # Keep original
        
        return adversarial_trajectories
    
    def _create_adversarial_perturbation(self, trajectory: TrajectoryData) -> TrajectoryData:
        """Create adversarial perturbation for trajectory."""
        # Start with original trajectory
        perturbed_x = trajectory.x_positions.copy()
        perturbed_y = trajectory.y_positions.copy()
        
        # Iteratively apply perturbations
        for iteration in range(self.config.adversarial_max_iterations):
            # Calculate perturbation
            perturbation_x = np.random.normal(0, self.config.adversarial_perturbation_scale, len(perturbed_x))
            perturbation_y = np.random.normal(0, self.config.adversarial_perturbation_scale, len(perturbed_y))
            
            # Apply perturbation
            perturbed_x += perturbation_x
            perturbed_y += perturbation_y
            
            # Clip perturbation to epsilon ball
            perturbation_magnitude = np.sqrt(perturbation_x**2 + perturbation_y**2)
            if np.max(perturbation_magnitude) > self.config.adversarial_epsilon:
                scale_factor = self.config.adversarial_epsilon / np.max(perturbation_magnitude)
                perturbed_x = trajectory.x_positions + perturbation_x * scale_factor
                perturbed_y = trajectory.y_positions + perturbation_y * scale_factor
        
        return TrajectoryData(
            vehicle_id=f"adversarial_{trajectory.vehicle_id}",
            timestamps=trajectory.timestamps,
            x_positions=perturbed_x,
            y_positions=perturbed_y
        )


class TrajectoryAugmentor:
    """Main augmentor that combines all augmentation techniques."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.augmentors = {}
        
        # Initialize augmentors based on configuration
        if config.enable_noise_injection:
            self.augmentors['noise'] = NoiseInjectionAugmentor(config)
        
        if config.enable_interpolation:
            self.augmentors['interpolation'] = InterpolationAugmentor(config)
        
        if config.enable_synthetic_scenarios:
            self.augmentors['synthetic'] = SyntheticScenarioGenerator(config)
        
        if config.enable_adversarial_examples:
            self.augmentors['adversarial'] = AdversarialExampleGenerator(config)
    
    def augment_trajectory(self, trajectory: TrajectoryData, methods: Optional[List[str]] = None) -> TrajectoryData:
        """Augment trajectory using specified methods."""
        if methods is None:
            methods = list(self.augmentors.keys())
        
        augmented_trajectory = trajectory
        
        for method in methods:
            if method in self.augmentors:
                try:
                    augmented_trajectory = self.augmentors[method].augment_trajectory(augmented_trajectory)
                    logger.debug(f"Applied {method} augmentation to trajectory {trajectory.vehicle_id}")
                except Exception as e:
                    logger.error(f"Error applying {method} augmentation: {e}")
        
        return augmented_trajectory
    
    def augment_trajectories(self, trajectories: List[TrajectoryData], methods: Optional[List[str]] = None) -> List[TrajectoryData]:
        """Augment multiple trajectories using specified methods."""
        if methods is None:
            methods = list(self.augmentors.keys())
        
        augmented_trajectories = []
        
        for trajectory in trajectories:
            try:
                augmented = self.augment_trajectory(trajectory, methods)
                augmented_trajectories.append(augmented)
            except Exception as e:
                logger.error(f"Error augmenting trajectory {trajectory.vehicle_id}: {e}")
                augmented_trajectories.append(trajectory)  # Keep original
        
        return augmented_trajectories
    
    def get_available_methods(self) -> List[str]:
        """Get list of available augmentation methods."""
        return list(self.augmentors.keys())
    
    def get_augmentation_statistics(self, original_trajectories: List[TrajectoryData], 
                                  augmented_trajectories: List[TrajectoryData]) -> Dict[str, Any]:
        """Get statistics about the augmentation process."""
        if len(original_trajectories) != len(augmented_trajectories):
            return {"error": "Mismatch in trajectory counts"}
        
        stats = {
            "total_trajectories": len(original_trajectories),
            "successful_augmentations": 0,
            "failed_augmentations": 0,
            "average_length_change": 0.0,
            "average_position_change": 0.0
        }
        
        length_changes = []
        position_changes = []
        
        for orig, aug in zip(original_trajectories, augmented_trajectories):
            try:
                # Check if augmentation was successful
                if orig.vehicle_id != aug.vehicle_id or len(orig.timestamps) != len(aug.timestamps):
                    stats["successful_augmentations"] += 1
                    
                    # Calculate changes
                    length_change = len(aug.timestamps) - len(orig.timestamps)
                    length_changes.append(length_change)
                    
                    # Calculate average position change
                    if len(orig.timestamps) > 0 and len(aug.timestamps) > 0:
                        min_len = min(len(orig.timestamps), len(aug.timestamps))
                        pos_change = np.mean(np.sqrt(
                            (orig.x_positions[:min_len] - aug.x_positions[:min_len])**2 +
                            (orig.y_positions[:min_len] - aug.y_positions[:min_len])**2
                        ))
                        position_changes.append(pos_change)
                else:
                    stats["failed_augmentations"] += 1
            except Exception as e:
                logger.error(f"Error calculating statistics: {e}")
                stats["failed_augmentations"] += 1
        
        if length_changes:
            stats["average_length_change"] = np.mean(length_changes)
        if position_changes:
            stats["average_position_change"] = np.mean(position_changes)
        
        return stats