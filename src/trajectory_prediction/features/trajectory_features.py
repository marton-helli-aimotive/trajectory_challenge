"""Trajectory feature extraction with physics-informed features."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TrajectoryFeatureExtractor:
    """Extract comprehensive features from trajectory data."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.window_size = config.get("window_size", 30)  # seconds
        self.prediction_horizon = config.get("prediction_horizon", 10)  # seconds
        self.sampling_rate = config.get("sampling_rate", 10)  # Hz
        
    async def extract_trajectory_features(self, trajectory_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract comprehensive features from a single vehicle trajectory."""
        if len(trajectory_df) < 2:
            return self._get_empty_features()
        
        # Ensure trajectory is sorted by timestamp
        trajectory_df = trajectory_df.sort_values("timestamp")
        
        features = {}
        
        # Basic trajectory properties
        features.update(await self._extract_basic_features(trajectory_df))
        
        # Kinematic features
        features.update(await self._extract_kinematic_features(trajectory_df))
        
        # Physics-informed features
        features.update(await self._extract_physics_features(trajectory_df))
        
        # Statistical features
        features.update(await self._extract_statistical_features(trajectory_df))
        
        # Temporal features
        features.update(await self._extract_temporal_features(trajectory_df))
        
        # Spatial features
        features.update(await self._extract_spatial_features(trajectory_df))
        
        # Behavioral features
        features.update(await self._extract_behavioral_features(trajectory_df))
        
        return features
    
    async def _extract_basic_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic trajectory properties."""
        features = {}
        
        # Trajectory length and duration
        features["trajectory_length"] = len(df)
        
        if df["timestamp"].dtype == "datetime64[ns]":
            duration = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
        else:
            duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
        
        features["trajectory_duration"] = duration
        features["sampling_frequency"] = len(df) / duration if duration > 0 else 0
        
        # Spatial extent
        features["x_min"] = df["x"].min()
        features["x_max"] = df["x"].max()
        features["y_min"] = df["y"].min()
        features["y_max"] = df["y"].max()
        features["spatial_range_x"] = features["x_max"] - features["x_min"]
        features["spatial_range_y"] = features["y_max"] - features["y_min"]
        
        # Total distance traveled
        distances = np.sqrt(np.diff(df["x"])**2 + np.diff(df["y"])**2)
        features["total_distance"] = np.sum(distances)
        features["avg_distance_per_step"] = np.mean(distances) if len(distances) > 0 else 0
        
        return features
    
    async def _extract_kinematic_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract kinematic features (velocity, acceleration, jerk)."""
        features = {}
        
        # Velocity statistics (if available in data)
        if "velocity" in df.columns:
            velocity = df["velocity"].dropna()
            if len(velocity) > 0:
                features["velocity_mean"] = velocity.mean()
                features["velocity_std"] = velocity.std()
                features["velocity_min"] = velocity.min()
                features["velocity_max"] = velocity.max()
                features["velocity_median"] = velocity.median()
                features["velocity_95th_percentile"] = velocity.quantile(0.95)
        
        # Compute velocity from position if not available
        if len(df) > 1:
            dx = np.diff(df["x"])
            dy = np.diff(df["y"])
            
            # Time differences
            if df["timestamp"].dtype == "datetime64[ns]":
                dt = np.diff(df["timestamp"]).astype('timedelta64[s]').astype(float)
            else:
                dt = np.diff(df["timestamp"])
            
            # Avoid division by zero
            dt = np.where(dt == 0, 1e-6, dt)
            
            # Computed velocity
            computed_velocity = np.sqrt(dx**2 + dy**2) / dt
            features["computed_velocity_mean"] = np.mean(computed_velocity)
            features["computed_velocity_std"] = np.std(computed_velocity)
            features["computed_velocity_max"] = np.max(computed_velocity)
        
        # Acceleration statistics
        if "acceleration" in df.columns:
            acceleration = df["acceleration"].dropna()
            if len(acceleration) > 0:
                features["acceleration_mean"] = acceleration.mean()
                features["acceleration_std"] = acceleration.std()
                features["acceleration_min"] = acceleration.min()
                features["acceleration_max"] = acceleration.max()
                features["acceleration_abs_mean"] = np.abs(acceleration).mean()
        
        # Compute acceleration from velocity
        if "velocity" in df.columns and len(df) > 2:
            velocity = df["velocity"].values
            if df["timestamp"].dtype == "datetime64[ns]":
                dt = np.diff(df["timestamp"]).astype('timedelta64[s]').astype(float)
            else:
                dt = np.diff(df["timestamp"])
            
            dt = np.where(dt == 0, 1e-6, dt)
            computed_acceleration = np.diff(velocity) / dt
            
            features["computed_acceleration_mean"] = np.mean(computed_acceleration)
            features["computed_acceleration_std"] = np.std(computed_acceleration)
            features["computed_acceleration_max"] = np.max(np.abs(computed_acceleration))
        
        # Jerk (rate of change of acceleration)
        if len(df) > 3 and "acceleration" in df.columns:
            acceleration = df["acceleration"].values
            if df["timestamp"].dtype == "datetime64[ns]":
                dt = np.diff(df["timestamp"]).astype('timedelta64[s]').astype(float)
            else:
                dt = np.diff(df["timestamp"])
            
            dt = np.where(dt == 0, 1e-6, dt)
            jerk = np.diff(acceleration) / dt[1:]  # Use dt from middle points
            
            features["jerk_mean"] = np.mean(jerk)
            features["jerk_std"] = np.std(jerk)
            features["jerk_max"] = np.max(np.abs(jerk))
        
        return features
    
    async def _extract_physics_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract physics-informed features."""
        features = {}
        
        if len(df) < 2:
            return features
        
        # Heading and turning behavior
        dx = np.diff(df["x"])
        dy = np.diff(df["y"])
        
        # Heading angles
        headings = np.arctan2(dy, dx)
        features["heading_mean"] = np.mean(headings)
        features["heading_std"] = np.std(headings)
        
        # Turning rate (angular velocity)
        if len(headings) > 1:
            heading_changes = np.diff(headings)
            # Handle angle wrapping
            heading_changes = np.where(heading_changes > np.pi, heading_changes - 2*np.pi, heading_changes)
            heading_changes = np.where(heading_changes < -np.pi, heading_changes + 2*np.pi, heading_changes)
            
            features["turning_rate_mean"] = np.mean(np.abs(heading_changes))
            features["turning_rate_max"] = np.max(np.abs(heading_changes))
            features["total_heading_change"] = np.sum(np.abs(heading_changes))
        
        # Curvature estimation
        if len(df) > 2:
            x, y = df["x"].values, df["y"].values
            
            # First and second derivatives
            dx_dt = np.gradient(x)
            dy_dt = np.gradient(y)
            d2x_dt2 = np.gradient(dx_dt)
            d2y_dt2 = np.gradient(dy_dt)
            
            # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
            denominator = np.power(dx_dt**2 + dy_dt**2, 1.5)
            
            # Avoid division by zero
            curvature = np.where(denominator > 1e-10, numerator / denominator, 0)
            
            features["curvature_mean"] = np.mean(curvature)
            features["curvature_max"] = np.max(curvature)
            features["curvature_std"] = np.std(curvature)
        
        # Energy-based features (assuming unit mass)
        if "velocity" in df.columns:
            velocity = df["velocity"].values
            kinetic_energy = 0.5 * velocity**2  # KE = 0.5 * m * v^2, assuming m=1
            
            features["kinetic_energy_mean"] = np.mean(kinetic_energy)
            features["kinetic_energy_max"] = np.max(kinetic_energy)
            features["kinetic_energy_change"] = np.max(kinetic_energy) - np.min(kinetic_energy)
        
        # Momentum-based features
        if "velocity" in df.columns and len(df) > 1:
            velocity = df["velocity"].values
            momentum_change = np.sum(np.abs(np.diff(velocity)))  # Change in momentum (assuming m=1)
            features["momentum_change_total"] = momentum_change
        
        return features
    
    async def _extract_statistical_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract statistical features from trajectory data."""
        features = {}
        
        # Position statistics
        for coord in ["x", "y"]:
            if coord in df.columns:
                values = df[coord].values
                features[f"{coord}_mean"] = np.mean(values)
                features[f"{coord}_std"] = np.std(values)
                features[f"{coord}_skewness"] = stats.skew(values)
                features[f"{coord}_kurtosis"] = stats.kurtosis(values)
                features[f"{coord}_range"] = np.max(values) - np.min(values)
        
        # Cross-correlation between x and y movements
        if len(df) > 10:  # Need sufficient data for correlation
            x_changes = np.diff(df["x"])
            y_changes = np.diff(df["y"])
            
            if len(x_changes) > 0 and np.std(x_changes) > 1e-10 and np.std(y_changes) > 1e-10:
                features["xy_correlation"] = np.corrcoef(x_changes, y_changes)[0, 1]
            else:
                features["xy_correlation"] = 0.0
        
        # Velocity distribution features (if available)
        if "velocity" in df.columns and len(df) > 10:
            velocity = df["velocity"].dropna()
            if len(velocity) > 0:
                features["velocity_skewness"] = stats.skew(velocity)
                features["velocity_kurtosis"] = stats.kurtosis(velocity)
                
                # Velocity percentiles
                features["velocity_10th_percentile"] = np.percentile(velocity, 10)
                features["velocity_90th_percentile"] = np.percentile(velocity, 90)
        
        return features
    
    async def _extract_temporal_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract temporal pattern features."""
        features = {}
        
        if "timestamp" not in df.columns:
            return features
        
        # Sampling regularity
        if df["timestamp"].dtype == "datetime64[ns]":
            time_diffs = np.diff(df["timestamp"]).astype('timedelta64[s]').astype(float)
        else:
            time_diffs = np.diff(df["timestamp"])
        
        if len(time_diffs) > 0:
            features["sampling_interval_mean"] = np.mean(time_diffs)
            features["sampling_interval_std"] = np.std(time_diffs)
            features["sampling_regularity"] = 1.0 - (np.std(time_diffs) / np.mean(time_diffs)) if np.mean(time_diffs) > 0 else 0
        
        # Time-of-day features (if timestamp is datetime)
        if df["timestamp"].dtype == "datetime64[ns]":
            hours = df["timestamp"].dt.hour
            features["avg_hour_of_day"] = hours.mean()
            features["hour_std"] = hours.std()
            
            # Day of week
            day_of_week = df["timestamp"].dt.dayofweek
            features["avg_day_of_week"] = day_of_week.mean()
        
        return features
    
    async def _extract_spatial_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract spatial pattern features."""
        features = {}
        
        if len(df) < 3:
            return features
        
        # Convex hull area (rough measure of spatial coverage)
        try:
            from scipy.spatial import ConvexHull
            points = df[["x", "y"]].values
            if len(points) >= 3:
                hull = ConvexHull(points)
                features["convex_hull_area"] = hull.volume  # In 2D, volume is area
                features["convex_hull_perimeter"] = hull.area  # In 2D, area is perimeter
        except:
            # Fallback: bounding box area
            x_range = df["x"].max() - df["x"].min()
            y_range = df["y"].max() - df["y"].min()
            features["bounding_box_area"] = x_range * y_range
        
        # Spatial density
        x_range = df["x"].max() - df["x"].min()
        y_range = df["y"].max() - df["y"].min()
        if x_range > 0 and y_range > 0:
            features["spatial_density"] = len(df) / (x_range * y_range)
        else:
            features["spatial_density"] = 0
        
        # Centroid and spread
        features["centroid_x"] = df["x"].mean()
        features["centroid_y"] = df["y"].mean()
        
        # Distance from centroid
        distances_from_centroid = np.sqrt(
            (df["x"] - features["centroid_x"])**2 + 
            (df["y"] - features["centroid_y"])**2
        )
        features["avg_distance_from_centroid"] = np.mean(distances_from_centroid)
        features["max_distance_from_centroid"] = np.max(distances_from_centroid)
        
        return features
    
    async def _extract_behavioral_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract behavioral pattern features."""
        features = {}
        
        if "velocity" not in df.columns or len(df) < 5:
            return features
        
        velocity = df["velocity"].values
        
        # Stop detection
        stop_threshold = self.config.get("stop_velocity_threshold", 1.0)  # m/s or ft/s
        stops = velocity < stop_threshold
        features["stop_ratio"] = np.mean(stops)
        features["num_stops"] = np.sum(np.diff(stops.astype(int)) == 1)  # Count stop events
        
        # Speed regimes
        low_speed_threshold = self.config.get("low_speed_threshold", 5.0)
        high_speed_threshold = self.config.get("high_speed_threshold", 25.0)
        
        low_speed = (velocity < low_speed_threshold) & (velocity >= stop_threshold)
        medium_speed = (velocity >= low_speed_threshold) & (velocity < high_speed_threshold)
        high_speed = velocity >= high_speed_threshold
        
        features["low_speed_ratio"] = np.mean(low_speed)
        features["medium_speed_ratio"] = np.mean(medium_speed)
        features["high_speed_ratio"] = np.mean(high_speed)
        
        # Acceleration patterns
        if "acceleration" in df.columns:
            acceleration = df["acceleration"].values
            
            # Acceleration/deceleration detection
            accel_threshold = self.config.get("acceleration_threshold", 1.0)
            acceleration_events = acceleration > accel_threshold
            deceleration_events = acceleration < -accel_threshold
            
            features["acceleration_ratio"] = np.mean(acceleration_events)
            features["deceleration_ratio"] = np.mean(deceleration_events)
            features["num_acceleration_events"] = np.sum(np.diff(acceleration_events.astype(int)) == 1)
            features["num_deceleration_events"] = np.sum(np.diff(deceleration_events.astype(int)) == 1)
        
        # Aggressive behavior indicators
        if len(df) > 10:
            # High jerk events (rapid changes in acceleration)
            if "acceleration" in df.columns and len(df) > 3:
                jerk = np.abs(np.diff(df["acceleration"]))
                high_jerk_threshold = np.percentile(jerk, 90)  # Top 10% of jerk values
                features["high_jerk_ratio"] = np.mean(jerk > high_jerk_threshold)
            
            # Velocity variability
            velocity_changes = np.abs(np.diff(velocity))
            features["velocity_variability"] = np.std(velocity_changes)
        
        return features
    
    def _get_empty_features(self) -> Dict[str, Any]:
        """Return empty features for invalid trajectories."""
        return {
            "trajectory_length": 0,
            "trajectory_duration": 0,
            "total_distance": 0,
            "velocity_mean": 0,
            "acceleration_mean": 0
        }


class SequenceFeatureExtractor:
    """Extract features for sequence-based models (sliding window approach)."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.window_size = config.get("window_size", 30)
        self.prediction_horizon = config.get("prediction_horizon", 10)
        self.overlap_ratio = config.get("overlap_ratio", 0.5)
        
    async def extract_sequence_features(
        self, 
        trajectory_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Extract features using sliding window approach."""
        features_list = []
        
        if len(trajectory_df) < self.window_size:
            return features_list
        
        # Calculate step size based on overlap
        step_size = max(1, int(self.window_size * (1 - self.overlap_ratio)))
        
        for start_idx in range(0, len(trajectory_df) - self.window_size + 1, step_size):
            end_idx = start_idx + self.window_size
            
            # Extract window
            window_df = trajectory_df.iloc[start_idx:end_idx].copy()
            
            # Check if we have enough future data for prediction target
            future_start = end_idx
            future_end = min(future_start + self.prediction_horizon, len(trajectory_df))
            
            if future_end > future_start:
                future_df = trajectory_df.iloc[future_start:future_end].copy()
                
                # Extract features for this window
                extractor = TrajectoryFeatureExtractor(self.config)
                features = await extractor.extract_trajectory_features(window_df)
                
                # Add sequence-specific features
                features["window_start_idx"] = start_idx
                features["window_end_idx"] = end_idx
                features["sequence_id"] = f"{trajectory_df.iloc[0]['vehicle_id']}_{start_idx}"
                
                # Add prediction targets
                features.update(self._extract_prediction_targets(future_df))
                
                features_list.append(features)
        
        return features_list
    
    def _extract_prediction_targets(self, future_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract prediction targets from future trajectory segment."""
        if len(future_df) == 0:
            return {}
        
        targets = {}
        
        # Position targets (final position)
        targets["target_x"] = future_df["x"].iloc[-1]
        targets["target_y"] = future_df["y"].iloc[-1]
        
        # Displacement targets
        targets["target_dx"] = future_df["x"].iloc[-1] - future_df["x"].iloc[0]
        targets["target_dy"] = future_df["y"].iloc[-1] - future_df["y"].iloc[0]
        
        # Velocity targets (if available)
        if "velocity" in future_df.columns:
            targets["target_velocity"] = future_df["velocity"].iloc[-1]
            targets["target_avg_velocity"] = future_df["velocity"].mean()
        
        # Trajectory shape targets
        if len(future_df) > 1:
            distances = np.sqrt(
                np.diff(future_df["x"])**2 + 
                np.diff(future_df["y"])**2
            )
            targets["target_path_length"] = np.sum(distances)
        
        return targets