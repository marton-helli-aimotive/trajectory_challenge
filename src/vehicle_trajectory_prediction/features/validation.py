"""Feature validation and quality assessment for vehicle trajectory prediction."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import cdist
import logging
import json
from datetime import datetime
from pathlib import Path

from ..core.config import BaseConfig
from ..core.types import TrajectoryData, TrajectoryPoint

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig(BaseConfig):
    """Configuration for feature validation."""
    
    # Validation thresholds
    min_feature_quality_score: float = 0.7
    max_outlier_ratio: float = 0.1
    min_correlation_threshold: float = 0.3
    max_duplicate_ratio: float = 0.05
    
    # Physics validation
    enable_physics_validation: bool = True
    max_velocity_violation_ratio: float = 0.05
    max_acceleration_violation_ratio: float = 0.05
    max_curvature_violation_ratio: float = 0.1
    
    # Statistical validation
    enable_statistical_validation: bool = True
    outlier_z_threshold: float = 3.0
    outlier_iqr_multiplier: float = 1.5
    normality_test_alpha: float = 0.05
    
    # Quality scoring
    enable_quality_scoring: bool = True
    quality_weights: Dict[str, float] = None
    
    # Report generation
    enable_detailed_reports: bool = True
    report_output_path: str = "validation_reports"
    save_validation_plots: bool = False
    
    def __post_init__(self):
        if self.quality_weights is None:
            self.quality_weights = {
                'completeness': 0.25,
                'consistency': 0.25,
                'physics': 0.25,
                'statistics': 0.25
            }


class BaseValidator(ABC):
    """Base class for feature validators."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    @abstractmethod
    def validate_features(self, features: Dict[str, Any], trajectory: TrajectoryData) -> Dict[str, Any]:
        """Validate features for a trajectory."""
        pass
    
    @abstractmethod
    def get_validation_name(self) -> str:
        """Get name of the validator."""
        pass


class FeatureValidator(BaseValidator):
    """Basic feature validation."""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
    
    def get_validation_name(self) -> str:
        return "basic_feature_validation"
    
    def validate_features(self, features: Dict[str, Any], trajectory: TrajectoryData) -> Dict[str, Any]:
        """Validate basic feature properties."""
        validation_results = {
            'validation_name': self.get_validation_name(),
            'trajectory_id': trajectory.vehicle_id,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_score': 0.0,
            'issues': [],
            'warnings': [],
            'passed': True
        }
        
        try:
            # Check feature completeness
            completeness_score = self._validate_completeness(features)
            validation_results['completeness_score'] = completeness_score
            
            # Check feature consistency
            consistency_score = self._validate_consistency(features)
            validation_results['consistency_score'] = consistency_score
            
            # Check for duplicates
            duplicate_score = self._validate_duplicates(features)
            validation_results['duplicate_score'] = duplicate_score
            
            # Check data types
            type_score = self._validate_data_types(features)
            validation_results['type_score'] = type_score
            
            # Calculate overall score
            scores = [completeness_score, consistency_score, duplicate_score, type_score]
            validation_results['overall_score'] = np.mean(scores)
            
            # Determine if validation passed
            validation_results['passed'] = validation_results['overall_score'] >= self.config.min_feature_quality_score
            
            # Generate issues and warnings
            self._generate_issues_and_warnings(validation_results, features)
            
        except Exception as e:
            logger.error(f"Error in feature validation: {e}")
            validation_results['passed'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def _validate_completeness(self, features: Dict[str, Any]) -> float:
        """Validate feature completeness."""
        if not features:
            return 0.0
        
        total_features = len(features)
        non_empty_features = 0
        
        for feature_name, feature_data in features.items():
            if self._is_feature_complete(feature_data):
                non_empty_features += 1
        
        completeness_ratio = non_empty_features / total_features if total_features > 0 else 0.0
        return completeness_ratio
    
    def _is_feature_complete(self, feature_data: Any) -> bool:
        """Check if a feature is complete."""
        if feature_data is None:
            return False
        
        if isinstance(feature_data, (list, np.ndarray)):
            return len(feature_data) > 0
        
        if isinstance(feature_data, dict):
            return len(feature_data) > 0
        
        if isinstance(feature_data, (int, float)):
            return not np.isnan(feature_data)
        
        return True
    
    def _validate_consistency(self, features: Dict[str, Any]) -> float:
        """Validate feature consistency."""
        if not features:
            return 0.0
        
        consistency_scores = []
        
        for feature_name, feature_data in features.items():
            if isinstance(feature_data, (list, np.ndarray)):
                # Check for consistent data types
                if len(feature_data) > 0:
                    data_types = set(type(x) for x in feature_data)
                    if len(data_types) == 1:
                        consistency_scores.append(1.0)
                    else:
                        consistency_scores.append(0.5)
                else:
                    consistency_scores.append(1.0)
            else:
                consistency_scores.append(1.0)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _validate_duplicates(self, features: Dict[str, Any]) -> float:
        """Validate for duplicate features."""
        if not features:
            return 1.0
        
        feature_names = list(features.keys())
        unique_names = set(feature_names)
        
        duplicate_ratio = 1.0 - (len(unique_names) / len(feature_names))
        
        if duplicate_ratio > self.config.max_duplicate_ratio:
            return 0.0
        
        return 1.0 - duplicate_ratio
    
    def _validate_data_types(self, features: Dict[str, Any]) -> float:
        """Validate data types of features."""
        if not features:
            return 0.0
        
        valid_types = {int, float, np.integer, np.floating, list, np.ndarray, dict}
        valid_features = 0
        
        for feature_data in features.values():
            if type(feature_data) in valid_types:
                valid_features += 1
        
        return valid_features / len(features) if features else 0.0
    
    def _generate_issues_and_warnings(self, validation_results: Dict[str, Any], features: Dict[str, Any]):
        """Generate issues and warnings based on validation results."""
        # Check completeness
        if validation_results.get('completeness_score', 0) < 0.8:
            validation_results['warnings'].append("Low feature completeness detected")
        
        if validation_results.get('completeness_score', 0) < 0.5:
            validation_results['issues'].append("Very low feature completeness")
        
        # Check consistency
        if validation_results.get('consistency_score', 0) < 0.8:
            validation_results['warnings'].append("Feature consistency issues detected")
        
        # Check duplicates
        if validation_results.get('duplicate_score', 0) < 0.9:
            validation_results['issues'].append("Duplicate features detected")
        
        # Check data types
        if validation_results.get('type_score', 0) < 1.0:
            validation_results['issues'].append("Invalid data types detected")


class PhysicsInformedValidator(BaseValidator):
    """Physics-informed feature validation."""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
    
    def get_validation_name(self) -> str:
        return "physics_informed_validation"
    
    def validate_features(self, features: Dict[str, Any], trajectory: TrajectoryData) -> Dict[str, Any]:
        """Validate features using physics constraints."""
        validation_results = {
            'validation_name': self.get_validation_name(),
            'trajectory_id': trajectory.vehicle_id,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_score': 0.0,
            'physics_violations': {},
            'issues': [],
            'warnings': [],
            'passed': True
        }
        
        if not self.config.enable_physics_validation:
            validation_results['overall_score'] = 1.0
            return validation_results
        
        try:
            # Validate velocity features
            velocity_score = self._validate_velocity_features(features, trajectory)
            validation_results['velocity_physics_score'] = velocity_score
            
            # Validate acceleration features
            acceleration_score = self._validate_acceleration_features(features, trajectory)
            validation_results['acceleration_physics_score'] = acceleration_score
            
            # Validate curvature features
            curvature_score = self._validate_curvature_features(features, trajectory)
            validation_results['curvature_physics_score'] = curvature_score
            
            # Validate spatial-temporal features
            spatial_temporal_score = self._validate_spatial_temporal_features(features, trajectory)
            validation_results['spatial_temporal_physics_score'] = spatial_temporal_score
            
            # Calculate overall physics score
            physics_scores = [
                velocity_score,
                acceleration_score,
                curvature_score,
                spatial_temporal_score
            ]
            validation_results['overall_score'] = np.mean(physics_scores)
            
            # Determine if validation passed
            validation_results['passed'] = validation_results['overall_score'] >= self.config.min_feature_quality_score
            
            # Generate physics-specific issues and warnings
            self._generate_physics_issues_and_warnings(validation_results)
            
        except Exception as e:
            logger.error(f"Error in physics validation: {e}")
            validation_results['passed'] = False
            validation_results['issues'].append(f"Physics validation error: {str(e)}")
        
        return validation_results
    
    def _validate_velocity_features(self, features: Dict[str, Any], trajectory: TrajectoryData) -> float:
        """Validate velocity-related features."""
        velocity_violations = 0
        total_checks = 0
        
        # Check velocity magnitude features
        if 'velocity' in features:
            velocity_data = features['velocity']
            if isinstance(velocity_data, dict) and 'velocity_magnitude' in velocity_data:
                velocity_magnitude = velocity_data['velocity_magnitude']
                if isinstance(velocity_magnitude, (list, np.ndarray)):
                    # Check for unrealistic velocities (> 50 m/s)
                    unrealistic_velocities = np.sum(np.array(velocity_magnitude) > 50.0)
                    total_checks += len(velocity_magnitude)
                    velocity_violations += unrealistic_velocities
                    
                    # Check for negative velocities
                    negative_velocities = np.sum(np.array(velocity_magnitude) < 0.0)
                    velocity_violations += negative_velocities
        
        # Check velocity consistency with trajectory
        if len(trajectory.timestamps) > 1:
            dt = np.diff(trajectory.timestamps)
            dx = np.diff(trajectory.x_positions)
            dy = np.diff(trajectory.y_positions)
            
            # Calculate actual velocity from trajectory
            actual_velocity = np.sqrt(dx**2 + dy**2) / dt
            actual_velocity = np.where(dt > 0, actual_velocity, 0)
            
            # Compare with stored velocity features
            if 'velocity' in features and isinstance(features['velocity'], dict):
                if 'velocity_magnitude' in features['velocity']:
                    stored_velocity = features['velocity']['velocity_magnitude']
                    if isinstance(stored_velocity, (list, np.ndarray)) and len(stored_velocity) == len(actual_velocity):
                        # Check correlation
                        correlation = np.corrcoef(actual_velocity, stored_velocity)[0, 1]
                        if np.isnan(correlation) or correlation < self.config.min_correlation_threshold:
                            velocity_violations += 1
                        total_checks += 1
        
        violation_ratio = velocity_violations / total_checks if total_checks > 0 else 0.0
        return max(0.0, 1.0 - violation_ratio)
    
    def _validate_acceleration_features(self, features: Dict[str, Any], trajectory: TrajectoryData) -> float:
        """Validate acceleration-related features."""
        acceleration_violations = 0
        total_checks = 0
        
        # Check acceleration magnitude features
        if 'acceleration' in features:
            acceleration_data = features['acceleration']
            if isinstance(acceleration_data, dict) and 'acceleration_magnitude' in acceleration_data:
                acceleration_magnitude = acceleration_data['acceleration_magnitude']
                if isinstance(acceleration_magnitude, (list, np.ndarray)):
                    # Check for unrealistic accelerations (> 10 m/s²)
                    unrealistic_accelerations = np.sum(np.abs(np.array(acceleration_magnitude)) > 10.0)
                    total_checks += len(acceleration_magnitude)
                    acceleration_violations += unrealistic_accelerations
        
        # Check jerk features
        if 'acceleration' in features:
            acceleration_data = features['acceleration']
            if isinstance(acceleration_data, dict) and 'jerk_magnitude' in acceleration_data:
                jerk_magnitude = acceleration_data['jerk_magnitude']
                if isinstance(jerk_magnitude, (list, np.ndarray)):
                    # Check for unrealistic jerk (> 5 m/s³)
                    unrealistic_jerk = np.sum(np.abs(np.array(jerk_magnitude)) > 5.0)
                    total_checks += len(jerk_magnitude)
                    acceleration_violations += unrealistic_jerk
        
        violation_ratio = acceleration_violations / total_checks if total_checks > 0 else 0.0
        return max(0.0, 1.0 - violation_ratio)
    
    def _validate_curvature_features(self, features: Dict[str, Any], trajectory: TrajectoryData) -> float:
        """Validate curvature-related features."""
        curvature_violations = 0
        total_checks = 0
        
        # Check curvature features
        if 'curvature' in features:
            curvature_data = features['curvature']
            if isinstance(curvature_data, dict) and 'curvature' in curvature_data:
                curvature = curvature_data['curvature']
                if isinstance(curvature, (list, np.ndarray)):
                    # Check for unrealistic curvature values
                    unrealistic_curvature = np.sum(np.abs(np.array(curvature)) > 1.0)
                    total_checks += len(curvature)
                    curvature_violations += unrealistic_curvature
        
        # Check turning radius features
        if 'curvature' in features:
            curvature_data = features['curvature']
            if isinstance(curvature_data, dict) and 'turning_radius' in curvature_data:
                turning_radius = curvature_data['turning_radius']
                if isinstance(turning_radius, (list, np.ndarray)):
                    # Check for unrealistic turning radius (< 5m)
                    unrealistic_radius = np.sum(np.array(turning_radius) < 5.0)
                    total_checks += len(turning_radius)
                    curvature_violations += unrealistic_radius
        
        violation_ratio = curvature_violations / total_checks if total_checks > 0 else 0.0
        return max(0.0, 1.0 - violation_ratio)
    
    def _validate_spatial_temporal_features(self, features: Dict[str, Any], trajectory: TrajectoryData) -> float:
        """Validate spatial-temporal features."""
        spatial_temporal_violations = 0
        total_checks = 0
        
        # Check spatial density features
        if 'spatial_temporal' in features:
            spatial_data = features['spatial_temporal']
            if isinstance(spatial_data, dict):
                # Check spatial density
                if 'spatial_density' in spatial_data:
                    spatial_density = spatial_data['spatial_density']
                    if isinstance(spatial_density, (int, float)):
                        # Check for unrealistic spatial density
                        if spatial_density < 0 or spatial_density > 1000:
                            spatial_temporal_violations += 1
                        total_checks += 1
                
                # Check temporal density
                if 'temporal_density' in spatial_data:
                    temporal_density = spatial_data['temporal_density']
                    if isinstance(temporal_density, (int, float)):
                        # Check for unrealistic temporal density
                        if temporal_density < 0 or temporal_density > 100:
                            spatial_temporal_violations += 1
                        total_checks += 1
        
        violation_ratio = spatial_temporal_violations / total_checks if total_checks > 0 else 0.0
        return max(0.0, 1.0 - violation_ratio)
    
    def _generate_physics_issues_and_warnings(self, validation_results: Dict[str, Any]):
        """Generate physics-specific issues and warnings."""
        # Check velocity physics
        if validation_results.get('velocity_physics_score', 0) < 0.8:
            validation_results['warnings'].append("Velocity physics violations detected")
        
        if validation_results.get('velocity_physics_score', 0) < 0.5:
            validation_results['issues'].append("Significant velocity physics violations")
        
        # Check acceleration physics
        if validation_results.get('acceleration_physics_score', 0) < 0.8:
            validation_results['warnings'].append("Acceleration physics violations detected")
        
        if validation_results.get('acceleration_physics_score', 0) < 0.5:
            validation_results['issues'].append("Significant acceleration physics violations")
        
        # Check curvature physics
        if validation_results.get('curvature_physics_score', 0) < 0.8:
            validation_results['warnings'].append("Curvature physics violations detected")
        
        # Check spatial-temporal physics
        if validation_results.get('spatial_temporal_physics_score', 0) < 0.8:
            validation_results['warnings'].append("Spatial-temporal physics violations detected")


class FeatureQualityScorer(BaseValidator):
    """Feature quality scoring."""
    
    def __init__(self, config: ValidationConfig):
        super().__init__(config)
    
    def get_validation_name(self) -> str:
        return "feature_quality_scoring"
    
    def validate_features(self, features: Dict[str, Any], trajectory: TrajectoryData) -> Dict[str, Any]:
        """Score feature quality."""
        validation_results = {
            'validation_name': self.get_validation_name(),
            'trajectory_id': trajectory.vehicle_id,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_quality_score': 0.0,
            'quality_breakdown': {},
            'quality_level': 'unknown',
            'recommendations': []
        }
        
        if not self.config.enable_quality_scoring:
            validation_results['overall_quality_score'] = 1.0
            validation_results['quality_level'] = 'excellent'
            return validation_results
        
        try:
            # Score completeness
            completeness_score = self._score_completeness(features)
            validation_results['quality_breakdown']['completeness'] = completeness_score
            
            # Score consistency
            consistency_score = self._score_consistency(features)
            validation_results['quality_breakdown']['consistency'] = consistency_score
            
            # Score physics compliance
            physics_score = self._score_physics_compliance(features, trajectory)
            validation_results['quality_breakdown']['physics'] = physics_score
            
            # Score statistical quality
            statistical_score = self._score_statistical_quality(features)
            validation_results['quality_breakdown']['statistics'] = statistical_score
            
            # Calculate overall quality score
            weights = self.config.quality_weights
            overall_score = (
                completeness_score * weights['completeness'] +
                consistency_score * weights['consistency'] +
                physics_score * weights['physics'] +
                statistical_score * weights['statistics']
            )
            validation_results['overall_quality_score'] = overall_score
            
            # Determine quality level
            validation_results['quality_level'] = self._determine_quality_level(overall_score)
            
            # Generate recommendations
            validation_results['recommendations'] = self._generate_recommendations(validation_results)
            
        except Exception as e:
            logger.error(f"Error in quality scoring: {e}")
            validation_results['overall_quality_score'] = 0.0
            validation_results['quality_level'] = 'poor'
            validation_results['recommendations'].append(f"Quality scoring error: {str(e)}")
        
        return validation_results
    
    def _score_completeness(self, features: Dict[str, Any]) -> float:
        """Score feature completeness."""
        if not features:
            return 0.0
        
        total_features = len(features)
        complete_features = 0
        
        for feature_data in features.values():
            if self._is_feature_complete(feature_data):
                complete_features += 1
        
        return complete_features / total_features if total_features > 0 else 0.0
    
    def _is_feature_complete(self, feature_data: Any) -> bool:
        """Check if a feature is complete."""
        if feature_data is None:
            return False
        
        if isinstance(feature_data, (list, np.ndarray)):
            return len(feature_data) > 0
        
        if isinstance(feature_data, dict):
            return len(feature_data) > 0
        
        if isinstance(feature_data, (int, float)):
            return not np.isnan(feature_data)
        
        return True
    
    def _score_consistency(self, features: Dict[str, Any]) -> float:
        """Score feature consistency."""
        if not features:
            return 0.0
        
        consistency_scores = []
        
        for feature_data in features.values():
            if isinstance(feature_data, (list, np.ndarray)):
                # Check for consistent data types and ranges
                if len(feature_data) > 0:
                    # Check data type consistency
                    data_types = set(type(x) for x in feature_data)
                    type_consistency = 1.0 if len(data_types) == 1 else 0.5
                    
                    # Check range consistency
                    if isinstance(feature_data[0], (int, float)):
                        feature_array = np.array(feature_data)
                        if len(feature_array) > 1:
                            range_consistency = 1.0 - (np.std(feature_array) / (np.mean(feature_array) + 1e-6))
                            range_consistency = max(0.0, min(1.0, range_consistency))
                        else:
                            range_consistency = 1.0
                    else:
                        range_consistency = 1.0
                    
                    consistency_scores.append((type_consistency + range_consistency) / 2)
                else:
                    consistency_scores.append(1.0)
            else:
                consistency_scores.append(1.0)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _score_physics_compliance(self, features: Dict[str, Any], trajectory: TrajectoryData) -> float:
        """Score physics compliance."""
        if not self.config.enable_physics_validation:
            return 1.0
        
        physics_scores = []
        
        # Score velocity physics
        if 'velocity' in features:
            velocity_score = self._score_velocity_physics(features['velocity'])
            physics_scores.append(velocity_score)
        
        # Score acceleration physics
        if 'acceleration' in features:
            acceleration_score = self._score_acceleration_physics(features['acceleration'])
            physics_scores.append(acceleration_score)
        
        # Score curvature physics
        if 'curvature' in features:
            curvature_score = self._score_curvature_physics(features['curvature'])
            physics_scores.append(curvature_score)
        
        return np.mean(physics_scores) if physics_scores else 1.0
    
    def _score_velocity_physics(self, velocity_features: Dict[str, Any]) -> float:
        """Score velocity physics compliance."""
        if not isinstance(velocity_features, dict):
            return 0.0
        
        violations = 0
        total_checks = 0
        
        if 'velocity_magnitude' in velocity_features:
            velocity_magnitude = velocity_features['velocity_magnitude']
            if isinstance(velocity_magnitude, (list, np.ndarray)):
                # Check for unrealistic velocities
                unrealistic = np.sum(np.array(velocity_magnitude) > 50.0)
                negative = np.sum(np.array(velocity_magnitude) < 0.0)
                violations += unrealistic + negative
                total_checks += len(velocity_magnitude)
        
        return max(0.0, 1.0 - (violations / total_checks)) if total_checks > 0 else 1.0
    
    def _score_acceleration_physics(self, acceleration_features: Dict[str, Any]) -> float:
        """Score acceleration physics compliance."""
        if not isinstance(acceleration_features, dict):
            return 0.0
        
        violations = 0
        total_checks = 0
        
        if 'acceleration_magnitude' in acceleration_features:
            acceleration_magnitude = acceleration_features['acceleration_magnitude']
            if isinstance(acceleration_magnitude, (list, np.ndarray)):
                # Check for unrealistic accelerations
                unrealistic = np.sum(np.abs(np.array(acceleration_magnitude)) > 10.0)
                violations += unrealistic
                total_checks += len(acceleration_magnitude)
        
        return max(0.0, 1.0 - (violations / total_checks)) if total_checks > 0 else 1.0
    
    def _score_curvature_physics(self, curvature_features: Dict[str, Any]) -> float:
        """Score curvature physics compliance."""
        if not isinstance(curvature_features, dict):
            return 0.0
        
        violations = 0
        total_checks = 0
        
        if 'curvature' in curvature_features:
            curvature = curvature_features['curvature']
            if isinstance(curvature, (list, np.ndarray)):
                # Check for unrealistic curvature
                unrealistic = np.sum(np.abs(np.array(curvature)) > 1.0)
                violations += unrealistic
                total_checks += len(curvature)
        
        return max(0.0, 1.0 - (violations / total_checks)) if total_checks > 0 else 1.0
    
    def _score_statistical_quality(self, features: Dict[str, Any]) -> float:
        """Score statistical quality of features."""
        if not self.config.enable_statistical_validation:
            return 1.0
        
        statistical_scores = []
        
        for feature_data in features.values():
            if isinstance(feature_data, (list, np.ndarray)) and len(feature_data) > 0:
                feature_array = np.array(feature_data)
                
                # Check for outliers using Z-score
                if len(feature_array) > 3:
                    z_scores = np.abs(stats.zscore(feature_array))
                    outlier_ratio = np.sum(z_scores > self.config.outlier_z_threshold) / len(feature_array)
                    outlier_score = max(0.0, 1.0 - outlier_ratio)
                    statistical_scores.append(outlier_score)
                else:
                    statistical_scores.append(1.0)
            else:
                statistical_scores.append(1.0)
        
        return np.mean(statistical_scores) if statistical_scores else 1.0
    
    def _determine_quality_level(self, quality_score: float) -> str:
        """Determine quality level based on score."""
        if quality_score >= 0.9:
            return 'excellent'
        elif quality_score >= 0.8:
            return 'good'
        elif quality_score >= 0.7:
            return 'fair'
        elif quality_score >= 0.6:
            return 'poor'
        else:
            return 'very_poor'
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality scores."""
        recommendations = []
        quality_breakdown = validation_results.get('quality_breakdown', {})
        
        # Completeness recommendations
        completeness_score = quality_breakdown.get('completeness', 0)
        if completeness_score < 0.8:
            recommendations.append("Improve feature completeness by ensuring all required features are computed")
        
        if completeness_score < 0.5:
            recommendations.append("Critical: Many features are missing, review feature extraction pipeline")
        
        # Consistency recommendations
        consistency_score = quality_breakdown.get('consistency', 0)
        if consistency_score < 0.8:
            recommendations.append("Improve feature consistency by ensuring uniform data types and ranges")
        
        # Physics recommendations
        physics_score = quality_breakdown.get('physics', 0)
        if physics_score < 0.8:
            recommendations.append("Review physics constraints in feature extraction")
        
        if physics_score < 0.5:
            recommendations.append("Critical: Many physics violations detected, check feature computation")
        
        # Statistical recommendations
        statistical_score = quality_breakdown.get('statistics', 0)
        if statistical_score < 0.8:
            recommendations.append("Consider outlier detection and removal in feature preprocessing")
        
        # Overall recommendations
        overall_score = validation_results.get('overall_quality_score', 0)
        if overall_score < 0.7:
            recommendations.append("Overall feature quality is low, consider comprehensive review of feature engineering pipeline")
        
        return recommendations


class ValidationReport:
    """Generate comprehensive validation reports."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validators = {
            'basic': FeatureValidator(config),
            'physics': PhysicsInformedValidator(config),
            'quality': FeatureQualityScorer(config)
        }
    
    def generate_validation_report(self, features: Dict[str, Any], trajectory: TrajectoryData) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            'report_metadata': {
                'trajectory_id': trajectory.vehicle_id,
                'generation_timestamp': datetime.now().isoformat(),
                'feature_count': len(features),
                'trajectory_length': len(trajectory.timestamps)
            },
            'validation_results': {},
            'summary': {},
            'recommendations': []
        }
        
        # Run all validators
        for validator_name, validator in self.validators.items():
            try:
                validation_result = validator.validate_features(features, trajectory)
                report['validation_results'][validator_name] = validation_result
            except Exception as e:
                logger.error(f"Error running {validator_name} validator: {e}")
                report['validation_results'][validator_name] = {
                    'error': str(e),
                    'passed': False
                }
        
        # Generate summary
        report['summary'] = self._generate_summary(report['validation_results'])
        
        # Generate recommendations
        report['recommendations'] = self._generate_overall_recommendations(report)
        
        # Save report if enabled
        if self.config.enable_detailed_reports:
            self._save_validation_report(report, trajectory.vehicle_id)
        
        return report
    
    def _generate_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary."""
        summary = {
            'overall_passed': True,
            'overall_score': 0.0,
            'validator_scores': {},
            'critical_issues': 0,
            'warnings': 0
        }
        
        scores = []
        for validator_name, result in validation_results.items():
            if 'error' not in result:
                score = result.get('overall_score', 0.0)
                scores.append(score)
                summary['validator_scores'][validator_name] = score
                
                if not result.get('passed', True):
                    summary['overall_passed'] = False
                
                # Count issues and warnings
                summary['critical_issues'] += len(result.get('issues', []))
                summary['warnings'] += len(result.get('warnings', []))
        
        if scores:
            summary['overall_score'] = np.mean(scores)
        
        return summary
    
    def _generate_overall_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations."""
        recommendations = []
        
        # Check overall score
        overall_score = report['summary'].get('overall_score', 0)
        if overall_score < 0.7:
            recommendations.append("Overall feature quality is below acceptable threshold")
        
        # Check critical issues
        critical_issues = report['summary'].get('critical_issues', 0)
        if critical_issues > 0:
            recommendations.append(f"Address {critical_issues} critical validation issues")
        
        # Check warnings
        warnings = report['summary'].get('warnings', 0)
        if warnings > 5:
            recommendations.append(f"Review {warnings} validation warnings")
        
        # Add specific recommendations from quality scorer
        if 'quality' in report['validation_results']:
            quality_recommendations = report['validation_results']['quality'].get('recommendations', [])
            recommendations.extend(quality_recommendations)
        
        return list(set(recommendations))  # Remove duplicates
    
    def _save_validation_report(self, report: Dict[str, Any], trajectory_id: str):
        """Save validation report to file."""
        try:
            output_path = Path(self.config.report_output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{trajectory_id}_{timestamp}.json"
            file_path = output_path / filename
            
            # Save report
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Saved validation report to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving validation report: {e}")
    
    def batch_validate_features(self, features_batch: Dict[str, Dict[str, Any]], 
                              trajectories: List[TrajectoryData]) -> Dict[str, Dict[str, Any]]:
        """Validate features for a batch of trajectories."""
        batch_reports = {}
        
        for trajectory in trajectories:
            trajectory_id = trajectory.vehicle_id
            if trajectory_id in features_batch:
                features = features_batch[trajectory_id]
                report = self.generate_validation_report(features, trajectory)
                batch_reports[trajectory_id] = report
        
        return batch_reports