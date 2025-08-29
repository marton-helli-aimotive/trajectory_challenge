"""Data quality monitoring and assessment tools."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Monitor and assess trajectory data quality."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.quality_thresholds = config.get("quality_thresholds", {})
        
    async def assess_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        logger.info(f"Assessing quality of {len(df)} records")
        
        assessments = {}
        
        # Basic completeness checks
        assessments["completeness"] = await self._assess_completeness(df)
        
        # Validity checks
        assessments["validity"] = await self._assess_validity(df)
        
        # Consistency checks
        assessments["consistency"] = await self._assess_consistency(df)
        
        # Trajectory-specific quality
        assessments["trajectory_quality"] = await self._assess_trajectory_quality(df)
        
        # Compute overall score
        scores = [
            assessments["completeness"]["score"],
            assessments["validity"]["score"],
            assessments["consistency"]["score"],
            assessments["trajectory_quality"]["score"]
        ]
        assessments["overall_score"] = np.mean(scores)
        
        # Generate recommendations
        assessments["recommendations"] = self._generate_recommendations(assessments)
        
        logger.info(f"Data quality assessment completed. Overall score: {assessments['overall_score']:.2f}")
        
        return assessments
    
    async def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness."""
        required_cols = ["vehicle_id", "timestamp", "x", "y"]
        optional_cols = ["velocity", "acceleration", "heading", "lane_id"]
        
        completeness = {}
        
        # Required column completeness
        for col in required_cols:
            if col in df.columns:
                non_null_ratio = (df[col].notna()).mean()
                completeness[f"{col}_completeness"] = non_null_ratio
            else:
                completeness[f"{col}_completeness"] = 0.0
        
        # Optional column completeness
        for col in optional_cols:
            if col in df.columns:
                non_null_ratio = (df[col].notna()).mean()
                completeness[f"{col}_completeness"] = non_null_ratio
            else:
                completeness[f"{col}_completeness"] = 0.0
        
        # Overall completeness score
        required_scores = [completeness[f"{col}_completeness"] for col in required_cols]
        completeness["score"] = np.mean(required_scores)
        
        # Issues
        completeness["issues"] = []
        for col in required_cols:
            if completeness[f"{col}_completeness"] < 0.95:
                completeness["issues"].append(f"High missing values in required column: {col}")
        
        return completeness
    
    async def _assess_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data validity (range checks, type consistency)."""
        validity = {}
        validity["issues"] = []
        
        # Coordinate validity
        if "x" in df.columns and "y" in df.columns:
            coord_valid = (
                (df["x"].abs() < 1e6) & 
                (df["y"].abs() < 1e6) &
                df["x"].notna() & 
                df["y"].notna()
            )
            validity["coordinate_validity"] = coord_valid.mean()
            
            if validity["coordinate_validity"] < 0.99:
                validity["issues"].append("Invalid coordinates detected")
        
        # Velocity validity
        if "velocity" in df.columns:
            velocity_valid = (
                (df["velocity"] >= 0) & 
                (df["velocity"] < 200) &  # < 200 m/s or ft/s
                df["velocity"].notna()
            )
            validity["velocity_validity"] = velocity_valid.mean()
            
            if validity["velocity_validity"] < 0.95:
                validity["issues"].append("Invalid velocities detected")
        
        # Acceleration validity
        if "acceleration" in df.columns:
            accel_valid = (
                (df["acceleration"].abs() < 20) &  # < 20 m/s² or ft/s²
                df["acceleration"].notna()
            )
            validity["acceleration_validity"] = accel_valid.mean()
            
            if validity["acceleration_validity"] < 0.95:
                validity["issues"].append("Invalid accelerations detected")
        
        # Timestamp validity
        if "timestamp" in df.columns:
            # Check for reasonable timestamp range
            current_year = datetime.now().year
            if df["timestamp"].dtype == "datetime64[ns]":
                timestamp_valid = (
                    (df["timestamp"].dt.year >= 1990) &
                    (df["timestamp"].dt.year <= current_year + 1)
                )
                validity["timestamp_validity"] = timestamp_valid.mean()
            else:
                # Assume numeric timestamp
                validity["timestamp_validity"] = 1.0
        
        # Overall validity score
        validity_scores = [
            validity.get("coordinate_validity", 1.0),
            validity.get("velocity_validity", 1.0),
            validity.get("acceleration_validity", 1.0),
            validity.get("timestamp_validity", 1.0)
        ]
        validity["score"] = np.mean(validity_scores)
        
        return validity
    
    async def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency (duplicates, ordering, etc.)."""
        consistency = {}
        consistency["issues"] = []
        
        # Duplicate detection
        if "vehicle_id" in df.columns and "timestamp" in df.columns:
            duplicate_ratio = df.duplicated(subset=["vehicle_id", "timestamp"]).mean()
            consistency["duplicate_ratio"] = duplicate_ratio
            
            if duplicate_ratio > 0.01:  # > 1% duplicates
                consistency["issues"].append(f"High duplicate ratio: {duplicate_ratio:.2%}")
        
        # Temporal ordering consistency
        if "vehicle_id" in df.columns and "timestamp" in df.columns:
            ordering_issues = 0
            total_vehicles = 0
            
            for vehicle_id, vehicle_df in df.groupby("vehicle_id"):
                if len(vehicle_df) > 1:
                    total_vehicles += 1
                    timestamps = vehicle_df["timestamp"].values
                    if not np.all(timestamps[1:] >= timestamps[:-1]):
                        ordering_issues += 1
            
            if total_vehicles > 0:
                consistency["temporal_ordering"] = 1.0 - (ordering_issues / total_vehicles)
            else:
                consistency["temporal_ordering"] = 1.0
            
            if consistency["temporal_ordering"] < 0.95:
                consistency["issues"].append("Temporal ordering issues detected")
        
        # Vehicle ID consistency
        if "vehicle_id" in df.columns:
            unique_vehicles = df["vehicle_id"].nunique()
            total_records = len(df)
            avg_records_per_vehicle = total_records / unique_vehicles if unique_vehicles > 0 else 0
            
            consistency["avg_records_per_vehicle"] = avg_records_per_vehicle
            consistency["unique_vehicles"] = unique_vehicles
        
        # Overall consistency score
        consistency_scores = [
            1.0 - consistency.get("duplicate_ratio", 0.0),
            consistency.get("temporal_ordering", 1.0)
        ]
        consistency["score"] = np.mean(consistency_scores)
        
        return consistency
    
    async def _assess_trajectory_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess trajectory-specific quality metrics."""
        trajectory_quality = {}
        trajectory_quality["issues"] = []
        
        if "vehicle_id" not in df.columns:
            trajectory_quality["score"] = 0.0
            trajectory_quality["issues"].append("No vehicle_id column for trajectory analysis")
            return trajectory_quality
        
        # Trajectory length analysis
        trajectory_lengths = df.groupby("vehicle_id").size()
        trajectory_quality["avg_trajectory_length"] = trajectory_lengths.mean()
        trajectory_quality["min_trajectory_length"] = trajectory_lengths.min()
        trajectory_quality["max_trajectory_length"] = trajectory_lengths.max()
        
        # Short trajectory detection
        short_trajectories = (trajectory_lengths < 10).mean()  # Less than 10 points
        trajectory_quality["short_trajectory_ratio"] = short_trajectories
        
        if short_trajectories > 0.2:  # > 20% short trajectories
            trajectory_quality["issues"].append(f"High ratio of short trajectories: {short_trajectories:.2%}")
        
        # Spatial coverage analysis
        if "x" in df.columns and "y" in df.columns:
            x_range = df["x"].max() - df["x"].min()
            y_range = df["y"].max() - df["y"].min()
            
            trajectory_quality["spatial_coverage_x"] = x_range
            trajectory_quality["spatial_coverage_y"] = y_range
            
            # Check for reasonable spatial distribution
            spatial_density = len(df) / (x_range * y_range) if x_range > 0 and y_range > 0 else 0
            trajectory_quality["spatial_density"] = spatial_density
        
        # Temporal coverage analysis
        if "timestamp" in df.columns:
            if df["timestamp"].dtype == "datetime64[ns]":
                time_span = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
                trajectory_quality["temporal_coverage_seconds"] = time_span
                
                # Average sampling rate
                total_records = len(df)
                avg_sampling_rate = total_records / time_span if time_span > 0 else 0
                trajectory_quality["avg_sampling_rate_hz"] = avg_sampling_rate
        
        # Physics consistency checks
        if all(col in df.columns for col in ["x", "y", "velocity", "timestamp"]):
            physics_consistency = await self._check_physics_consistency(df)
            trajectory_quality["physics_consistency"] = physics_consistency
        
        # Overall trajectory quality score
        quality_factors = []
        
        # Penalize short trajectories
        quality_factors.append(1.0 - trajectory_quality["short_trajectory_ratio"])
        
        # Reward good spatial coverage (heuristic)
        if "spatial_density" in trajectory_quality:
            spatial_score = min(1.0, trajectory_quality["spatial_density"] / 1000)  # Normalize
            quality_factors.append(spatial_score)
        
        # Physics consistency
        if "physics_consistency" in trajectory_quality:
            quality_factors.append(trajectory_quality["physics_consistency"]["score"])
        
        trajectory_quality["score"] = np.mean(quality_factors) if quality_factors else 0.5
        
        return trajectory_quality
    
    async def _check_physics_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check physics consistency in trajectory data."""
        physics = {}
        physics["issues"] = []
        
        inconsistent_count = 0
        total_checks = 0
        
        for vehicle_id, vehicle_df in df.groupby("vehicle_id"):
            vehicle_df = vehicle_df.sort_values("timestamp")
            
            if len(vehicle_df) < 2:
                continue
                
            for i in range(1, len(vehicle_df)):
                total_checks += 1
                
                # Calculate implied velocity from position
                dx = vehicle_df.iloc[i]["x"] - vehicle_df.iloc[i-1]["x"]
                dy = vehicle_df.iloc[i]["y"] - vehicle_df.iloc[i-1]["y"]
                distance = np.sqrt(dx**2 + dy**2)
                
                if vehicle_df.iloc[i]["timestamp"] != vehicle_df.iloc[i-1]["timestamp"]:
                    if df["timestamp"].dtype == "datetime64[ns]":
                        dt = (vehicle_df.iloc[i]["timestamp"] - vehicle_df.iloc[i-1]["timestamp"]).total_seconds()
                    else:
                        dt = vehicle_df.iloc[i]["timestamp"] - vehicle_df.iloc[i-1]["timestamp"]
                    
                    if dt > 0:
                        implied_velocity = distance / dt
                        reported_velocity = vehicle_df.iloc[i]["velocity"]
                        
                        # Check consistency (allow 20% tolerance)
                        if abs(implied_velocity - reported_velocity) > 0.2 * reported_velocity:
                            inconsistent_count += 1
        
        if total_checks > 0:
            physics["consistency_ratio"] = 1.0 - (inconsistent_count / total_checks)
            physics["score"] = physics["consistency_ratio"]
        else:
            physics["consistency_ratio"] = 1.0
            physics["score"] = 1.0
        
        physics["total_checks"] = total_checks
        physics["inconsistent_checks"] = inconsistent_count
        
        if physics["consistency_ratio"] < 0.8:
            physics["issues"].append("Physics inconsistencies detected between position and velocity")
        
        return physics
    
    def _generate_recommendations(self, assessments: Dict[str, Any]) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        # Completeness recommendations
        if assessments["completeness"]["score"] < 0.9:
            recommendations.append("Consider data imputation or filtering for missing values")
        
        # Validity recommendations
        if assessments["validity"]["score"] < 0.9:
            recommendations.append("Implement data validation and outlier detection")
        
        # Consistency recommendations
        if assessments["consistency"]["score"] < 0.9:
            recommendations.append("Add duplicate detection and temporal ordering checks")
        
        # Trajectory quality recommendations
        if assessments["trajectory_quality"]["score"] < 0.8:
            recommendations.append("Filter short trajectories and improve sampling consistency")
        
        # Overall score recommendations
        if assessments["overall_score"] < 0.8:
            recommendations.append("Consider comprehensive data cleaning pipeline")
        
        return recommendations


class QualityReporter:
    """Generate quality reports and visualizations."""
    
    def __init__(self, config: DictConfig):
        self.config = config
    
    async def generate_report(
        self, 
        quality_assessment: Dict[str, Any], 
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": quality_assessment["overall_score"],
            "summary": {
                "completeness": quality_assessment["completeness"]["score"],
                "validity": quality_assessment["validity"]["score"],
                "consistency": quality_assessment["consistency"]["score"],
                "trajectory_quality": quality_assessment["trajectory_quality"]["score"]
            },
            "issues": [],
            "recommendations": quality_assessment["recommendations"]
        }
        
        # Collect all issues
        for category in ["completeness", "validity", "consistency", "trajectory_quality"]:
            if "issues" in quality_assessment[category]:
                for issue in quality_assessment[category]["issues"]:
                    report["issues"].append(f"{category.title()}: {issue}")
        
        # Save report if path specified
        if output_path:
            import json
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Quality report saved to {output_path}")
        
        return report