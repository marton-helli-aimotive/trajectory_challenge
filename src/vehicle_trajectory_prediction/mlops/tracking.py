"""MLflow experiment tracking and model management."""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.entities import Experiment, Run
from mlflow.tracking import MlflowClient

# from ..core.config import Config  # Removed for now
from ..core.logging import get_logger
from ..core.models import TrajectoryPoint, Trajectory
# from ..evaluation.metrics import EvaluationMetrics  # Removed for now


class MLflowTracker:
    """MLflow experiment tracking and model management."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize MLflow tracker.
        
        Args:
            config: Configuration object with MLflow settings
        """
        self.logger = get_logger(__name__)
        self.config = config
        
        # Set up MLflow tracking URI
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Initialize MLflow client
        self.client = MlflowClient()
        
        # Set experiment name
        self.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "vehicle_trajectory_prediction")
        self._setup_experiment()
        
        self.logger.info("MLflow tracker initialized", 
                        tracking_uri=tracking_uri, 
                        experiment_name=self.experiment_name)
    
    def _setup_experiment(self) -> None:
        """Set up MLflow experiment."""
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(self.experiment_name)
                self.logger.info("Created new MLflow experiment", 
                               experiment_id=experiment_id,
                               experiment_name=self.experiment_name)
            else:
                self.logger.info("Using existing MLflow experiment", 
                               experiment_id=experiment.experiment_id,
                               experiment_name=self.experiment_name)
        except Exception as e:
            self.logger.error("Failed to setup MLflow experiment", error=str(e))
            raise
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Additional tags for the run
            
        Returns:
            Run ID
        """
        try:
            mlflow.set_experiment(self.experiment_name)
            
            # Set default tags
            default_tags = {
                "project": "vehicle_trajectory_prediction",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            }
            if tags:
                default_tags.update(tags)
            
            with mlflow.start_run(run_name=run_name, tags=default_tags) as run:
                run_id = run.info.run_id
                self.logger.info("Started MLflow run", run_id=run_id, run_name=run_name)
                return run_id
                
        except Exception as e:
            self.logger.error("Failed to start MLflow run", error=str(e))
            raise
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log parameters to current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            mlflow.log_params(params)
            self.logger.debug("Logged parameters", params=params)
        except Exception as e:
            self.logger.error("Failed to log parameters", error=str(e))
            raise
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to current run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            self.logger.debug("Logged metrics", metrics=metrics, step=step)
        except Exception as e:
            self.logger.error("Failed to log metrics", error=str(e))
            raise
    
    def log_model(self, model: Any, model_name: str, model_type: str = "sklearn") -> None:
        """Log model to MLflow.
        
        Args:
            model: The trained model to log
            model_name: Name for the model
            model_type: Type of model (sklearn, pytorch, xgboost)
        """
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, model_name)
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(model, model_name)
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(model, model_name)
            else:
                mlflow.log_model(model, model_name)
            
            self.logger.info("Logged model", model_name=model_name, model_type=model_type)
        except Exception as e:
            self.logger.error("Failed to log model", error=str(e), model_name=model_name)
            raise
    
    def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifacts to current run.
        
        Args:
            local_path: Local path to the artifact
            artifact_path: Path within the run's artifact directory
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            self.logger.debug("Logged artifact", local_path=local_path, artifact_path=artifact_path)
        except Exception as e:
            self.logger.error("Failed to log artifact", error=str(e), local_path=local_path)
            raise
    
    def log_evaluation_results(self, results: Dict[str, Any], model_name: str) -> None:
        """Log evaluation results for a model.
        
        Args:
            results: Evaluation results dictionary
            model_name: Name of the evaluated model
        """
        try:
            # Log metrics
            metrics = results.get("metrics", {})
            self.log_metrics(metrics)
            
            # Log model performance summary
            summary = {
                "model_name": model_name,
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_trajectories": results.get("total_trajectories", 0),
                "prediction_horizon": results.get("prediction_horizon", 0),
                "inference_time_avg": results.get("inference_time_avg", 0.0)
            }
            
            # Save detailed results as artifact
            results_file = f"evaluation_results_{model_name}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            self.log_artifacts(results_file, "evaluation_results")
            os.remove(results_file)  # Clean up temporary file
            
            self.logger.info("Logged evaluation results", model_name=model_name)
        except Exception as e:
            self.logger.error("Failed to log evaluation results", error=str(e), model_name=model_name)
            raise
    
    def log_training_data_info(self, data_info: Dict[str, Any]) -> None:
        """Log information about training data.
        
        Args:
            data_info: Dictionary containing data information
        """
        try:
            # Log data statistics as parameters
            self.log_parameters({
                "training_samples": data_info.get("total_samples", 0),
                "training_trajectories": data_info.get("total_trajectories", 0),
                "feature_count": data_info.get("feature_count", 0),
                "data_start_date": data_info.get("start_date", ""),
                "data_end_date": data_info.get("end_date", "")
            })
            
            # Log data distribution as metrics
            if "velocity_stats" in data_info:
                self.log_metrics({
                    "velocity_mean": data_info["velocity_stats"].get("mean", 0.0),
                    "velocity_std": data_info["velocity_stats"].get("std", 0.0),
                    "velocity_min": data_info["velocity_stats"].get("min", 0.0),
                    "velocity_max": data_info["velocity_stats"].get("max", 0.0)
                })
            
            self.logger.info("Logged training data info")
        except Exception as e:
            self.logger.error("Failed to log training data info", error=str(e))
            raise
    
    def get_best_model(self, metric_name: str = "rmse", experiment_name: Optional[str] = None) -> Optional[str]:
        """Get the best model based on a metric.
        
        Args:
            metric_name: Metric to optimize for
            experiment_name: Name of the experiment (defaults to current)
            
        Returns:
            Run ID of the best model, or None if no runs found
        """
        try:
            exp_name = experiment_name or self.experiment_name
            experiment = self.client.get_experiment_by_name(exp_name)
            
            if experiment is None:
                self.logger.warning("Experiment not found", experiment_name=exp_name)
                return None
            
            # Get all runs for the experiment
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric_name} ASC"]
            )
            
            if not runs:
                self.logger.warning("No runs found in experiment", experiment_name=exp_name)
                return None
            
            best_run = runs[0]
            self.logger.info("Found best model", 
                           run_id=best_run.info.run_id,
                           metric_name=metric_name,
                           metric_value=best_run.data.metrics.get(metric_name))
            
            return best_run.info.run_id
        except Exception as e:
            self.logger.error("Failed to get best model", error=str(e))
            return None
    
    def load_model(self, run_id: str, model_name: str) -> Any:
        """Load a model from MLflow.
        
        Args:
            run_id: MLflow run ID
            model_name: Name of the model within the run
            
        Returns:
            Loaded model
        """
        try:
            model_uri = f"runs:/{run_id}/{model_name}"
            model = mlflow.sklearn.load_model(model_uri)
            self.logger.info("Loaded model from MLflow", run_id=run_id, model_name=model_name)
            return model
        except Exception as e:
            self.logger.error("Failed to load model", error=str(e), run_id=run_id, model_name=model_name)
            raise
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments.
        
        Returns:
            List of experiment information
        """
        try:
            experiments = self.client.list_experiments()
            experiment_info = []
            
            for exp in experiments:
                experiment_info.append({
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage
                })
            
            return experiment_info
        except Exception as e:
            self.logger.error("Failed to list experiments", error=str(e))
            return []
    
    def list_runs(self, experiment_name: Optional[str] = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """List runs for an experiment.
        
        Args:
            experiment_name: Name of the experiment (defaults to current)
            max_results: Maximum number of runs to return
            
        Returns:
            List of run information
        """
        try:
            exp_name = experiment_name or self.experiment_name
            experiment = self.client.get_experiment_by_name(exp_name)
            
            if experiment is None:
                self.logger.warning("Experiment not found", experiment_name=exp_name)
                return []
            
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results
            )
            
            run_info = []
            for run in runs:
                run_info.append({
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params),
                    "tags": dict(run.data.tags)
                })
            
            return run_info
        except Exception as e:
            self.logger.error("Failed to list runs", error=str(e))
            return []
    
    def delete_run(self, run_id: str) -> bool:
        """Delete a run from MLflow.
        
        Args:
            run_id: MLflow run ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_run(run_id)
            self.logger.info("Deleted MLflow run", run_id=run_id)
            return True
        except Exception as e:
            self.logger.error("Failed to delete run", error=str(e), run_id=run_id)
            return False
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            self.logger.info("Ended MLflow run")
        except Exception as e:
            self.logger.error("Failed to end run", error=str(e))
            raise