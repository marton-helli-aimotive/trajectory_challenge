"""
Experiment tracking framework with MLflow and Weights & Biases integration.

This module provides comprehensive experiment tracking capabilities including:
- MLflow experiment management and logging
- Weights & Biases integration for advanced visualization
- Model artifact versioning and storage
- Hyperparameter optimization tracking
- Comparative experiment analysis
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime
import uuid

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from ..models.base import TrajectoryPredictor, PredictionResult
from ..data.schemas import TrajectoryData
from ..evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    experiment_name: str
    run_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tracking_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    use_mlflow: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


class ExperimentTracker:
    """
    Base experiment tracking interface.
    
    Provides unified interface for multiple tracking backends.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.experiment_config = None
        self.current_run_id = None
        
    async def start_experiment(self, experiment_config: ExperimentConfig) -> str:
        """Start a new experiment run."""
        raise NotImplementedError
        
    async def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Log experiment parameters."""
        raise NotImplementedError
        
    async def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        """Log metrics for current run."""
        raise NotImplementedError
        
    async def log_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Log artifacts (models, plots, data)."""
        raise NotImplementedError
        
    async def end_experiment(self, status: str = "FINISHED") -> None:
        """End current experiment run."""
        raise NotImplementedError


class MLflowTracker(ExperimentTracker):
    """
    MLflow-based experiment tracking implementation.
    
    Provides comprehensive MLflow integration for trajectory prediction experiments.
    """
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        
        # MLflow configuration
        self.mlflow_config = config.get("mlflow", {})
        self.tracking_uri = self.mlflow_config.get("tracking_uri", "file:./mlflow_tracking")
        self.registry_uri = self.mlflow_config.get("registry_uri", None)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
            
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        # Experiment state
        self.current_experiment_id = None
        self.current_run = None
        
        logger.info(f"MLflow tracker initialized with URI: {self.tracking_uri}")
    
    async def start_experiment(self, experiment_config: ExperimentConfig) -> str:
        """Start MLflow experiment run."""
        
        self.experiment_config = experiment_config
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_config.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=experiment_config.experiment_name,
                    artifact_location=experiment_config.artifact_location
                )
            else:
                experiment_id = experiment.experiment_id
                
            self.current_experiment_id = experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create/get MLflow experiment: {e}")
            raise
        
        # Start run
        run_name = experiment_config.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.current_run = mlflow.start_run(
                experiment_id=self.current_experiment_id,
                run_name=run_name,
                tags=experiment_config.tags
            )
            
            self.current_run_id = self.current_run.info.run_id
            
            # Log initial parameters
            if experiment_config.parameters:
                await self.log_parameters(experiment_config.parameters)
                
            logger.info(f"Started MLflow run: {run_name} (ID: {self.current_run_id})")
            return self.current_run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise
    
    async def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        
        if not self.current_run:
            logger.warning("No active MLflow run to log parameters")
            return
            
        try:
            # Flatten nested parameters
            flattened_params = self._flatten_params(parameters)
            
            # MLflow has parameter value length limits
            for key, value in flattened_params.items():
                str_value = str(value)
                if len(str_value) > 250:  # MLflow parameter limit
                    str_value = str_value[:247] + "..."
                    
                mlflow.log_param(key, str_value)
                
        except Exception as e:
            logger.error(f"Failed to log parameters to MLflow: {e}")
    
    async def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        
        if not self.current_run:
            logger.warning("No active MLflow run to log metrics")
            return
            
        try:
            # Flatten nested metrics
            flattened_metrics = self._flatten_params(metrics)
            
            for key, value in flattened_metrics.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    mlflow.log_metric(key, value, step=step)
                    
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")
    
    async def log_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Log artifacts to MLflow."""
        
        if not self.current_run:
            logger.warning("No active MLflow run to log artifacts")
            return
            
        try:
            for artifact_name, artifact_data in artifacts.items():
                
                if isinstance(artifact_data, (str, Path)) and Path(artifact_data).exists():
                    # Log file artifact
                    mlflow.log_artifact(str(artifact_data), artifact_path=artifact_name)
                    
                elif hasattr(artifact_data, 'save'):
                    # Log model artifact
                    temp_path = f"temp_{artifact_name}_{uuid.uuid4().hex[:8]}"
                    artifact_data.save(temp_path)
                    mlflow.log_artifact(temp_path, artifact_path=artifact_name)
                    
                    # Cleanup
                    if Path(temp_path).exists():
                        Path(temp_path).unlink()
                        
                elif isinstance(artifact_data, dict):
                    # Log JSON artifact
                    temp_file = f"temp_{artifact_name}.json"
                    with open(temp_file, 'w') as f:
                        json.dump(artifact_data, f, indent=2, default=str)
                    mlflow.log_artifact(temp_file, artifact_path=artifact_name)
                    Path(temp_file).unlink()
                    
                elif isinstance(artifact_data, pd.DataFrame):
                    # Log DataFrame as CSV
                    temp_file = f"temp_{artifact_name}.csv"
                    artifact_data.to_csv(temp_file, index=False)
                    mlflow.log_artifact(temp_file, artifact_path=artifact_name)
                    Path(temp_file).unlink()
                    
        except Exception as e:
            logger.error(f"Failed to log artifacts to MLflow: {e}")
    
    async def log_model(
        self,
        model: TrajectoryPredictor,
        model_name: str = "trajectory_model",
        registered_model_name: Optional[str] = None
    ) -> None:
        """Log trajectory prediction model to MLflow."""
        
        if not self.current_run:
            logger.warning("No active MLflow run to log model")
            return
            
        try:
            # Create temporary model directory
            temp_model_dir = f"temp_model_{uuid.uuid4().hex[:8]}"
            Path(temp_model_dir).mkdir(exist_ok=True)
            
            # Save model artifacts
            model_path = Path(temp_model_dir) / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Create MLmodel file
            mlmodel_content = f"""
artifact_path: {model_name}
flavors:
  python_function:
    env: conda.yaml
    loader_module: trajectory_prediction.models.loader
    python_version: 3.9.0
model_uuid: {uuid.uuid4()}
run_id: {self.current_run_id}
signature:
  inputs: '[{{"name": "trajectory", "type": "object"}}]'
  outputs: '[{{"name": "prediction", "type": "object"}}]'
"""
            
            mlmodel_path = Path(temp_model_dir) / "MLmodel"
            with open(mlmodel_path, 'w') as f:
                f.write(mlmodel_content)
            
            # Log the model
            mlflow.log_artifacts(temp_model_dir, artifact_path=model_name)
            
            # Register model if requested
            if registered_model_name:
                model_uri = f"runs:/{self.current_run_id}/{model_name}"
                mlflow.register_model(model_uri, registered_model_name)
                logger.info(f"Registered model: {registered_model_name}")
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_model_dir)
            
            logger.info(f"Logged model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}")
    
    async def end_experiment(self, status: str = "FINISHED") -> None:
        """End MLflow run."""
        
        if self.current_run:
            try:
                mlflow.end_run(status=status)
                logger.info(f"Ended MLflow run: {self.current_run_id} with status: {status}")
                
                self.current_run = None
                self.current_run_id = None
                
            except Exception as e:
                logger.error(f"Failed to end MLflow run: {e}")
    
    def _flatten_params(self, params: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested parameter dictionary."""
        
        flattened = {}
        
        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(self._flatten_params(value, full_key))
            elif isinstance(value, (list, tuple, np.ndarray)):
                # Convert to string representation for complex types
                flattened[full_key] = str(value)
            else:
                flattened[full_key] = value
                
        return flattened
    
    async def get_experiment_runs(self, experiment_name: str) -> List[Dict[str, Any]]:
        """Get all runs for an experiment."""
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                return []
            
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            return runs.to_dict('records') if not runs.empty else []
            
        except Exception as e:
            logger.error(f"Failed to get experiment runs: {e}")
            return []
    
    async def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs."""
        
        try:
            comparison_data = {
                "run_ids": run_ids,
                "runs": [],
                "metrics_comparison": {},
                "parameters_comparison": {}
            }
            
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                
                run_data = {
                    "run_id": run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                }
                
                comparison_data["runs"].append(run_data)
            
            # Extract common metrics and parameters
            if comparison_data["runs"]:
                all_metrics = set()
                all_params = set()
                
                for run_data in comparison_data["runs"]:
                    all_metrics.update(run_data["metrics"].keys())
                    all_params.update(run_data["params"].keys())
                
                # Create comparison matrices
                for metric in all_metrics:
                    comparison_data["metrics_comparison"][metric] = [
                        run_data["metrics"].get(metric, None) 
                        for run_data in comparison_data["runs"]
                    ]
                
                for param in all_params:
                    comparison_data["parameters_comparison"][param] = [
                        run_data["params"].get(param, None)
                        for run_data in comparison_data["runs"]
                    ]
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            return {}


class WandbTracker(ExperimentTracker):
    """
    Weights & Biases experiment tracking implementation.
    
    Provides advanced visualization and experiment tracking capabilities.
    """
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        
        if not WANDB_AVAILABLE:
            raise ImportError("Weights & Biases not available. Install with: pip install wandb")
        
        # W&B configuration
        self.wandb_config = config.get("wandb", {})
        self.project_name = self.wandb_config.get("project", "trajectory-prediction")
        self.entity = self.wandb_config.get("entity", None)
        
        # Initialize W&B
        if not wandb.api.api_key:
            logger.warning("W&B API key not found. Please run 'wandb login' first.")
        
        self.current_run = None
        
        logger.info(f"W&B tracker initialized for project: {self.project_name}")
    
    async def start_experiment(self, experiment_config: ExperimentConfig) -> str:
        """Start W&B experiment run."""
        
        self.experiment_config = experiment_config
        
        try:
            # Configure run
            config = experiment_config.parameters.copy() if experiment_config.parameters else {}
            
            # Start W&B run
            self.current_run = wandb.init(
                project=experiment_config.wandb_project or self.project_name,
                entity=experiment_config.wandb_entity or self.entity,
                name=experiment_config.run_name,
                tags=list(experiment_config.tags.values()) if experiment_config.tags else None,
                config=config,
                reinit=True
            )
            
            self.current_run_id = self.current_run.id
            
            logger.info(f"Started W&B run: {experiment_config.run_name} (ID: {self.current_run_id})")
            return self.current_run_id
            
        except Exception as e:
            logger.error(f"Failed to start W&B run: {e}")
            raise
    
    async def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Log parameters to W&B."""
        
        if not self.current_run:
            logger.warning("No active W&B run to log parameters")
            return
            
        try:
            # W&B handles nested configs well
            wandb.config.update(parameters)
            
        except Exception as e:
            logger.error(f"Failed to log parameters to W&B: {e}")
    
    async def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        
        if not self.current_run:
            logger.warning("No active W&B run to log metrics")
            return
            
        try:
            # Flatten and filter metrics
            flattened_metrics = self._flatten_params(metrics)
            
            valid_metrics = {
                key: value for key, value in flattened_metrics.items()
                if isinstance(value, (int, float)) and np.isfinite(value)
            }
            
            if valid_metrics:
                log_data = valid_metrics.copy()
                if step is not None:
                    log_data["step"] = step
                    
                wandb.log(log_data)
                
        except Exception as e:
            logger.error(f"Failed to log metrics to W&B: {e}")
    
    async def log_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Log artifacts to W&B."""
        
        if not self.current_run:
            logger.warning("No active W&B run to log artifacts")
            return
            
        try:
            for artifact_name, artifact_data in artifacts.items():
                
                if isinstance(artifact_data, (str, Path)) and Path(artifact_data).exists():
                    # Log file artifact
                    wandb.save(str(artifact_data))
                    
                elif isinstance(artifact_data, dict):
                    # Log as W&B Table or JSON
                    wandb.log({artifact_name: artifact_data})
                    
                elif isinstance(artifact_data, pd.DataFrame):
                    # Log DataFrame as W&B Table
                    wandb.log({artifact_name: wandb.Table(dataframe=artifact_data)})
                    
        except Exception as e:
            logger.error(f"Failed to log artifacts to W&B: {e}")
    
    async def end_experiment(self, status: str = "FINISHED") -> None:
        """End W&B run."""
        
        if self.current_run:
            try:
                wandb.finish()
                logger.info(f"Ended W&B run: {self.current_run_id}")
                
                self.current_run = None
                self.current_run_id = None
                
            except Exception as e:
                logger.error(f"Failed to end W&B run: {e}")
    
    def _flatten_params(self, params: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested parameter dictionary."""
        
        flattened = {}
        
        for key, value in params.items():
            full_key = f"{prefix}/{key}" if prefix else key  # W&B uses / for nesting
            
            if isinstance(value, dict):
                flattened.update(self._flatten_params(value, full_key))
            elif isinstance(value, (list, tuple, np.ndarray)):
                # W&B can handle lists better than MLflow
                if len(str(value)) < 1000:  # Reasonable size limit
                    flattened[full_key] = value
                else:
                    flattened[full_key] = str(value)[:1000] + "..."
            else:
                flattened[full_key] = value
                
        return flattened


class ExperimentManager:
    """
    High-level experiment management interface.
    
    Orchestrates multiple tracking backends and provides experiment lifecycle management.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # Initialize trackers
        self.trackers = []
        
        if config.get("mlflow", {}).get("enabled", True):
            self.trackers.append(MLflowTracker(config))
            
        if config.get("wandb", {}).get("enabled", False) and WANDB_AVAILABLE:
            self.trackers.append(WandbTracker(config))
        
        if not self.trackers:
            logger.warning("No experiment trackers configured")
        
        # Experiment state
        self.current_experiment = None
        self.model_evaluator = ModelEvaluator(config)
    
    async def run_tracked_experiment(
        self,
        experiment_config: ExperimentConfig,
        model: TrajectoryPredictor,
        train_data: List[TrajectoryData],
        test_data: List[TrajectoryData],
        prediction_horizons: List[float] = [5.0, 10.0, 15.0]
    ) -> Dict[str, Any]:
        """
        Run complete tracked experiment with model training and evaluation.
        
        Args:
            experiment_config: Experiment configuration
            model: Model to train and evaluate
            train_data: Training trajectories
            test_data: Testing trajectories  
            prediction_horizons: List of prediction horizons to evaluate
            
        Returns:
            Experiment results with tracking information
        """
        
        experiment_results = {
            "experiment_name": experiment_config.experiment_name,
            "run_name": experiment_config.run_name,
            "model_name": model.name,
            "start_time": datetime.now().isoformat(),
            "run_ids": {},
            "evaluation_results": {},
            "artifacts": {}
        }
        
        try:
            # Start tracking across all backends
            for tracker in self.trackers:
                run_id = await tracker.start_experiment(experiment_config)
                tracker_name = tracker.__class__.__name__
                experiment_results["run_ids"][tracker_name] = run_id
            
            # Log model configuration
            model_config = {
                "model_name": model.name,
                "model_type": model.__class__.__name__,
                "train_size": len(train_data),
                "test_size": len(test_data),
                "prediction_horizons": prediction_horizons
            }
            
            for tracker in self.trackers:
                await tracker.log_parameters(model_config)
            
            # Train model if needed
            logger.info("Training model...")
            if hasattr(model, 'fit') and hasattr(model, 'is_trained') and not model.is_trained:
                await model.fit(train_data)
            
            # Evaluate model at different horizons
            evaluation_results = {}
            
            for horizon in prediction_horizons:
                logger.info(f"Evaluating model at {horizon}s horizon...")
                
                eval_result = await self.model_evaluator.evaluate_model(
                    model, test_data, horizon
                )
                evaluation_results[f"horizon_{horizon}s"] = eval_result
                
                # Log metrics to trackers
                horizon_metrics = {}
                
                for category in ["trajectory_metrics", "safety_metrics", "probabilistic_metrics"]:
                    if category in eval_result:
                        for metric_name, metric_data in eval_result[category].items():
                            if isinstance(metric_data, dict) and "mean" in metric_data:
                                horizon_metrics[f"{category}_{metric_name}_mean"] = metric_data["mean"]
                                horizon_metrics[f"{category}_{metric_name}_std"] = metric_data.get("std", 0)
                
                # Add horizon prefix to metrics
                prefixed_metrics = {f"h{horizon}s_{k}": v for k, v in horizon_metrics.items()}
                
                for tracker in self.trackers:
                    await tracker.log_metrics(prefixed_metrics)
            
            experiment_results["evaluation_results"] = evaluation_results
            
            # Log model artifacts
            for tracker in self.trackers:
                if hasattr(tracker, 'log_model'):
                    await tracker.log_model(model, registered_model_name=f"{model.name}_registered")
            
            # Log evaluation artifacts
            artifacts = {
                "evaluation_summary": evaluation_results,
                "model_config": model_config
            }
            
            for tracker in self.trackers:
                await tracker.log_artifacts(artifacts)
            
            experiment_results["artifacts"] = artifacts
            experiment_results["status"] = "SUCCESS"
            
            logger.info("Experiment completed successfully")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            experiment_results["status"] = "FAILED"
            experiment_results["error"] = str(e)
            
            # End runs with failed status
            for tracker in self.trackers:
                await tracker.end_experiment(status="FAILED")
            
            raise
        
        finally:
            # End all tracking runs
            for tracker in self.trackers:
                await tracker.end_experiment(
                    status="FINISHED" if experiment_results.get("status") == "SUCCESS" else "FAILED"
                )
            
            experiment_results["end_time"] = datetime.now().isoformat()
        
        return experiment_results
    
    async def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        
        comparison_results = {
            "experiments": experiment_names,
            "comparison_data": {},
            "best_models": {},
            "summary": {}
        }
        
        try:
            # Use MLflow tracker for comparison (most comprehensive)
            mlflow_tracker = next(
                (t for t in self.trackers if isinstance(t, MLflowTracker)), 
                None
            )
            
            if not mlflow_tracker:
                logger.warning("MLflow tracker not available for experiment comparison")
                return comparison_results
            
            # Get runs for each experiment
            all_runs = []
            
            for exp_name in experiment_names:
                exp_runs = await mlflow_tracker.get_experiment_runs(exp_name)
                for run in exp_runs:
                    run["experiment_name"] = exp_name
                    all_runs.append(run)
            
            comparison_results["comparison_data"] = all_runs
            
            # Find best models by key metrics
            if all_runs:
                metrics_to_compare = ["h10.0s_trajectory_metrics_ade_mean", "h10.0s_safety_metrics_collision_risk_mean"]
                
                for metric in metrics_to_compare:
                    metric_values = []
                    for run in all_runs:
                        if f"metrics.{metric}" in run:
                            metric_values.append({
                                "run_id": run.get("run_id", "unknown"),
                                "experiment": run.get("experiment_name", "unknown"),
                                "value": run[f"metrics.{metric}"]
                            })
                    
                    if metric_values:
                        # For most metrics, lower is better
                        best_run = min(metric_values, key=lambda x: x["value"])
                        comparison_results["best_models"][metric] = best_run
            
            logger.info(f"Compared {len(experiment_names)} experiments with {len(all_runs)} total runs")
            
        except Exception as e:
            logger.error(f"Failed to compare experiments: {e}")
            comparison_results["error"] = str(e)
        
        return comparison_results