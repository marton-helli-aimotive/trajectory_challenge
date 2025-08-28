"""Model registry for managing model versions and deployments."""

import os
import json
import pickle
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

# from ..core.config import Config, get_config  # Removed for now
from ..core.logging import get_logger, setup_logging
from ..core.models import TrajectoryPoint, Trajectory
from .tracking import MLflowTracker


@dataclass
class ModelVersion:
    """Model version information."""
    
    version: str
    model_name: str
    model_type: str
    file_path: str
    created_at: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    description: str
    status: str  # 'active', 'archived', 'deprecated'
    deployment_status: str  # 'deployed', 'staging', 'not_deployed'
    mlflow_run_id: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class DeploymentInfo:
    """Deployment information."""
    
    deployment_id: str
    model_version: str
    model_name: str
    deployed_at: str
    environment: str  # 'production', 'staging', 'development'
    status: str  # 'active', 'inactive', 'failed'
    endpoint_url: Optional[str] = None
    replicas: int = 1
    resources: Optional[Dict[str, Any]] = None
    health_checks: Optional[Dict[str, Any]] = None


class ModelRegistry:
    """Model registry for managing model versions and deployments."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize model registry.
        
        Args:
            config: Configuration object
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.tracker = MLflowTracker(config)
        
        # Set up registry paths
        self.registry_path = Path(self.config.get("data", {}).get("models_path", "models")) / "registry"
        self.versions_path = self.registry_path / "versions"
        self.deployments_path = self.registry_path / "deployments"
        self.metadata_path = self.registry_path / "metadata"
        
        # Create registry directories
        self._setup_registry()
        
        self.logger.info("Model registry initialized", registry_path=str(self.registry_path))
    
    def _setup_registry(self) -> None:
        """Set up registry directory structure."""
        try:
            self.registry_path.mkdir(parents=True, exist_ok=True)
            self.versions_path.mkdir(parents=True, exist_ok=True)
            self.deployments_path.mkdir(parents=True, exist_ok=True)
            self.metadata_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Registry directories created")
        except Exception as e:
            self.logger.error("Failed to setup registry", error=str(e))
            raise
    
    def register_model(self, 
                      model: Any,
                      model_name: str,
                      model_type: str,
                      version: str,
                      metrics: Dict[str, float],
                      parameters: Dict[str, Any],
                      description: str = "",
                      mlflow_run_id: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None) -> str:
        """Register a new model version.
        
        Args:
            model: The trained model
            model_name: Name of the model
            model_type: Type of model (sklearn, pytorch, etc.)
            version: Version string
            metrics: Performance metrics
            parameters: Model parameters
            description: Model description
            mlflow_run_id: MLflow run ID
            tags: Additional tags
            
        Returns:
            Version ID
        """
        try:
            # Create version directory
            version_dir = self.versions_path / model_name / version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model file
            model_file = version_dir / f"{model_name}_{version}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            
            # Create version metadata
            version_info = ModelVersion(
                version=version,
                model_name=model_name,
                model_type=model_type,
                file_path=str(model_file),
                created_at=datetime.now().isoformat(),
                metrics=metrics,
                parameters=parameters,
                description=description,
                status="active",
                deployment_status="not_deployed",
                mlflow_run_id=mlflow_run_id,
                tags=tags or {}
            )
            
            # Save version metadata
            metadata_file = version_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(asdict(version_info), f, indent=2, default=str)
            
            # Update model index
            self._update_model_index(model_name, version)
            
            self.logger.info("Model registered", 
                           model_name=model_name, 
                           version=version,
                           file_path=str(model_file))
            
            return version
            
        except Exception as e:
            self.logger.error("Failed to register model", error=str(e))
            raise
    
    def _update_model_index(self, model_name: str, version: str) -> None:
        """Update model index file."""
        try:
            index_file = self.metadata_path / f"{model_name}_index.json"
            
            if index_file.exists():
                with open(index_file, "r") as f:
                    index = json.load(f)
            else:
                index = {"model_name": model_name, "versions": []}
            
            # Add version if not already present
            if version not in index["versions"]:
                index["versions"].append(version)
                index["versions"].sort(reverse=True)  # Sort by version (assuming semantic versioning)
            
            # Update latest version
            index["latest_version"] = index["versions"][0]
            index["last_updated"] = datetime.now().isoformat()
            
            with open(index_file, "w") as f:
                json.dump(index, f, indent=2)
                
        except Exception as e:
            self.logger.error("Failed to update model index", error=str(e))
    
    def get_model_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get model version information.
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            Model version information
        """
        try:
            metadata_file = self.versions_path / model_name / version / "metadata.json"
            
            if not metadata_file.exists():
                self.logger.warning("Model version not found", model_name=model_name, version=version)
                return None
            
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            return ModelVersion(**metadata)
            
        except Exception as e:
            self.logger.error("Failed to get model version", error=str(e))
            return None
    
    def load_model(self, model_name: str, version: str) -> Any:
        """Load a model from registry.
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            Loaded model
        """
        try:
            version_info = self.get_model_version(model_name, version)
            if not version_info:
                raise ValueError(f"Model version not found: {model_name}:{version}")
            
            model_file = Path(version_info.file_path)
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            with open(model_file, "rb") as f:
                model = pickle.load(f)
            
            self.logger.info("Model loaded from registry", 
                           model_name=model_name, 
                           version=version)
            
            return model
            
        except Exception as e:
            self.logger.error("Failed to load model", error=str(e))
            raise
    
    def list_model_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of model versions
        """
        try:
            model_dir = self.versions_path / model_name
            if not model_dir.exists():
                return []
            
            versions = []
            for version_dir in model_dir.iterdir():
                if version_dir.is_dir():
                    metadata_file = version_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        versions.append(ModelVersion(**metadata))
            
            # Sort by creation date (newest first)
            versions.sort(key=lambda v: v.created_at, reverse=True)
            
            return versions
            
        except Exception as e:
            self.logger.error("Failed to list model versions", error=str(e))
            return []
    
    def list_models(self) -> List[str]:
        """List all registered models.
        
        Returns:
            List of model names
        """
        try:
            models = []
            for model_dir in self.versions_path.iterdir():
                if model_dir.is_dir():
                    models.append(model_dir.name)
            
            return sorted(models)
            
        except Exception as e:
            self.logger.error("Failed to list models", error=str(e))
            return []
    
    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest model version
        """
        try:
            versions = self.list_model_versions(model_name)
            return versions[0] if versions else None
            
        except Exception as e:
            self.logger.error("Failed to get latest version", error=str(e))
            return None
    
    def update_model_status(self, model_name: str, version: str, status: str) -> bool:
        """Update model version status.
        
        Args:
            model_name: Name of the model
            version: Version string
            status: New status
            
        Returns:
            True if successful
        """
        try:
            version_info = self.get_model_version(model_name, version)
            if not version_info:
                return False
            
            version_info.status = status
            
            # Save updated metadata
            metadata_file = self.versions_path / model_name / version / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(asdict(version_info), f, indent=2, default=str)
            
            self.logger.info("Model status updated", 
                           model_name=model_name, 
                           version=version, 
                           status=status)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to update model status", error=str(e))
            return False
    
    def deploy_model(self, 
                    model_name: str, 
                    version: str, 
                    environment: str = "production",
                    endpoint_url: Optional[str] = None,
                    replicas: int = 1,
                    resources: Optional[Dict[str, Any]] = None) -> str:
        """Deploy a model version.
        
        Args:
            model_name: Name of the model
            version: Version string
            environment: Deployment environment
            endpoint_url: Endpoint URL
            replicas: Number of replicas
            resources: Resource requirements
            
        Returns:
            Deployment ID
        """
        try:
            # Verify model version exists
            version_info = self.get_model_version(model_name, version)
            if not version_info:
                raise ValueError(f"Model version not found: {model_name}:{version}")
            
            # Generate deployment ID
            deployment_id = f"{model_name}-{version}-{environment}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Create deployment info
            deployment_info = DeploymentInfo(
                deployment_id=deployment_id,
                model_version=version,
                model_name=model_name,
                deployed_at=datetime.now().isoformat(),
                environment=environment,
                status="active",
                endpoint_url=endpoint_url,
                replicas=replicas,
                resources=resources or {},
                health_checks={}
            )
            
            # Save deployment info
            deployment_file = self.deployments_path / f"{deployment_id}.json"
            with open(deployment_file, "w") as f:
                json.dump(asdict(deployment_info), f, indent=2, default=str)
            
            # Update model version deployment status
            self.update_model_status(model_name, version, "active")
            
            # Update version metadata
            version_info.deployment_status = "deployed"
            metadata_file = self.versions_path / model_name / version / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(asdict(version_info), f, indent=2, default=str)
            
            self.logger.info("Model deployed", 
                           deployment_id=deployment_id,
                           model_name=model_name, 
                           version=version,
                           environment=environment)
            
            return deployment_id
            
        except Exception as e:
            self.logger.error("Failed to deploy model", error=str(e))
            raise
    
    def get_deployment_info(self, deployment_id: str) -> Optional[DeploymentInfo]:
        """Get deployment information.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            Deployment information
        """
        try:
            deployment_file = self.deployments_path / f"{deployment_id}.json"
            
            if not deployment_file.exists():
                return None
            
            with open(deployment_file, "r") as f:
                deployment_data = json.load(f)
            
            return DeploymentInfo(**deployment_data)
            
        except Exception as e:
            self.logger.error("Failed to get deployment info", error=str(e))
            return None
    
    def list_deployments(self, model_name: Optional[str] = None, environment: Optional[str] = None) -> List[DeploymentInfo]:
        """List deployments.
        
        Args:
            model_name: Filter by model name
            environment: Filter by environment
            
        Returns:
            List of deployments
        """
        try:
            deployments = []
            
            for deployment_file in self.deployments_path.glob("*.json"):
                with open(deployment_file, "r") as f:
                    deployment_data = json.load(f)
                
                deployment_info = DeploymentInfo(**deployment_data)
                
                # Apply filters
                if model_name and deployment_info.model_name != model_name:
                    continue
                if environment and deployment_info.environment != environment:
                    continue
                
                deployments.append(deployment_info)
            
            # Sort by deployment date (newest first)
            deployments.sort(key=lambda d: d.deployed_at, reverse=True)
            
            return deployments
            
        except Exception as e:
            self.logger.error("Failed to list deployments", error=str(e))
            return []
    
    def undeploy_model(self, deployment_id: str) -> bool:
        """Undeploy a model.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            True if successful
        """
        try:
            deployment_info = self.get_deployment_info(deployment_id)
            if not deployment_info:
                return False
            
            # Update deployment status
            deployment_info.status = "inactive"
            deployment_file = self.deployments_path / f"{deployment_id}.json"
            with open(deployment_file, "w") as f:
                json.dump(asdict(deployment_info), f, indent=2, default=str)
            
            # Update model version deployment status
            self.update_model_status(deployment_info.model_name, deployment_info.model_version, "active")
            
            # Update version metadata
            version_info = self.get_model_version(deployment_info.model_name, deployment_info.model_version)
            if version_info:
                version_info.deployment_status = "not_deployed"
                metadata_file = self.versions_path / deployment_info.model_name / deployment_info.model_version / "metadata.json"
                with open(metadata_file, "w") as f:
                    json.dump(asdict(version_info), f, indent=2, default=str)
            
            self.logger.info("Model undeployed", deployment_id=deployment_id)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to undeploy model", error=str(e))
            return False
    
    def delete_model_version(self, model_name: str, version: str) -> bool:
        """Delete a model version.
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            True if successful
        """
        try:
            version_dir = self.versions_path / model_name / version
            
            if not version_dir.exists():
                return False
            
            # Check if version is deployed
            version_info = self.get_model_version(model_name, version)
            if version_info and version_info.deployment_status == "deployed":
                raise ValueError(f"Cannot delete deployed model version: {model_name}:{version}")
            
            # Remove version directory
            shutil.rmtree(version_dir)
            
            # Update model index
            self._remove_from_index(model_name, version)
            
            self.logger.info("Model version deleted", model_name=model_name, version=version)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to delete model version", error=str(e))
            return False
    
    def _remove_from_index(self, model_name: str, version: str) -> None:
        """Remove version from model index."""
        try:
            index_file = self.metadata_path / f"{model_name}_index.json"
            
            if index_file.exists():
                with open(index_file, "r") as f:
                    index = json.load(f)
                
                if version in index["versions"]:
                    index["versions"].remove(version)
                
                if index["versions"]:
                    index["latest_version"] = index["versions"][0]
                else:
                    # Remove index file if no versions left
                    index_file.unlink()
                    return
                
                index["last_updated"] = datetime.now().isoformat()
                
                with open(index_file, "w") as f:
                    json.dump(index, f, indent=2)
                    
        except Exception as e:
            self.logger.error("Failed to remove from index", error=str(e))
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get registry summary.
        
        Returns:
            Registry summary
        """
        try:
            models = self.list_models()
            total_versions = 0
            total_deployments = 0
            
            model_summaries = {}
            for model_name in models:
                versions = self.list_model_versions(model_name)
                deployments = self.list_deployments(model_name=model_name)
                
                model_summaries[model_name] = {
                    "total_versions": len(versions),
                    "latest_version": versions[0].version if versions else None,
                    "active_deployments": len([d for d in deployments if d.status == "active"]),
                    "last_updated": versions[0].created_at if versions else None
                }
                
                total_versions += len(versions)
                total_deployments += len(deployments)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(models),
                "total_versions": total_versions,
                "total_deployments": total_deployments,
                "models": model_summaries
            }
            
        except Exception as e:
            self.logger.error("Failed to get registry summary", error=str(e))
            return {}
    
    def export_registry(self, export_path: str) -> None:
        """Export registry to a file.
        
        Args:
            export_path: Path to export file
        """
        try:
            summary = self.get_registry_summary()
            
            with open(export_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info("Registry exported", export_path=export_path)
            
        except Exception as e:
            self.logger.error("Failed to export registry", error=str(e))