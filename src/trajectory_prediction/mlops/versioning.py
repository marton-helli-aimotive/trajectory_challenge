"""
Model versioning and artifact management system.

This module provides:
- Semantic model versioning
- Model registry with metadata
- Artifact storage and retrieval
- Model lifecycle management
- Deployment tracking and rollback capabilities
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import logging
from pathlib import Path
import json
import pickle
import hashlib
from datetime import datetime
import shutil
import uuid

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..models.base import TrajectoryPredictor
from ..data.schemas import TrajectoryData

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Model version information."""
    model_name: str
    version: str
    stage: ModelStage
    created_at: str
    created_by: str
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_hash: Optional[str] = None
    parent_version: Optional[str] = None


@dataclass 
class DeploymentRecord:
    """Deployment tracking record."""
    model_name: str
    version: str
    deployment_id: str
    environment: str
    deployed_at: str
    deployed_by: str
    status: str
    endpoint_url: Optional[str] = None
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class ArtifactStore:
    """
    Artifact storage management system.
    
    Handles storage, retrieval, and versioning of model artifacts.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.storage_config = config.get("artifact_storage", {})
        
        # Storage backends
        self.local_storage_path = Path(self.storage_config.get("local_path", "artifacts/models"))
        self.local_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Storage metadata
        self.metadata_file = self.local_storage_path / "artifacts_metadata.json"
        self.load_metadata()
        
        logger.info(f"Artifact store initialized at: {self.local_storage_path}")
    
    def load_metadata(self) -> None:
        """Load artifact metadata from storage."""
        
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load artifact metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def save_metadata(self) -> None:
        """Save artifact metadata to storage."""
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save artifact metadata: {e}")
    
    async def store_artifact(
        self,
        artifact_id: str,
        artifact_data: Any,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store artifact and return storage path.
        
        Args:
            artifact_id: Unique artifact identifier
            artifact_data: Artifact to store (model, data, etc.)
            artifact_type: Type of artifact (model, data, config, etc.)
            metadata: Additional metadata
            
        Returns:
            Storage path of the artifact
        """
        
        try:
            # Create artifact directory
            artifact_dir = self.local_storage_path / artifact_type / artifact_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            
            storage_path = None
            
            if isinstance(artifact_data, TrajectoryPredictor):
                # Store model
                model_path = artifact_dir / "model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(artifact_data, f)
                storage_path = str(model_path)
                
                # Store model metadata
                model_metadata = {
                    "name": artifact_data.name,
                    "type": artifact_data.__class__.__name__,
                    "is_trained": getattr(artifact_data, 'is_trained', False)
                }
                
                metadata_path = artifact_dir / "model_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(model_metadata, f, indent=2)
                    
            elif isinstance(artifact_data, dict):
                # Store dictionary as JSON
                json_path = artifact_dir / f"{artifact_id}.json"
                with open(json_path, 'w') as f:
                    json.dump(artifact_data, f, indent=2, default=str)
                storage_path = str(json_path)
                
            elif isinstance(artifact_data, pd.DataFrame):
                # Store DataFrame as Parquet
                parquet_path = artifact_dir / f"{artifact_id}.parquet"
                artifact_data.to_parquet(parquet_path)
                storage_path = str(parquet_path)
                
            elif isinstance(artifact_data, (str, Path)) and Path(artifact_data).exists():
                # Copy existing file
                source_path = Path(artifact_data)
                dest_path = artifact_dir / source_path.name
                shutil.copy2(source_path, dest_path)
                storage_path = str(dest_path)
                
            else:
                # Store as pickle
                pickle_path = artifact_dir / f"{artifact_id}.pkl"
                with open(pickle_path, 'wb') as f:
                    pickle.dump(artifact_data, f)
                storage_path = str(pickle_path)
            
            # Update metadata
            artifact_metadata = {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "storage_path": storage_path,
                "created_at": datetime.now().isoformat(),
                "size_bytes": Path(storage_path).stat().st_size if storage_path else 0,
                "metadata": metadata or {}
            }
            
            self.metadata[artifact_id] = artifact_metadata
            self.save_metadata()
            
            logger.info(f"Stored artifact: {artifact_id} at {storage_path}")
            return storage_path
            
        except Exception as e:
            logger.error(f"Failed to store artifact {artifact_id}: {e}")
            raise
    
    async def retrieve_artifact(self, artifact_id: str) -> Any:
        """Retrieve artifact by ID."""
        
        if artifact_id not in self.metadata:
            raise ValueError(f"Artifact not found: {artifact_id}")
        
        try:
            artifact_metadata = self.metadata[artifact_id]
            storage_path = Path(artifact_metadata["storage_path"])
            
            if not storage_path.exists():
                raise FileNotFoundError(f"Artifact file not found: {storage_path}")
            
            artifact_type = artifact_metadata["artifact_type"]
            
            if artifact_type == "model" and storage_path.suffix == ".pkl":
                # Load pickled model
                with open(storage_path, 'rb') as f:
                    return pickle.load(f)
                    
            elif storage_path.suffix == ".json":
                # Load JSON
                with open(storage_path, 'r') as f:
                    return json.load(f)
                    
            elif storage_path.suffix == ".parquet":
                # Load DataFrame
                return pd.read_parquet(storage_path)
                
            elif storage_path.suffix == ".pkl":
                # Load pickle
                with open(storage_path, 'rb') as f:
                    return pickle.load(f)
                    
            else:
                # Return file path for other types
                return str(storage_path)
                
        except Exception as e:
            logger.error(f"Failed to retrieve artifact {artifact_id}: {e}")
            raise
    
    async def delete_artifact(self, artifact_id: str) -> bool:
        """Delete artifact and its metadata."""
        
        if artifact_id not in self.metadata:
            logger.warning(f"Artifact not found for deletion: {artifact_id}")
            return False
        
        try:
            artifact_metadata = self.metadata[artifact_id]
            storage_path = Path(artifact_metadata["storage_path"])
            
            # Delete file
            if storage_path.exists():
                storage_path.unlink()
            
            # Delete directory if empty
            if storage_path.parent.exists() and not any(storage_path.parent.iterdir()):
                storage_path.parent.rmdir()
            
            # Remove metadata
            del self.metadata[artifact_id]
            self.save_metadata()
            
            logger.info(f"Deleted artifact: {artifact_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete artifact {artifact_id}: {e}")
            return False
    
    def list_artifacts(self, artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all artifacts, optionally filtered by type."""
        
        artifacts = []
        
        for artifact_id, metadata in self.metadata.items():
            if artifact_type is None or metadata["artifact_type"] == artifact_type:
                artifacts.append({
                    "artifact_id": artifact_id,
                    **metadata
                })
        
        return sorted(artifacts, key=lambda x: x["created_at"], reverse=True)


class ModelRegistry:
    """
    Model registry for version management and lifecycle tracking.
    
    Provides semantic versioning, stage management, and deployment tracking.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.artifact_store = ArtifactStore(config)
        
        # Registry storage
        self.registry_path = Path(config.get("model_registry", {}).get("path", "model_registry"))
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Registry files
        self.versions_file = self.registry_path / "model_versions.json"
        self.deployments_file = self.registry_path / "deployments.json"
        
        # Load existing data
        self.load_registry_data()
        
        logger.info(f"Model registry initialized at: {self.registry_path}")
    
    def load_registry_data(self) -> None:
        """Load registry data from storage."""
        
        # Load model versions
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    versions_data = json.load(f)
                    self.model_versions = {
                        model_name: [ModelVersion(**v) for v in versions]
                        for model_name, versions in versions_data.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load model versions: {e}")
                self.model_versions = {}
        else:
            self.model_versions = {}
        
        # Load deployment records
        if self.deployments_file.exists():
            try:
                with open(self.deployments_file, 'r') as f:
                    deployments_data = json.load(f)
                    self.deployments = [DeploymentRecord(**d) for d in deployments_data]
            except Exception as e:
                logger.warning(f"Failed to load deployment records: {e}")
                self.deployments = []
        else:
            self.deployments = []
    
    def save_registry_data(self) -> None:
        """Save registry data to storage."""
        
        try:
            # Save model versions
            versions_data = {}
            for model_name, versions in self.model_versions.items():
                versions_data[model_name] = [
                    {
                        **v.__dict__,
                        "stage": v.stage.value,  # Convert enum to string
                        "created_at": v.created_at
                    }
                    for v in versions
                ]
            
            with open(self.versions_file, 'w') as f:
                json.dump(versions_data, f, indent=2, default=str)
            
            # Save deployment records
            deployments_data = [
                {
                    **d.__dict__,
                    "deployed_at": d.deployed_at
                }
                for d in self.deployments
            ]
            
            with open(self.deployments_file, 'w') as f:
                json.dump(deployments_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save registry data: {e}")
    
    def _calculate_model_hash(self, model: TrajectoryPredictor) -> str:
        """Calculate hash of model for versioning."""
        
        try:
            # Create a reproducible hash of the model
            model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.sha256(model_bytes).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Failed to calculate model hash: {e}")
            return str(uuid.uuid4())[:16]
    
    def _generate_version_number(self, model_name: str, version_type: str = "patch") -> str:
        """Generate next version number."""
        
        if model_name not in self.model_versions or not self.model_versions[model_name]:
            return "1.0.0"
        
        # Get latest version
        latest_version = max(
            self.model_versions[model_name],
            key=lambda v: [int(x) for x in v.version.split('.')]
        )
        
        major, minor, patch = [int(x) for x in latest_version.version.split('.')]
        
        if version_type == "major":
            return f"{major + 1}.0.0"
        elif version_type == "minor":
            return f"{major}.{minor + 1}.0"
        else:  # patch
            return f"{major}.{minor}.{patch + 1}"
    
    async def register_model(
        self,
        model: TrajectoryPredictor,
        model_name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        version_type: str = "patch",
        created_by: str = "system"
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model: Trajectory prediction model
            model_name: Name for the model
            description: Version description
            tags: Model tags
            metrics: Performance metrics
            version_type: Version increment type (major, minor, patch)
            created_by: User/system that created the version
            
        Returns:
            ModelVersion object
        """
        
        try:
            # Generate version number
            version_number = self._generate_version_number(model_name, version_type)
            
            # Calculate model hash
            model_hash = self._calculate_model_hash(model)
            
            # Check for duplicate model
            if model_name in self.model_versions:
                for existing_version in self.model_versions[model_name]:
                    if existing_version.model_hash == model_hash:
                        logger.warning(f"Model with same hash already exists: {existing_version.version}")
                        return existing_version
            
            # Store model artifact
            artifact_id = f"{model_name}_v{version_number}"
            storage_path = await self.artifact_store.store_artifact(
                artifact_id=artifact_id,
                artifact_data=model,
                artifact_type="model",
                metadata={"model_name": model_name, "version": version_number}
            )
            
            # Create model version
            model_version = ModelVersion(
                model_name=model_name,
                version=version_number,
                stage=ModelStage.DEVELOPMENT,
                created_at=datetime.now().isoformat(),
                created_by=created_by,
                description=description,
                tags=tags or {},
                metrics=metrics or {},
                artifacts={"model": storage_path},
                model_hash=model_hash
            )
            
            # Add to registry
            if model_name not in self.model_versions:
                self.model_versions[model_name] = []
            
            self.model_versions[model_name].append(model_version)
            self.save_registry_data()
            
            logger.info(f"Registered model: {model_name} v{version_number}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise
    
    async def get_model(self, model_name: str, version: Optional[str] = None) -> TrajectoryPredictor:
        """Get model by name and version."""
        
        if model_name not in self.model_versions:
            raise ValueError(f"Model not found: {model_name}")
        
        model_versions = self.model_versions[model_name]
        
        if version is None:
            # Get latest version
            target_version = max(
                model_versions,
                key=lambda v: [int(x) for x in v.version.split('.')]
            )
        else:
            # Get specific version
            target_version = next(
                (v for v in model_versions if v.version == version),
                None
            )
            
            if target_version is None:
                raise ValueError(f"Version not found: {model_name} v{version}")
        
        # Retrieve model artifact
        artifact_id = f"{model_name}_v{target_version.version}"
        model = await self.artifact_store.retrieve_artifact(artifact_id)
        
        return model
    
    def list_models(self) -> Dict[str, List[ModelVersion]]:
        """List all registered models."""
        return self.model_versions.copy()
    
    def list_model_versions(self, model_name: str) -> List[ModelVersion]:
        """List versions of a specific model."""
        return self.model_versions.get(model_name, [])
    
    async def promote_model(
        self,
        model_name: str,
        version: str,
        target_stage: ModelStage,
        promoted_by: str = "system"
    ) -> ModelVersion:
        """Promote model to different stage."""
        
        if model_name not in self.model_versions:
            raise ValueError(f"Model not found: {model_name}")
        
        # Find the version
        target_version = next(
            (v for v in self.model_versions[model_name] if v.version == version),
            None
        )
        
        if target_version is None:
            raise ValueError(f"Version not found: {model_name} v{version}")
        
        # Update stage
        old_stage = target_version.stage
        target_version.stage = target_stage
        
        # Add promotion metadata
        if "promotions" not in target_version.metadata:
            target_version.metadata["promotions"] = []
        
        target_version.metadata["promotions"].append({
            "from_stage": old_stage.value,
            "to_stage": target_stage.value,
            "promoted_at": datetime.now().isoformat(),
            "promoted_by": promoted_by
        })
        
        self.save_registry_data()
        
        logger.info(f"Promoted {model_name} v{version} from {old_stage.value} to {target_stage.value}")
        return target_version
    
    async def archive_model_version(
        self,
        model_name: str,
        version: str,
        archived_by: str = "system"
    ) -> ModelVersion:
        """Archive a model version."""
        return await self.promote_model(model_name, version, ModelStage.ARCHIVED, archived_by)
    
    def get_production_models(self) -> Dict[str, ModelVersion]:
        """Get all models currently in production stage."""
        
        production_models = {}
        
        for model_name, versions in self.model_versions.items():
            production_versions = [v for v in versions if v.stage == ModelStage.PRODUCTION]
            
            if production_versions:
                # Get the latest production version
                latest_production = max(
                    production_versions,
                    key=lambda v: [int(x) for x in v.version.split('.')]
                )
                production_models[model_name] = latest_production
        
        return production_models
    
    async def record_deployment(
        self,
        model_name: str,
        version: str,
        environment: str,
        deployed_by: str = "system",
        endpoint_url: Optional[str] = None,
        deployment_config: Optional[Dict[str, Any]] = None
    ) -> DeploymentRecord:
        """Record a model deployment."""
        
        deployment_record = DeploymentRecord(
            model_name=model_name,
            version=version,
            deployment_id=str(uuid.uuid4()),
            environment=environment,
            deployed_at=datetime.now().isoformat(),
            deployed_by=deployed_by,
            status="active",
            endpoint_url=endpoint_url,
            deployment_config=deployment_config or {}
        )
        
        self.deployments.append(deployment_record)
        self.save_registry_data()
        
        logger.info(f"Recorded deployment: {model_name} v{version} to {environment}")
        return deployment_record
    
    def get_deployments(
        self,
        model_name: Optional[str] = None,
        environment: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[DeploymentRecord]:
        """Get deployment records with optional filtering."""
        
        deployments = self.deployments
        
        if model_name:
            deployments = [d for d in deployments if d.model_name == model_name]
        
        if environment:
            deployments = [d for d in deployments if d.environment == environment]
            
        if status:
            deployments = [d for d in deployments if d.status == status]
        
        return sorted(deployments, key=lambda d: d.deployed_at, reverse=True)


class ModelVersionManager:
    """
    High-level model version management interface.
    
    Orchestrates model registry, artifact storage, and deployment tracking.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.model_registry = ModelRegistry(config)
        self.artifact_store = ArtifactStore(config)
    
    async def version_and_deploy_model(
        self,
        model: TrajectoryPredictor,
        model_name: str,
        target_environment: str = "staging",
        evaluation_metrics: Optional[Dict[str, float]] = None,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        auto_promote: bool = False,
        deployment_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[ModelVersion, DeploymentRecord]:
        """
        Complete model versioning and deployment workflow.
        
        Args:
            model: Model to version and deploy
            model_name: Name for the model
            target_environment: Deployment environment
            evaluation_metrics: Model performance metrics
            description: Version description
            tags: Model tags
            auto_promote: Whether to auto-promote based on metrics
            deployment_config: Deployment configuration
            
        Returns:
            Tuple of (ModelVersion, DeploymentRecord)
        """
        
        try:
            # Register new model version
            model_version = await self.model_registry.register_model(
                model=model,
                model_name=model_name,
                description=description,
                tags=tags,
                metrics=evaluation_metrics
            )
            
            # Auto-promote based on metrics if requested
            if auto_promote and evaluation_metrics:
                should_promote = await self._evaluate_promotion_criteria(
                    model_name, model_version, evaluation_metrics
                )
                
                if should_promote:
                    if target_environment == "production":
                        # Promote to production
                        model_version = await self.model_registry.promote_model(
                            model_name, model_version.version, ModelStage.PRODUCTION
                        )
                    elif target_environment == "staging":
                        # Promote to staging
                        model_version = await self.model_registry.promote_model(
                            model_name, model_version.version, ModelStage.STAGING
                        )
            
            # Record deployment
            deployment_record = await self.model_registry.record_deployment(
                model_name=model_name,
                version=model_version.version,
                environment=target_environment,
                deployment_config=deployment_config
            )
            
            logger.info(f"Successfully versioned and deployed {model_name} v{model_version.version} to {target_environment}")
            
            return model_version, deployment_record
            
        except Exception as e:
            logger.error(f"Failed to version and deploy model {model_name}: {e}")
            raise
    
    async def _evaluate_promotion_criteria(
        self,
        model_name: str,
        model_version: ModelVersion,
        metrics: Dict[str, float]
    ) -> bool:
        """Evaluate whether model meets promotion criteria."""
        
        try:
            # Get promotion thresholds from config
            promotion_config = self.config.get("model_promotion", {})
            
            # Default thresholds
            default_thresholds = {
                "min_accuracy": 0.8,
                "max_ade_error": 2.0,
                "max_collision_risk": 0.1,
                "min_ttc": 3.0
            }
            
            thresholds = {**default_thresholds, **promotion_config.get("thresholds", {})}
            
            # Check thresholds
            promotion_checks = []
            
            # Check ADE (lower is better)
            if "trajectory_metrics_ade_mean" in metrics:
                ade_check = metrics["trajectory_metrics_ade_mean"] <= thresholds["max_ade_error"]
                promotion_checks.append(ade_check)
                logger.info(f"ADE check: {metrics['trajectory_metrics_ade_mean']:.3f} <= {thresholds['max_ade_error']} = {ade_check}")
            
            # Check collision risk (lower is better)
            if "safety_metrics_collision_risk_mean" in metrics:
                collision_check = metrics["safety_metrics_collision_risk_mean"] <= thresholds["max_collision_risk"]
                promotion_checks.append(collision_check)
                logger.info(f"Collision risk check: {metrics['safety_metrics_collision_risk_mean']:.3f} <= {thresholds['max_collision_risk']} = {collision_check}")
            
            # Check minimum TTC (higher is better)
            if "safety_metrics_min_ttc_mean" in metrics:
                ttc_check = metrics["safety_metrics_min_ttc_mean"] >= thresholds["min_ttc"]
                promotion_checks.append(ttc_check)
                logger.info(f"Min TTC check: {metrics['safety_metrics_min_ttc_mean']:.3f} >= {thresholds['min_ttc']} = {ttc_check}")
            
            # Require all checks to pass
            should_promote = all(promotion_checks) and len(promotion_checks) > 0
            
            logger.info(f"Promotion evaluation for {model_name} v{model_version.version}: {should_promote}")
            
            return should_promote
            
        except Exception as e:
            logger.error(f"Failed to evaluate promotion criteria: {e}")
            return False
    
    async def rollback_deployment(
        self,
        model_name: str,
        environment: str,
        target_version: Optional[str] = None
    ) -> Optional[DeploymentRecord]:
        """Rollback to previous model version."""
        
        try:
            # Get current deployments for the environment
            current_deployments = self.model_registry.get_deployments(
                model_name=model_name,
                environment=environment,
                status="active"
            )
            
            if not current_deployments:
                logger.warning(f"No active deployments found for {model_name} in {environment}")
                return None
            
            # Mark current deployment as rolled back
            current_deployment = current_deployments[0]
            current_deployment.status = "rolled_back"
            
            # Determine rollback target
            if target_version:
                rollback_version = target_version
            else:
                # Find previous version
                all_deployments = self.model_registry.get_deployments(
                    model_name=model_name,
                    environment=environment
                )
                
                if len(all_deployments) < 2:
                    logger.error("No previous version available for rollback")
                    return None
                    
                rollback_version = all_deployments[1].version
            
            # Create new deployment record for rollback
            rollback_deployment = await self.model_registry.record_deployment(
                model_name=model_name,
                version=rollback_version,
                environment=environment,
                deployed_by="system_rollback",
                deployment_config={"rollback_from": current_deployment.version}
            )
            
            logger.info(f"Rolled back {model_name} in {environment} from v{current_deployment.version} to v{rollback_version}")
            
            return rollback_deployment
            
        except Exception as e:
            logger.error(f"Failed to rollback deployment: {e}")
            return None
    
    def get_model_lineage(self, model_name: str) -> Dict[str, Any]:
        """Get complete model lineage and deployment history."""
        
        lineage = {
            "model_name": model_name,
            "versions": [],
            "deployments": [],
            "promotion_history": [],
            "current_production": None
        }
        
        # Get model versions
        versions = self.model_registry.list_model_versions(model_name)
        for version in sorted(versions, key=lambda v: [int(x) for x in v.version.split('.')]):
            version_info = {
                "version": version.version,
                "stage": version.stage.value,
                "created_at": version.created_at,
                "created_by": version.created_by,
                "description": version.description,
                "tags": version.tags,
                "metrics": version.metrics
            }
            
            # Add promotion history
            if "promotions" in version.metadata:
                version_info["promotions"] = version.metadata["promotions"]
                lineage["promotion_history"].extend(version.metadata["promotions"])
            
            lineage["versions"].append(version_info)
            
            # Track current production version
            if version.stage == ModelStage.PRODUCTION:
                lineage["current_production"] = version.version
        
        # Get deployment history
        deployments = self.model_registry.get_deployments(model_name=model_name)
        lineage["deployments"] = [
            {
                "version": d.version,
                "environment": d.environment,
                "deployed_at": d.deployed_at,
                "deployed_by": d.deployed_by,
                "status": d.status,
                "endpoint_url": d.endpoint_url
            }
            for d in deployments
        ]
        
        return lineage