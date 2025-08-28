"""CLI commands for MLOps functionality."""

import json
import click
from pathlib import Path
from typing import Optional

from ..core.logging import get_logger
from ..mlops.tracking import MLflowTracker
from ..mlops.registry import ModelRegistry
from ..mlops.monitoring import ModelMonitor, DataDriftDetector
from ..mlops.serving import PredictionService


@click.group()
def mlops():
    """MLOps commands for model management and monitoring."""
    pass


@mlops.group()
def tracking():
    """MLflow experiment tracking commands."""
    pass


@tracking.command()
@click.option("--experiment-name", default="vehicle_trajectory_prediction", help="Experiment name")
def list_experiments(experiment_name: str):
    """List MLflow experiments."""
    logger = get_logger(__name__)
    
    try:
        tracker = MLflowTracker()
        experiments = tracker.list_experiments()
        
        if not experiments:
            click.echo("No experiments found.")
            return
        
        click.echo(f"Found {len(experiments)} experiments:")
        for exp in experiments:
            click.echo(f"  - {exp['name']} (ID: {exp['experiment_id']})")
            
    except Exception as e:
        logger.error("Failed to list experiments", error=str(e))
        click.echo(f"Error: {e}")


@tracking.command()
@click.option("--experiment-name", default="vehicle_trajectory_prediction", help="Experiment name")
@click.option("--max-results", default=10, help="Maximum number of runs to show")
def list_runs(experiment_name: str, max_results: int):
    """List MLflow runs for an experiment."""
    logger = get_logger(__name__)
    
    try:
        tracker = MLflowTracker()
        runs = tracker.list_runs(experiment_name, max_results)
        
        if not runs:
            click.echo(f"No runs found for experiment '{experiment_name}'.")
            return
        
        click.echo(f"Found {len(runs)} runs for experiment '{experiment_name}':")
        for run in runs:
            click.echo(f"  - {run['run_name']} (ID: {run['run_id']}) - {run['status']}")
            if run['metrics']:
                click.echo(f"    Metrics: {run['metrics']}")
                
    except Exception as e:
        logger.error("Failed to list runs", error=str(e))
        click.echo(f"Error: {e}")


@tracking.command()
@click.option("--metric", default="rmse", help="Metric to optimize for")
@click.option("--experiment-name", default="vehicle_trajectory_prediction", help="Experiment name")
def get_best_model(metric: str, experiment_name: str):
    """Get the best model based on a metric."""
    logger = get_logger(__name__)
    
    try:
        tracker = MLflowTracker()
        best_run_id = tracker.get_best_model(metric, experiment_name)
        
        if best_run_id:
            click.echo(f"Best model for metric '{metric}': {best_run_id}")
        else:
            click.echo(f"No models found for experiment '{experiment_name}'.")
            
    except Exception as e:
        logger.error("Failed to get best model", error=str(e))
        click.echo(f"Error: {e}")


@mlops.group()
def registry():
    """Model registry commands."""
    pass


@registry.command()
def list_models():
    """List all registered models."""
    logger = get_logger(__name__)
    
    try:
        registry = ModelRegistry()
        models = registry.list_models()
        
        if not models:
            click.echo("No models found in registry.")
            return
        
        click.echo(f"Found {len(models)} models in registry:")
        for model_name in models:
            click.echo(f"  - {model_name}")
            
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        click.echo(f"Error: {e}")


@registry.command()
@click.argument("model_name")
def list_versions(model_name: str):
    """List versions of a model."""
    logger = get_logger(__name__)
    
    try:
        registry = ModelRegistry()
        versions = registry.list_model_versions(model_name)
        
        if not versions:
            click.echo(f"No versions found for model '{model_name}'.")
            return
        
        click.echo(f"Found {len(versions)} versions for model '{model_name}':")
        for version in versions:
            click.echo(f"  - {version.version} ({version.status}) - {version.created_at}")
            if version.metrics:
                click.echo(f"    Metrics: {version.metrics}")
                
    except Exception as e:
        logger.error("Failed to list versions", error=str(e))
        click.echo(f"Error: {e}")


@registry.command()
@click.argument("model_name")
@click.argument("version")
@click.option("--environment", default="production", help="Deployment environment")
@click.option("--endpoint-url", help="Endpoint URL")
@click.option("--replicas", default=1, help="Number of replicas")
def deploy_model(model_name: str, version: str, environment: str, endpoint_url: Optional[str], replicas: int):
    """Deploy a model version."""
    logger = get_logger(__name__)
    
    try:
        registry = ModelRegistry()
        deployment_id = registry.deploy_model(
            model_name=model_name,
            version=version,
            environment=environment,
            endpoint_url=endpoint_url,
            replicas=replicas
        )
        
        click.echo(f"Model deployed successfully: {deployment_id}")
        
    except Exception as e:
        logger.error("Failed to deploy model", error=str(e))
        click.echo(f"Error: {e}")


@registry.command()
@click.argument("deployment_id")
def undeploy_model(deployment_id: str):
    """Undeploy a model."""
    logger = get_logger(__name__)
    
    try:
        registry = ModelRegistry()
        success = registry.undeploy_model(deployment_id)
        
        if success:
            click.echo(f"Model undeployed successfully: {deployment_id}")
        else:
            click.echo(f"Failed to undeploy model: {deployment_id}")
            
    except Exception as e:
        logger.error("Failed to undeploy model", error=str(e))
        click.echo(f"Error: {e}")


@registry.command()
@click.option("--output", default="registry_summary.json", help="Output file path")
def export_summary(output: str):
    """Export registry summary."""
    logger = get_logger(__name__)
    
    try:
        registry = ModelRegistry()
        registry.export_registry(output)
        
        click.echo(f"Registry summary exported to: {output}")
        
    except Exception as e:
        logger.error("Failed to export registry summary", error=str(e))
        click.echo(f"Error: {e}")


@mlops.group()
def monitoring():
    """Model monitoring commands."""
    pass


@monitoring.command()
@click.option("--model-name", help="Filter by model name")
@click.option("--hours", default=24, help="Time window in hours")
def get_performance(model_name: Optional[str], hours: int):
    """Get model performance statistics."""
    logger = get_logger(__name__)
    
    try:
        monitor = ModelMonitor()
        
        if model_name:
            performance = monitor.get_model_performance(model_name, hours)
            if performance:
                click.echo(f"Performance for {model_name} (last {hours} hours):")
                click.echo(json.dumps(performance, indent=2))
            else:
                click.echo(f"No performance data found for {model_name}")
        else:
            # Get performance for all models
            summary = monitor.get_monitoring_summary()
            if summary:
                click.echo("Overall monitoring summary:")
                click.echo(json.dumps(summary, indent=2))
            else:
                click.echo("No monitoring data found")
                
    except Exception as e:
        logger.error("Failed to get performance", error=str(e))
        click.echo(f"Error: {e}")


@monitoring.command()
def check_alerts():
    """Check for monitoring alerts."""
    logger = get_logger(__name__)
    
    try:
        monitor = ModelMonitor()
        alerts = monitor.check_alerts()
        
        if not alerts:
            click.echo("No alerts detected.")
            return
        
        click.echo(f"Found {len(alerts)} alerts:")
        for alert in alerts:
            click.echo(f"  - [{alert['severity'].upper()}] {alert['type']}: {alert['message']}")
            
    except Exception as e:
        logger.error("Failed to check alerts", error=str(e))
        click.echo(f"Error: {e}")


@monitoring.command()
@click.option("--output", default="monitoring_summary.json", help="Output file path")
def export_monitoring(output: str):
    """Export monitoring data."""
    logger = get_logger(__name__)
    
    try:
        monitor = ModelMonitor()
        monitor.save_monitoring_data(output)
        
        click.echo(f"Monitoring data exported to: {output}")
        
    except Exception as e:
        logger.error("Failed to export monitoring data", error=str(e))
        click.echo(f"Error: {e}")


@mlops.group()
def drift():
    """Data drift detection commands."""
    pass


@drift.command()
@click.argument("reference_data")
@click.argument("current_data")
@click.option("--output", default="drift_report.json", help="Output file path")
def detect_drift(reference_data: str, current_data: str, output: str):
    """Detect data drift between reference and current data."""
    logger = get_logger(__name__)
    
    try:
        detector = DataDriftDetector()
        
        # Load reference data
        detector.load_reference_data(reference_data)
        
        # Load current data
        import pandas as pd
        if current_data.endswith('.parquet'):
            current_df = pd.read_parquet(current_data)
        elif current_data.endswith('.csv'):
            current_df = pd.read_csv(current_data)
        else:
            raise ValueError(f"Unsupported file format: {current_data}")
        
        # Detect drift
        drift_result = detector.detect_drift(current_df)
        
        # Save report
        detector.save_drift_report(output)
        
        click.echo("Drift detection completed:")
        click.echo(json.dumps(drift_result, indent=2))
        click.echo(f"Report saved to: {output}")
        
    except Exception as e:
        logger.error("Failed to detect drift", error=str(e))
        click.echo(f"Error: {e}")


@drift.command()
@click.option("--output", default="drift_summary.json", help="Output file path")
def get_drift_summary(output: str):
    """Get drift detection summary."""
    logger = get_logger(__name__)
    
    try:
        detector = DataDriftDetector()
        summary = detector.get_drift_summary()
        
        if summary:
            click.echo("Drift detection summary:")
            click.echo(json.dumps(summary, indent=2))
            
            # Save summary
            with open(output, "w") as f:
                json.dump(summary, f, indent=2)
            click.echo(f"Summary saved to: {output}")
        else:
            click.echo("No drift detection data found")
            
    except Exception as e:
        logger.error("Failed to get drift summary", error=str(e))
        click.echo(f"Error: {e}")


@mlops.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--workers", default=4, type=int, help="Number of worker processes")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, workers: int, reload: bool):
    """Start the model serving API server."""
    logger = get_logger(__name__)
    logger.info("Starting model serving API server", host=host, port=port, workers=workers)
    
    try:
        import uvicorn
        from ..mlops.serving import create_app
        
        app = create_app()
        
        if reload:
            uvicorn.run(
                "vehicle_trajectory_prediction.mlops.serving:app",
                host=host,
                port=port,
                reload=True,
                log_level="info"
            )
        else:
            uvicorn.run(
                app,
                host=host,
                port=port,
                workers=workers,
                log_level="info"
            )
    except ImportError as e:
        logger.error("Failed to import serving dependencies", error=str(e))
        click.echo(f"Error: {e}")
    except Exception as e:
        logger.error("Failed to start server", error=str(e))
        click.echo(f"Error: {e}")


@mlops.command()
def health_check():
    """Check the health of MLOps services."""
    logger = get_logger(__name__)
    
    try:
        # Check MLflow
        try:
            tracker = MLflowTracker()
            experiments = tracker.list_experiments()
            click.echo(f"✅ MLflow: Connected ({len(experiments)} experiments)")
        except Exception as e:
            click.echo(f"❌ MLflow: Failed to connect - {e}")
        
        # Check Model Registry
        try:
            registry = ModelRegistry()
            models = registry.list_models()
            click.echo(f"✅ Model Registry: Connected ({len(models)} models)")
        except Exception as e:
            click.echo(f"❌ Model Registry: Failed to connect - {e}")
        
        # Check Model Monitor
        try:
            monitor = ModelMonitor()
            summary = monitor.get_monitoring_summary()
            click.echo(f"✅ Model Monitor: Connected ({summary.get('total_predictions', 0)} predictions)")
        except Exception as e:
            click.echo(f"❌ Model Monitor: Failed to connect - {e}")
        
        # Check Data Drift Detector
        try:
            detector = DataDriftDetector()
            summary = detector.get_drift_summary()
            click.echo(f"✅ Data Drift Detector: Connected ({summary.get('total_drift_checks', 0)} checks)")
        except Exception as e:
            click.echo(f"❌ Data Drift Detector: Failed to connect - {e}")
            
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        click.echo(f"Error: {e}")