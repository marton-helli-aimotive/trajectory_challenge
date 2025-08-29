"""
End-to-end integration tests for trajectory prediction system.

This module tests:
- Complete data pipeline from raw data to predictions
- Model training and inference workflows
- API integration with model serving
- Monitoring and alerting integration
"""

import pytest
import numpy as np
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.trajectory_prediction.data.schemas import TrajectoryData, Position, Velocity
from src.trajectory_prediction.data.etl.pipeline import TrajectoryETLPipeline
from src.trajectory_prediction.models.base import TrajectoryPredictor
from src.trajectory_prediction.models.factory import ModelFactory
from src.trajectory_prediction.evaluation.evaluator import ModelEvaluator
from src.trajectory_prediction.api.models import TrajectoryRequest, TrajectoryResponse
from src.trajectory_prediction.mlops.monitoring import ModelMonitor, DataDriftDetector
from tests.conftest import validate_trajectory_data, validate_prediction_response


class TestEndToEndDataPipeline:
    """Test complete data pipeline from raw data to features."""
    
    @pytest.fixture
    def mock_data_pipeline(self):
        """Create mock data pipeline components."""
        class MockEndToEndPipeline:
            def __init__(self):
                self.raw_data_processed = 0
                self.trajectories_generated = 0
                self.features_extracted = 0
            
            async def ingest_raw_data(self, data_source: str, batch_size: int = 100) -> List[Dict[str, Any]]:
                """Mock raw data ingestion."""
                await asyncio.sleep(0.01)  # Simulate I/O
                
                raw_data = []
                for i in range(batch_size):
                    raw_data.append({
                        'trajectory_id': f'raw_{i}',
                        'vehicle_id': f'vehicle_{i}',
                        'x_coords': [float(j) for j in range(10)],
                        'y_coords': [float(j * 0.5) for j in range(10)],
                        'timestamps': [float(j * 0.1) for j in range(10)],
                        'velocities_x': [1.0 for _ in range(10)],
                        'velocities_y': [0.5 for _ in range(10)],
                        'metadata': {'source': data_source}
                    })
                
                self.raw_data_processed += len(raw_data)
                return raw_data
            
            async def process_to_trajectories(self, raw_data: List[Dict[str, Any]]) -> List[TrajectoryData]:
                """Convert raw data to TrajectoryData objects."""
                await asyncio.sleep(0.01)  # Simulate processing
                
                trajectories = []
                for item in raw_data:
                    positions = [
                        Position(x=x, y=y)
                        for x, y in zip(item['x_coords'], item['y_coords'])
                    ]
                    velocities = [
                        Velocity(vx=vx, vy=vy)
                        for vx, vy in zip(item['velocities_x'], item['velocities_y'])
                    ]
                    
                    trajectory = TrajectoryData(
                        trajectory_id=item['trajectory_id'],
                        vehicle_id=item['vehicle_id'],
                        positions=positions,
                        velocities=velocities,
                        timestamps=item['timestamps'],
                        metadata=item.get('metadata', {})
                    )
                    trajectories.append(trajectory)
                
                self.trajectories_generated += len(trajectories)
                return trajectories
            
            async def extract_features(self, trajectories: List[TrajectoryData]) -> Dict[str, np.ndarray]:
                """Extract features from trajectories."""
                await asyncio.sleep(0.01)  # Simulate feature extraction
                
                n_trajectories = len(trajectories)
                n_features = 50  # Mock feature dimension
                
                features = {
                    'temporal_features': np.random.randn(n_trajectories, 10),
                    'spatial_features': np.random.randn(n_trajectories, 20),
                    'kinematic_features': np.random.randn(n_trajectories, 20)
                }
                
                self.features_extracted += n_trajectories
                return features
            
            async def run_complete_pipeline(
                self,
                data_source: str,
                batch_size: int = 50
            ) -> Tuple[List[TrajectoryData], Dict[str, np.ndarray]]:
                """Run the complete pipeline."""
                # Step 1: Ingest raw data
                raw_data = await self.ingest_raw_data(data_source, batch_size)
                
                # Step 2: Process to trajectories
                trajectories = await self.process_to_trajectories(raw_data)
                
                # Step 3: Extract features
                features = await self.extract_features(trajectories)
                
                return trajectories, features
        
        return MockEndToEndPipeline()
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_execution(self, mock_data_pipeline):
        """Test complete pipeline execution."""
        trajectories, features = await mock_data_pipeline.run_complete_pipeline(
            data_source='test_source',
            batch_size=20
        )
        
        # Verify pipeline execution
        assert mock_data_pipeline.raw_data_processed == 20
        assert mock_data_pipeline.trajectories_generated == 20
        assert mock_data_pipeline.features_extracted == 20
        
        # Verify outputs
        assert len(trajectories) == 20
        assert all(isinstance(traj, TrajectoryData) for traj in trajectories)
        assert all(validate_trajectory_data(traj) for traj in trajectories)
        
        assert isinstance(features, dict)
        assert len(features['temporal_features']) == 20
        assert len(features['spatial_features']) == 20
        assert len(features['kinematic_features']) == 20
    
    @pytest.mark.asyncio
    async def test_pipeline_with_multiple_sources(self, mock_data_pipeline):
        """Test pipeline with multiple data sources."""
        sources = ['source_1', 'source_2', 'source_3']
        all_trajectories = []
        all_features = []
        
        for source in sources:
            trajectories, features = await mock_data_pipeline.run_complete_pipeline(
                data_source=source,
                batch_size=10
            )
            all_trajectories.extend(trajectories)
            all_features.append(features)
        
        # Verify processing across multiple sources
        assert len(all_trajectories) == 30  # 3 sources × 10 batch size
        assert len(all_features) == 3
        
        # Verify data source tracking
        source_counts = {}
        for traj in all_trajectories:
            source = traj.metadata.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        assert len(source_counts) == 3
        assert all(count == 10 for count in source_counts.values())
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, mock_data_pipeline):
        """Test pipeline error handling and recovery."""
        # Mock a failure in processing step
        original_process = mock_data_pipeline.process_to_trajectories
        
        async def failing_process(raw_data):
            if len(raw_data) > 15:  # Fail for large batches
                raise RuntimeError("Processing overload")
            return await original_process(raw_data)
        
        mock_data_pipeline.process_to_trajectories = failing_process
        
        # Test with small batch (should succeed)
        trajectories, features = await mock_data_pipeline.run_complete_pipeline(
            data_source='small_batch',
            batch_size=10
        )
        assert len(trajectories) == 10
        
        # Test with large batch (should fail)
        with pytest.raises(RuntimeError):
            await mock_data_pipeline.run_complete_pipeline(
                data_source='large_batch',
                batch_size=20
            )
    
    @pytest.mark.asyncio
    async def test_pipeline_concurrent_processing(self, mock_data_pipeline):
        """Test concurrent pipeline processing."""
        # Run multiple pipelines concurrently
        tasks = []
        for i in range(5):
            task = mock_data_pipeline.run_complete_pipeline(
                data_source=f'concurrent_source_{i}',
                batch_size=5
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        # Verify all pipelines completed
        assert len(results) == 5
        assert all(len(trajectories) == 5 for trajectories, _ in results)
        
        # Verify total processing
        assert mock_data_pipeline.raw_data_processed == 25
        assert mock_data_pipeline.trajectories_generated == 25
        assert mock_data_pipeline.features_extracted == 25


class TestModelTrainingWorkflow:
    """Test complete model training workflow."""
    
    @pytest.fixture
    def mock_training_workflow(self, sample_trajectories):
        """Create mock training workflow."""
        class MockTrainingWorkflow:
            def __init__(self, trajectories):
                self.trajectories = trajectories
                self.trained_models = {}
                self.evaluation_results = {}
            
            async def prepare_training_data(
                self,
                train_split: float = 0.8
            ) -> Tuple[List[TrajectoryData], List[TrajectoryData]]:
                """Split data into training and validation sets."""
                n_train = int(len(self.trajectories) * train_split)
                
                train_data = self.trajectories[:n_train]
                val_data = self.trajectories[n_train:]
                
                return train_data, val_data
            
            async def train_model(
                self,
                model_name: str,
                train_data: List[TrajectoryData],
                model_config: Dict[str, Any] = None
            ) -> TrajectoryPredictor:
                """Train a model with given data."""
                await asyncio.sleep(0.1)  # Simulate training time
                
                # Create mock trained model
                mock_model = Mock(spec=TrajectoryPredictor)
                mock_model.model_name = model_name
                mock_model.is_trained = True
                mock_model.config = model_config or {}
                
                # Mock predict method
                async def mock_predict(trajectory, **kwargs):
                    # Simple extrapolation prediction
                    last_pos = trajectory.positions[-1]
                    last_vel = trajectory.velocities[-1] if trajectory.velocities else Velocity(vx=1.0, vy=0.0)
                    
                    pred_positions = []
                    pred_velocities = []
                    pred_timestamps = []
                    
                    for i in range(5):  # Predict 5 steps
                        dt = 0.1
                        new_time = trajectory.timestamps[-1] + (i + 1) * dt
                        new_x = last_pos.x + last_vel.vx * (i + 1) * dt
                        new_y = last_pos.y + last_vel.vy * (i + 1) * dt
                        
                        pred_positions.append(Position(x=new_x, y=new_y))
                        pred_velocities.append(last_vel)
                        pred_timestamps.append(new_time)
                    
                    return TrajectoryData(
                        trajectory_id=f"{trajectory.trajectory_id}_pred",
                        vehicle_id=trajectory.vehicle_id,
                        positions=pred_positions,
                        velocities=pred_velocities,
                        timestamps=pred_timestamps
                    )
                
                mock_model.predict = mock_predict
                
                self.trained_models[model_name] = mock_model
                return mock_model
            
            async def evaluate_model(
                self,
                model: TrajectoryPredictor,
                val_data: List[TrajectoryData]
            ) -> Dict[str, float]:
                """Evaluate model on validation data."""
                await asyncio.sleep(0.05)  # Simulate evaluation time
                
                # Mock evaluation results
                results = {
                    'rmse': np.random.uniform(0.1, 1.0),
                    'mae': np.random.uniform(0.05, 0.8),
                    'ade': np.random.uniform(0.1, 1.0),
                    'fde': np.random.uniform(0.2, 1.5)
                }
                
                self.evaluation_results[model.model_name] = results
                return results
            
            async def run_training_workflow(
                self,
                model_configs: Dict[str, Dict[str, Any]]
            ) -> Dict[str, Tuple[TrajectoryPredictor, Dict[str, float]]]:
                """Run complete training workflow for multiple models."""
                # Prepare data
                train_data, val_data = await self.prepare_training_data()
                
                results = {}
                
                # Train and evaluate each model
                for model_name, config in model_configs.items():
                    # Train model
                    model = await self.train_model(model_name, train_data, config)
                    
                    # Evaluate model
                    eval_results = await self.evaluate_model(model, val_data)
                    
                    results[model_name] = (model, eval_results)
                
                return results
        
        return MockTrainingWorkflow(sample_trajectories)
    
    @pytest.mark.asyncio
    async def test_single_model_training(self, mock_training_workflow):
        """Test training workflow for single model."""
        model_configs = {
            'constant_velocity': {
                'prediction_horizon': 2.0,
                'time_step': 0.1
            }
        }
        
        results = await mock_training_workflow.run_training_workflow(model_configs)
        
        assert len(results) == 1
        assert 'constant_velocity' in results
        
        model, eval_results = results['constant_velocity']
        assert model.model_name == 'constant_velocity'
        assert model.is_trained
        assert isinstance(eval_results, dict)
        assert all(metric in eval_results for metric in ['rmse', 'mae', 'ade', 'fde'])
    
    @pytest.mark.asyncio
    async def test_multiple_model_training(self, mock_training_workflow):
        """Test training workflow for multiple models."""
        model_configs = {
            'constant_velocity': {'prediction_horizon': 2.0},
            'constant_acceleration': {'prediction_horizon': 3.0},
            'polynomial': {'degree': 2, 'prediction_horizon': 2.5}
        }
        
        results = await mock_training_workflow.run_training_workflow(model_configs)
        
        assert len(results) == 3
        
        for model_name in model_configs.keys():
            assert model_name in results
            model, eval_results = results[model_name]
            assert model.model_name == model_name
            assert model.is_trained
            assert isinstance(eval_results, dict)
    
    @pytest.mark.asyncio
    async def test_training_data_preparation(self, mock_training_workflow):
        """Test training data preparation and splitting."""
        train_data, val_data = await mock_training_workflow.prepare_training_data(train_split=0.7)
        
        total_samples = len(mock_training_workflow.trajectories)
        expected_train = int(total_samples * 0.7)
        expected_val = total_samples - expected_train
        
        assert len(train_data) == expected_train
        assert len(val_data) == expected_val
        assert len(train_data) + len(val_data) == total_samples
    
    @pytest.mark.asyncio
    async def test_model_evaluation_consistency(self, mock_training_workflow):
        """Test that model evaluation is consistent."""
        model_configs = {'test_model': {}}
        
        # Run workflow twice
        results1 = await mock_training_workflow.run_training_workflow(model_configs)
        results2 = await mock_training_workflow.run_training_workflow(model_configs)
        
        # Models should be trained successfully both times
        assert 'test_model' in results1
        assert 'test_model' in results2
        
        model1, eval1 = results1['test_model']
        model2, eval2 = results2['test_model']
        
        assert model1.is_trained
        assert model2.is_trained


class TestModelServingWorkflow:
    """Test model serving and inference workflow."""
    
    @pytest.fixture
    def mock_serving_workflow(self, mock_model, sample_trajectory):
        """Create mock model serving workflow."""
        class MockServingWorkflow:
            def __init__(self, model, sample_trajectory):
                self.models = {'mock_model': model}
                self.sample_trajectory = sample_trajectory
                self.prediction_cache = {}
                self.request_count = 0
            
            async def load_models(self, model_paths: Dict[str, str]) -> Dict[str, TrajectoryPredictor]:
                """Load models from disk."""
                await asyncio.sleep(0.1)  # Simulate loading time
                
                loaded_models = {}
                for model_name, path in model_paths.items():
                    if model_name in self.models:
                        loaded_models[model_name] = self.models[model_name]
                    else:
                        # Create dummy model if not found
                        mock = Mock(spec=TrajectoryPredictor)
                        mock.model_name = model_name
                        mock.is_trained = True
                        loaded_models[model_name] = mock
                
                return loaded_models
            
            async def serve_prediction(
                self,
                trajectory_request: TrajectoryRequest
            ) -> List[TrajectoryResponse]:
                """Serve prediction requests."""
                self.request_count += 1
                await asyncio.sleep(0.02)  # Simulate inference time
                
                responses = []
                
                for model_name in trajectory_request.config.models:
                    if model_name in self.models:
                        model = self.models[model_name]
                        
                        # Generate prediction
                        predicted_traj = await model.predict(self.sample_trajectory)
                        
                        # Create response
                        response = TrajectoryResponse(
                            request_id=f"req_{self.request_count}",
                            model_name=model_name,
                            predicted_trajectory=predicted_traj,
                            confidence=np.random.uniform(0.7, 0.95),
                            inference_time=0.02,
                            metadata={'timestamp': asyncio.get_event_loop().time()}
                        )
                        responses.append(response)
                
                return responses
            
            async def batch_serve_predictions(
                self,
                requests: List[TrajectoryRequest]
            ) -> List[List[TrajectoryResponse]]:
                """Serve multiple prediction requests."""
                tasks = [self.serve_prediction(req) for req in requests]
                return await asyncio.gather(*tasks)
            
            def get_serving_stats(self) -> Dict[str, Any]:
                """Get serving statistics."""
                return {
                    'total_requests': self.request_count,
                    'models_loaded': len(self.models),
                    'cache_size': len(self.prediction_cache)
                }
        
        return MockServingWorkflow(mock_model, sample_trajectory)
    
    @pytest.mark.asyncio
    async def test_model_loading(self, mock_serving_workflow):
        """Test model loading workflow."""
        model_paths = {
            'model_1': '/path/to/model1.pkl',
            'model_2': '/path/to/model2.pkl'
        }
        
        loaded_models = await mock_serving_workflow.load_models(model_paths)
        
        assert len(loaded_models) == 2
        assert 'model_1' in loaded_models
        assert 'model_2' in loaded_models
        assert all(model.is_trained for model in loaded_models.values())
    
    @pytest.mark.asyncio
    async def test_single_prediction_serving(self, mock_serving_workflow, sample_trajectory_request):
        """Test serving single prediction request."""
        responses = await mock_serving_workflow.serve_prediction(sample_trajectory_request)
        
        assert len(responses) >= 1
        
        for response in responses:
            assert isinstance(response, TrajectoryResponse)
            assert validate_prediction_response(response)
            assert response.model_name in sample_trajectory_request.config.models
            assert 0.0 <= response.confidence <= 1.0
            assert response.inference_time > 0
    
    @pytest.mark.asyncio
    async def test_batch_prediction_serving(self, mock_serving_workflow, sample_trajectory_request):
        """Test serving batch prediction requests."""
        # Create multiple requests
        requests = [sample_trajectory_request for _ in range(5)]
        
        batch_responses = await mock_serving_workflow.batch_serve_predictions(requests)
        
        assert len(batch_responses) == 5
        
        for responses in batch_responses:
            assert len(responses) >= 1
            for response in responses:
                assert isinstance(response, TrajectoryResponse)
                assert validate_prediction_response(response)
    
    @pytest.mark.asyncio
    async def test_serving_performance_monitoring(self, mock_serving_workflow, sample_trajectory_request):
        """Test serving performance monitoring."""
        # Initial stats
        initial_stats = mock_serving_workflow.get_serving_stats()
        assert initial_stats['total_requests'] == 0
        
        # Serve some requests
        for _ in range(10):
            await mock_serving_workflow.serve_prediction(sample_trajectory_request)
        
        # Check updated stats
        final_stats = mock_serving_workflow.get_serving_stats()
        assert final_stats['total_requests'] == 10
        assert final_stats['models_loaded'] >= 1
    
    @pytest.mark.asyncio
    async def test_concurrent_serving(self, mock_serving_workflow, sample_trajectory_request):
        """Test concurrent prediction serving."""
        # Create concurrent requests
        tasks = [
            mock_serving_workflow.serve_prediction(sample_trajectory_request)
            for _ in range(10)
        ]
        
        # Execute concurrently
        all_responses = await asyncio.gather(*tasks)
        
        assert len(all_responses) == 10
        
        # Verify all requests were processed
        stats = mock_serving_workflow.get_serving_stats()
        assert stats['total_requests'] == 10


class TestMonitoringAndAlertingIntegration:
    """Test monitoring and alerting system integration."""
    
    @pytest.fixture
    def mock_monitoring_system(self):
        """Create mock monitoring system."""
        class MockMonitoringSystem:
            def __init__(self):
                self.metrics_history = []
                self.alerts_generated = []
                self.thresholds = {
                    'rmse': 1.0,
                    'mae': 0.8,
                    'inference_time': 0.1,
                    'memory_usage': 500  # MB
                }
            
            async def collect_metrics(
                self,
                model_name: str,
                predictions: List[TrajectoryResponse],
                ground_truth: List[TrajectoryData] = None
            ) -> Dict[str, float]:
                """Collect performance metrics."""
                await asyncio.sleep(0.01)
                
                metrics = {
                    'model_name': model_name,
                    'timestamp': asyncio.get_event_loop().time(),
                    'request_count': len(predictions),
                    'average_confidence': np.mean([p.confidence for p in predictions]),
                    'average_inference_time': np.mean([p.inference_time for p in predictions]),
                    'memory_usage': np.random.uniform(100, 600)  # Mock memory usage
                }
                
                # Add accuracy metrics if ground truth available
                if ground_truth:
                    metrics.update({
                        'rmse': np.random.uniform(0.1, 1.5),
                        'mae': np.random.uniform(0.05, 1.2),
                        'ade': np.random.uniform(0.1, 1.3)
                    })
                
                self.metrics_history.append(metrics)
                return metrics
            
            async def check_thresholds(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
                """Check metrics against thresholds and generate alerts."""
                alerts = []
                
                for metric_name, threshold in self.thresholds.items():
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        
                        if value > threshold:
                            alert = {
                                'type': 'threshold_exceeded',
                                'metric': metric_name,
                                'value': value,
                                'threshold': threshold,
                                'severity': 'high' if value > threshold * 1.5 else 'medium',
                                'timestamp': asyncio.get_event_loop().time(),
                                'model': metrics.get('model_name', 'unknown')
                            }
                            alerts.append(alert)
                            self.alerts_generated.append(alert)
                
                return alerts
            
            async def detect_drift(
                self,
                current_metrics: Dict[str, float],
                reference_window: int = 100
            ) -> Dict[str, Any]:
                """Detect performance drift."""
                await asyncio.sleep(0.01)
                
                if len(self.metrics_history) < reference_window:
                    return {'drift_detected': False, 'reason': 'insufficient_history'}
                
                # Simple drift detection based on recent vs historical average
                recent_metrics = self.metrics_history[-10:]
                historical_metrics = self.metrics_history[-reference_window:-10]
                
                drift_info = {'drift_detected': False, 'metrics': {}}
                
                for metric in ['rmse', 'mae', 'average_confidence']:
                    if metric in current_metrics:
                        recent_avg = np.mean([m.get(metric, 0) for m in recent_metrics])
                        historical_avg = np.mean([m.get(metric, 0) for m in historical_metrics])
                        
                        if abs(recent_avg - historical_avg) > historical_avg * 0.2:  # 20% change
                            drift_info['drift_detected'] = True
                            drift_info['metrics'][metric] = {
                                'recent_avg': recent_avg,
                                'historical_avg': historical_avg,
                                'change_pct': abs(recent_avg - historical_avg) / historical_avg
                            }
                
                return drift_info
            
            async def run_monitoring_cycle(
                self,
                model_name: str,
                predictions: List[TrajectoryResponse],
                ground_truth: List[TrajectoryData] = None
            ) -> Dict[str, Any]:
                """Run complete monitoring cycle."""
                # Collect metrics
                metrics = await self.collect_metrics(model_name, predictions, ground_truth)
                
                # Check thresholds
                alerts = await self.check_thresholds(metrics)
                
                # Detect drift
                drift_info = await self.detect_drift(metrics)
                
                return {
                    'metrics': metrics,
                    'alerts': alerts,
                    'drift_info': drift_info
                }
        
        return MockMonitoringSystem()
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, mock_monitoring_system):
        """Test metrics collection from predictions."""
        # Create mock predictions
        predictions = []
        for i in range(5):
            pred = Mock(spec=TrajectoryResponse)
            pred.confidence = np.random.uniform(0.7, 0.95)
            pred.inference_time = np.random.uniform(0.01, 0.05)
            predictions.append(pred)
        
        metrics = await mock_monitoring_system.collect_metrics('test_model', predictions)
        
        assert isinstance(metrics, dict)
        assert metrics['model_name'] == 'test_model'
        assert metrics['request_count'] == 5
        assert 0.0 <= metrics['average_confidence'] <= 1.0
        assert metrics['average_inference_time'] > 0
        assert len(mock_monitoring_system.metrics_history) == 1
    
    @pytest.mark.asyncio
    async def test_threshold_monitoring(self, mock_monitoring_system):
        """Test threshold-based alerting."""
        # Create metrics that exceed thresholds
        bad_metrics = {
            'model_name': 'failing_model',
            'rmse': 2.0,  # Exceeds threshold of 1.0
            'mae': 1.5,   # Exceeds threshold of 0.8
            'inference_time': 0.2  # Exceeds threshold of 0.1
        }
        
        alerts = await mock_monitoring_system.check_thresholds(bad_metrics)
        
        assert len(alerts) == 3  # All three metrics exceeded
        
        for alert in alerts:
            assert alert['type'] == 'threshold_exceeded'
            assert alert['value'] > alert['threshold']
            assert alert['severity'] in ['medium', 'high']
    
    @pytest.mark.asyncio
    async def test_drift_detection(self, mock_monitoring_system):
        """Test performance drift detection."""
        # Populate history with stable metrics
        for i in range(50):
            metrics = {
                'rmse': 0.5 + np.random.normal(0, 0.05),
                'mae': 0.3 + np.random.normal(0, 0.03),
                'average_confidence': 0.85 + np.random.normal(0, 0.05)
            }
            mock_monitoring_system.metrics_history.append(metrics)
        
        # Add recent metrics with drift
        for i in range(10):
            metrics = {
                'rmse': 0.8 + np.random.normal(0, 0.05),  # Higher RMSE
                'mae': 0.3 + np.random.normal(0, 0.03),
                'average_confidence': 0.85 + np.random.normal(0, 0.05)
            }
            mock_monitoring_system.metrics_history.append(metrics)
        
        current_metrics = {'rmse': 0.8, 'mae': 0.3, 'average_confidence': 0.85}
        
        drift_info = await mock_monitoring_system.detect_drift(current_metrics)
        
        assert drift_info['drift_detected'] == True
        assert 'rmse' in drift_info['metrics']  # RMSE should show drift
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_cycle(self, mock_monitoring_system):
        """Test complete monitoring cycle integration."""
        # Create mock predictions
        predictions = []
        for i in range(10):
            pred = Mock(spec=TrajectoryResponse)
            pred.confidence = 0.8
            pred.inference_time = 0.15  # Exceeds threshold
            predictions.append(pred)
        
        result = await mock_monitoring_system.run_monitoring_cycle(
            'monitored_model',
            predictions
        )
        
        assert 'metrics' in result
        assert 'alerts' in result
        assert 'drift_info' in result
        
        # Should generate alert for inference time
        assert len(result['alerts']) > 0
        assert any(alert['metric'] == 'inference_time' for alert in result['alerts'])
    
    @pytest.mark.asyncio
    async def test_continuous_monitoring(self, mock_monitoring_system):
        """Test continuous monitoring over multiple cycles."""
        model_name = 'continuous_model'
        
        # Run multiple monitoring cycles
        for cycle in range(5):
            predictions = []
            for i in range(5):
                pred = Mock(spec=TrajectoryResponse)
                pred.confidence = np.random.uniform(0.7, 0.9)
                pred.inference_time = np.random.uniform(0.01, 0.08)
                predictions.append(pred)
            
            result = await mock_monitoring_system.run_monitoring_cycle(
                model_name,
                predictions
            )
        
        # Should have accumulated metrics history
        assert len(mock_monitoring_system.metrics_history) == 5
        
        # All metrics should be for the same model
        assert all(m['model_name'] == model_name for m in mock_monitoring_system.metrics_history)


class TestSystemIntegrationTests:
    """Test complete system integration across all components."""
    
    @pytest.mark.asyncio
    async def test_complete_system_workflow(
        self,
        mock_data_pipeline,
        mock_training_workflow,
        mock_serving_workflow,
        mock_monitoring_system
    ):
        """Test complete end-to-end system workflow."""
        # Step 1: Data Pipeline
        trajectories, features = await mock_data_pipeline.run_complete_pipeline(
            data_source='integration_test',
            batch_size=10
        )
        
        # Step 2: Model Training
        model_configs = {
            'integration_model': {'prediction_horizon': 2.0}
        }
        training_results = await mock_training_workflow.run_training_workflow(model_configs)
        
        # Step 3: Model Serving
        from src.trajectory_prediction.api.models import TrajectoryRequest, TrajectoryInput, TrajectoryPoint, PredictionConfig
        
        # Create mock request
        trajectory_points = [
            TrajectoryPoint(timestamp=0.0, x=0.0, y=0.0, vx=1.0, vy=0.5),
            TrajectoryPoint(timestamp=0.1, x=0.1, y=0.05, vx=1.0, vy=0.5)
        ]
        
        trajectory_input = TrajectoryInput(
            trajectory_id='integration_test',
            vehicle_id='test_vehicle',
            points=trajectory_points
        )
        
        config = PredictionConfig(
            prediction_horizon=2.0,
            models=['mock_model']
        )
        
        request = TrajectoryRequest(trajectory=trajectory_input, config=config)
        
        predictions = await mock_serving_workflow.serve_prediction(request)
        
        # Step 4: Monitoring
        monitoring_result = await mock_monitoring_system.run_monitoring_cycle(
            'integration_model',
            predictions
        )
        
        # Verify complete workflow
        assert len(trajectories) == 10
        assert 'integration_model' in training_results
        assert len(predictions) >= 1
        assert 'metrics' in monitoring_result
        
        # Verify data flow
        model, eval_results = training_results['integration_model']
        assert model.is_trained
        assert isinstance(eval_results, dict)
        
        for prediction in predictions:
            assert validate_prediction_response(prediction)
        
        assert monitoring_result['metrics']['request_count'] == len(predictions)
    
    @pytest.mark.asyncio
    async def test_system_fault_tolerance(self, mock_data_pipeline, mock_monitoring_system):
        """Test system behavior under failure conditions."""
        # Test with pipeline failures
        original_process = mock_data_pipeline.process_to_trajectories
        
        async def intermittent_failure(raw_data):
            if np.random.random() < 0.3:  # 30% failure rate
                raise RuntimeError("Intermittent failure")
            return await original_process(raw_data)
        
        mock_data_pipeline.process_to_trajectories = intermittent_failure
        
        # Attempt multiple pipeline runs
        successful_runs = 0
        failed_runs = 0
        
        for i in range(10):
            try:
                trajectories, features = await mock_data_pipeline.run_complete_pipeline(
                    data_source=f'fault_test_{i}',
                    batch_size=5
                )
                successful_runs += 1
            except RuntimeError:
                failed_runs += 1
        
        # Should have both successes and failures
        assert successful_runs > 0
        assert failed_runs > 0
        assert successful_runs + failed_runs == 10
    
    @pytest.mark.asyncio
    async def test_system_performance_under_load(
        self,
        mock_serving_workflow,
        mock_monitoring_system,
        sample_trajectory_request
    ):
        """Test system performance under high load."""
        # Generate high load
        num_requests = 50
        concurrent_requests = []
        
        for i in range(num_requests):
            concurrent_requests.append(
                mock_serving_workflow.serve_prediction(sample_trajectory_request)
            )
        
        # Execute all requests concurrently
        start_time = asyncio.get_event_loop().time()
        all_responses = await asyncio.gather(*concurrent_requests)
        end_time = asyncio.get_event_loop().time()
        
        total_time = end_time - start_time
        
        # Verify all requests completed
        assert len(all_responses) == num_requests
        
        # Verify throughput is reasonable
        throughput = num_requests / total_time
        assert throughput > 10  # At least 10 requests per second
        
        # Collect all predictions for monitoring
        all_predictions = []
        for responses in all_responses:
            all_predictions.extend(responses)
        
        # Run monitoring on high load
        monitoring_result = await mock_monitoring_system.run_monitoring_cycle(
            'load_test_model',
            all_predictions
        )
        
        # Verify monitoring can handle high load
        assert monitoring_result['metrics']['request_count'] == len(all_predictions)
        
        # Check serving statistics
        stats = mock_serving_workflow.get_serving_stats()
        assert stats['total_requests'] == num_requests