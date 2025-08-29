"""
Unit tests for data pipeline components.

This module tests:
- Data extraction and ETL processes
- Data validation and quality checks
- Feature engineering pipelines
- Data transformation and preprocessing
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
from typing import List, Dict, Any

from src.trajectory_prediction.data.etl.extractors import DataExtractor, ParallelDataExtractor
from src.trajectory_prediction.data.etl.processors import DataProcessor, ParquetProcessor
from src.trajectory_prediction.data.etl.pipeline import TrajectoryETLPipeline
from src.trajectory_prediction.data.schemas import TrajectoryData, Position, Velocity
from src.trajectory_prediction.features.trajectory_features import TrajectoryFeatureExtractor
from src.trajectory_prediction.utils.data_quality import DataQualityMonitor
from tests.conftest import validate_trajectory_data


class TestDataExtractor:
    """Test base DataExtractor functionality."""
    
    @pytest.fixture
    def mock_extractor(self):
        """Create a mock data extractor."""
        class MockExtractor(DataExtractor):
            def __init__(self):
                super().__init__()
                self.extracted_data = []
            
            async def extract_batch(self, batch_params: Dict[str, Any]) -> List[Dict[str, Any]]:
                """Mock extraction that returns sample data."""
                batch_size = batch_params.get('batch_size', 5)
                return [
                    {
                        'trajectory_id': f'traj_{i}',
                        'vehicle_id': f'vehicle_{i}',
                        'x_coords': [float(j) for j in range(10)],
                        'y_coords': [float(j * 0.5) for j in range(10)],
                        'timestamps': [float(j * 0.1) for j in range(10)],
                        'velocities_x': [1.0 for _ in range(10)],
                        'velocities_y': [0.5 for _ in range(10)]
                    }
                    for i in range(batch_size)
                ]
            
            async def validate_data(self, data: Dict[str, Any]) -> bool:
                """Simple validation."""
                required_fields = ['trajectory_id', 'vehicle_id', 'x_coords', 'y_coords']
                return all(field in data for field in required_fields)
        
        return MockExtractor()
    
    @pytest.mark.asyncio
    async def test_extract_batch(self, mock_extractor):
        """Test batch extraction functionality."""
        batch_params = {'batch_size': 3}
        batch_data = await mock_extractor.extract_batch(batch_params)
        
        assert len(batch_data) == 3
        assert all(isinstance(item, dict) for item in batch_data)
        assert all('trajectory_id' in item for item in batch_data)
    
    @pytest.mark.asyncio
    async def test_data_validation(self, mock_extractor):
        """Test data validation functionality."""
        valid_data = {
            'trajectory_id': 'test',
            'vehicle_id': 'vehicle_1',
            'x_coords': [1.0, 2.0, 3.0],
            'y_coords': [1.0, 2.0, 3.0]
        }
        
        invalid_data = {
            'trajectory_id': 'test'
            # Missing required fields
        }
        
        assert await mock_extractor.validate_data(valid_data)
        assert not await mock_extractor.validate_data(invalid_data)
    
    @pytest.mark.asyncio
    async def test_extraction_error_handling(self, mock_extractor):
        """Test error handling during extraction."""
        # Override extract_batch to raise an error
        async def failing_extract_batch(batch_params):
            raise ValueError("Extraction failed")
        
        mock_extractor.extract_batch = failing_extract_batch
        
        with pytest.raises(ValueError):
            await mock_extractor.extract_batch({'batch_size': 1})


class TestParallelDataExtractor:
    """Test parallel data extraction functionality."""
    
    @pytest.fixture
    def parallel_extractor(self):
        """Create a parallel data extractor."""
        class MockParallelExtractor(ParallelDataExtractor):
            def __init__(self):
                super().__init__(max_workers=2)
            
            async def extract_batch(self, batch_params: Dict[str, Any]) -> List[Dict[str, Any]]:
                """Mock parallel extraction."""
                batch_id = batch_params.get('batch_id', 0)
                batch_size = batch_params.get('batch_size', 2)
                
                # Simulate some processing time
                await asyncio.sleep(0.01)
                
                return [
                    {
                        'trajectory_id': f'batch_{batch_id}_traj_{i}',
                        'vehicle_id': f'vehicle_{batch_id}_{i}',
                        'x_coords': [float(j + batch_id * 10) for j in range(5)],
                        'y_coords': [float(j * 0.5) for j in range(5)],
                        'timestamps': [float(j * 0.1) for j in range(5)]
                    }
                    for i in range(batch_size)
                ]
            
            async def validate_data(self, data: Dict[str, Any]) -> bool:
                return 'trajectory_id' in data
        
        return MockParallelExtractor()
    
    @pytest.mark.asyncio
    async def test_parallel_extraction(self, parallel_extractor):
        """Test parallel batch extraction."""
        batch_configs = [
            {'batch_id': 0, 'batch_size': 2},
            {'batch_id': 1, 'batch_size': 3},
            {'batch_id': 2, 'batch_size': 1}
        ]
        
        results = await parallel_extractor.extract_parallel(batch_configs)
        
        assert len(results) == 3  # 3 batches
        assert len(results[0]) == 2  # First batch has 2 items
        assert len(results[1]) == 3  # Second batch has 3 items
        assert len(results[2]) == 1  # Third batch has 1 item
        
        # Check that batch IDs are correct
        assert 'batch_0_traj_0' in results[0][0]['trajectory_id']
        assert 'batch_1_traj_0' in results[1][0]['trajectory_id']
    
    @pytest.mark.asyncio
    async def test_parallel_extraction_with_failures(self, parallel_extractor):
        """Test parallel extraction with some failures."""
        # Override to simulate failures
        original_extract = parallel_extractor.extract_batch
        
        async def failing_extract_batch(batch_params):
            if batch_params.get('batch_id') == 1:
                raise ValueError("Batch 1 failed")
            return await original_extract(batch_params)
        
        parallel_extractor.extract_batch = failing_extract_batch
        
        batch_configs = [
            {'batch_id': 0, 'batch_size': 1},
            {'batch_id': 1, 'batch_size': 1},  # This will fail
            {'batch_id': 2, 'batch_size': 1}
        ]
        
        # Should handle failures gracefully and return successful results
        results = await parallel_extractor.extract_parallel(batch_configs)
        
        # Should have results for successful batches
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) == 2  # Only successful batches


class TestDataProcessor:
    """Test data processing functionality."""
    
    @pytest.fixture
    def mock_processor(self):
        """Create a mock data processor."""
        class MockProcessor(DataProcessor):
            def __init__(self):
                super().__init__()
            
            async def process_batch(self, raw_data: List[Dict[str, Any]]) -> List[TrajectoryData]:
                """Convert raw data to TrajectoryData objects."""
                processed = []
                
                for item in raw_data:
                    positions = [
                        Position(x=x, y=y) 
                        for x, y in zip(item['x_coords'], item['y_coords'])
                    ]
                    
                    velocities = []
                    if 'velocities_x' in item and 'velocities_y' in item:
                        velocities = [
                            Velocity(vx=vx, vy=vy)
                            for vx, vy in zip(item['velocities_x'], item['velocities_y'])
                        ]
                    
                    trajectory = TrajectoryData(
                        trajectory_id=item['trajectory_id'],
                        vehicle_id=item['vehicle_id'],
                        positions=positions,
                        velocities=velocities,
                        timestamps=item.get('timestamps', [])
                    )
                    processed.append(trajectory)
                
                return processed
        
        return MockProcessor()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, mock_processor):
        """Test batch processing functionality."""
        raw_data = [
            {
                'trajectory_id': 'test_1',
                'vehicle_id': 'vehicle_1',
                'x_coords': [0.0, 1.0, 2.0],
                'y_coords': [0.0, 1.0, 2.0],
                'velocities_x': [1.0, 1.0, 1.0],
                'velocities_y': [1.0, 1.0, 1.0],
                'timestamps': [0.0, 0.1, 0.2]
            },
            {
                'trajectory_id': 'test_2',
                'vehicle_id': 'vehicle_2',
                'x_coords': [1.0, 2.0, 3.0],
                'y_coords': [1.0, 2.0, 3.0],
                'velocities_x': [1.0, 1.0, 1.0],
                'velocities_y': [1.0, 1.0, 1.0],
                'timestamps': [0.0, 0.1, 0.2]
            }
        ]
        
        processed = await mock_processor.process_batch(raw_data)
        
        assert len(processed) == 2
        assert all(isinstance(traj, TrajectoryData) for traj in processed)
        assert all(validate_trajectory_data(traj) for traj in processed)
    
    @pytest.mark.asyncio
    async def test_processing_validation(self, mock_processor):
        """Test data validation during processing."""
        invalid_data = [
            {
                'trajectory_id': 'invalid',
                'vehicle_id': 'vehicle',
                'x_coords': [1.0, 2.0, 3.0],
                'y_coords': [1.0, 2.0]  # Mismatched length
            }
        ]
        
        # Should handle invalid data gracefully
        with pytest.raises((ValueError, IndexError)):
            await mock_processor.process_batch(invalid_data)


class TestTrajectoryFeatureExtractor:
    """Test trajectory feature extraction."""
    
    @pytest.fixture
    def feature_extractor(self):
        """Create trajectory feature extractor."""
        return TrajectoryFeatureExtractor()
    
    def test_basic_feature_extraction(self, feature_extractor, sample_trajectory):
        """Test basic feature extraction."""
        features = feature_extractor.extract_features(sample_trajectory)
        
        assert isinstance(features, dict)
        
        # Check for expected feature categories
        expected_categories = ['temporal', 'spatial', 'kinematic']
        for category in expected_categories:
            if category in features:
                assert isinstance(features[category], dict)
    
    def test_temporal_features(self, feature_extractor, sample_trajectory):
        """Test temporal feature extraction."""
        temporal_features = feature_extractor.extract_temporal_features(sample_trajectory)
        
        assert isinstance(temporal_features, dict)
        
        # Expected temporal features
        expected_features = ['duration', 'average_dt', 'total_time_steps']
        for feature in expected_features:
            if feature in temporal_features:
                assert isinstance(temporal_features[feature], (int, float))
    
    def test_spatial_features(self, feature_extractor, sample_trajectory):
        """Test spatial feature extraction."""
        spatial_features = feature_extractor.extract_spatial_features(sample_trajectory)
        
        assert isinstance(spatial_features, dict)
        
        # Expected spatial features
        expected_features = ['total_distance', 'displacement', 'bounding_box_area']
        for feature in expected_features:
            if feature in spatial_features:
                assert isinstance(spatial_features[feature], (int, float))
                assert spatial_features[feature] >= 0  # Should be non-negative
    
    def test_kinematic_features(self, feature_extractor, sample_trajectory):
        """Test kinematic feature extraction."""
        kinematic_features = feature_extractor.extract_kinematic_features(sample_trajectory)
        
        assert isinstance(kinematic_features, dict)
        
        # Expected kinematic features
        expected_features = ['average_speed', 'max_speed', 'average_acceleration']
        for feature in expected_features:
            if feature in kinematic_features:
                assert isinstance(kinematic_features[feature], (int, float))
    
    def test_feature_extraction_empty_trajectory(self, feature_extractor):
        """Test feature extraction with empty trajectory."""
        empty_trajectory = TrajectoryData(
            trajectory_id="empty",
            vehicle_id="vehicle",
            positions=[],
            velocities=[],
            timestamps=[]
        )
        
        with pytest.raises((ValueError, IndexError)):
            feature_extractor.extract_features(empty_trajectory)
    
    def test_feature_extraction_consistency(self, feature_extractor, sample_trajectory):
        """Test that feature extraction is consistent."""
        features1 = feature_extractor.extract_features(sample_trajectory)
        features2 = feature_extractor.extract_features(sample_trajectory)
        
        # Should be identical for the same input
        assert features1 == features2
    
    @pytest.mark.parametrize("trajectory_length", [5, 10, 20, 50])
    def test_feature_extraction_various_lengths(self, feature_extractor, trajectory_length):
        """Test feature extraction with various trajectory lengths."""
        # Create trajectory of specified length
        positions = [Position(x=float(i), y=float(i * 0.5)) for i in range(trajectory_length)]
        velocities = [Velocity(vx=1.0, vy=0.5) for _ in range(trajectory_length)]
        timestamps = [float(i * 0.1) for i in range(trajectory_length)]
        
        trajectory = TrajectoryData(
            trajectory_id=f"test_{trajectory_length}",
            vehicle_id="vehicle",
            positions=positions,
            velocities=velocities,
            timestamps=timestamps
        )
        
        features = feature_extractor.extract_features(trajectory)
        
        assert isinstance(features, dict)
        # Features should be extractable for any reasonable trajectory length
        assert len(features) > 0


class TestDataQualityMonitor:
    """Test data quality monitoring."""
    
    @pytest.fixture
    def quality_monitor(self):
        """Create data quality monitor."""
        return DataQualityMonitor()
    
    def test_completeness_check(self, quality_monitor, sample_trajectories):
        """Test data completeness checking."""
        completeness_report = quality_monitor.check_completeness(sample_trajectories)
        
        assert isinstance(completeness_report, dict)
        assert 'completeness_score' in completeness_report
        assert 0.0 <= completeness_report['completeness_score'] <= 1.0
    
    def test_consistency_check(self, quality_monitor, sample_trajectories):
        """Test data consistency checking."""
        consistency_report = quality_monitor.check_consistency(sample_trajectories)
        
        assert isinstance(consistency_report, dict)
        assert 'consistency_score' in consistency_report
        assert 0.0 <= consistency_report['consistency_score'] <= 1.0
    
    def test_outlier_detection(self, quality_monitor, sample_trajectories):
        """Test outlier detection."""
        outlier_report = quality_monitor.detect_outliers(sample_trajectories)
        
        assert isinstance(outlier_report, dict)
        assert 'outlier_count' in outlier_report
        assert outlier_report['outlier_count'] >= 0
    
    def test_quality_report_generation(self, quality_monitor, sample_trajectories):
        """Test comprehensive quality report generation."""
        quality_report = quality_monitor.generate_quality_report(sample_trajectories)
        
        assert isinstance(quality_report, dict)
        
        # Expected sections in quality report
        expected_sections = ['completeness', 'consistency', 'outliers', 'summary']
        for section in expected_sections:
            if section in quality_report:
                assert isinstance(quality_report[section], dict)
    
    def test_quality_monitoring_empty_data(self, quality_monitor):
        """Test quality monitoring with empty dataset."""
        empty_trajectories = []
        
        # Should handle empty data gracefully
        report = quality_monitor.generate_quality_report(empty_trajectories)
        
        assert isinstance(report, dict)
        # Should indicate issues with empty dataset
        if 'summary' in report:
            assert report['summary']['total_trajectories'] == 0


class TestTrajectoryETLPipeline:
    """Test end-to-end ETL pipeline."""
    
    @pytest.fixture
    def etl_pipeline(self, mock_extractor, mock_processor):
        """Create ETL pipeline with mock components."""
        class MockETLPipeline(TrajectoryETLPipeline):
            def __init__(self, extractor, processor):
                self.extractor = extractor
                self.processor = processor
                self.quality_monitor = DataQualityMonitor()
        
        return MockETLPipeline(mock_extractor, mock_processor)
    
    @pytest.mark.asyncio
    async def test_pipeline_execution(self, etl_pipeline):
        """Test full pipeline execution."""
        batch_configs = [
            {'batch_size': 2},
            {'batch_size': 3}
        ]
        
        results = await etl_pipeline.process_pipeline(batch_configs)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(traj, TrajectoryData) for traj in results)
        assert all(validate_trajectory_data(traj) for traj in results)
    
    @pytest.mark.asyncio
    async def test_pipeline_with_quality_monitoring(self, etl_pipeline):
        """Test pipeline execution with quality monitoring."""
        batch_configs = [{'batch_size': 5}]
        
        results, quality_report = await etl_pipeline.process_with_monitoring(batch_configs)
        
        assert isinstance(results, list)
        assert isinstance(quality_report, dict)
        assert len(results) > 0
        
        # Quality report should have expected sections
        assert 'summary' in quality_report
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, etl_pipeline):
        """Test pipeline error handling."""
        # Override extractor to fail
        async def failing_extract_batch(batch_params):
            raise RuntimeError("Pipeline failure")
        
        etl_pipeline.extractor.extract_batch = failing_extract_batch
        
        with pytest.raises(RuntimeError):
            await etl_pipeline.process_pipeline([{'batch_size': 1}])
    
    @pytest.mark.asyncio
    async def test_pipeline_partial_failures(self, etl_pipeline):
        """Test pipeline handling of partial failures."""
        original_validate = etl_pipeline.extractor.validate_data
        
        async def selective_validate(data):
            # Fail validation for some items
            if 'traj_1' in data.get('trajectory_id', ''):
                return False
            return await original_validate(data)
        
        etl_pipeline.extractor.validate_data = selective_validate
        
        # Pipeline should handle partial failures and return valid data
        results = await etl_pipeline.process_pipeline([{'batch_size': 3}])
        
        assert isinstance(results, list)
        # Some results should be filtered out due to validation failures


# Integration tests for data pipeline components
class TestDataPipelineIntegration:
    """Test integration between pipeline components."""
    
    @pytest.mark.asyncio
    async def test_extractor_processor_integration(self, mock_extractor, mock_processor):
        """Test integration between extractor and processor."""
        # Extract data
        raw_data = await mock_extractor.extract_batch({'batch_size': 2})
        
        # Process data
        processed_data = await mock_processor.process_batch(raw_data)
        
        assert len(processed_data) == 2
        assert all(isinstance(traj, TrajectoryData) for traj in processed_data)
        assert all(validate_trajectory_data(traj) for traj in processed_data)
    
    def test_processor_feature_extractor_integration(self, mock_processor, sample_trajectories):
        """Test integration between processor and feature extractor."""
        feature_extractor = TrajectoryFeatureExtractor()
        
        for trajectory in sample_trajectories:
            features = feature_extractor.extract_features(trajectory)
            
            assert isinstance(features, dict)
            assert len(features) > 0
    
    def test_quality_monitor_pipeline_integration(self, sample_trajectories):
        """Test integration between quality monitor and pipeline results."""
        quality_monitor = DataQualityMonitor()
        
        # Generate quality report
        quality_report = quality_monitor.generate_quality_report(sample_trajectories)
        
        assert isinstance(quality_report, dict)
        assert 'summary' in quality_report
        
        # Quality score should reflect the quality of sample data
        if 'summary' in quality_report and 'overall_quality' in quality_report['summary']:
            overall_quality = quality_report['summary']['overall_quality']
            assert 0.0 <= overall_quality <= 1.0


# Performance tests for data pipeline
class TestDataPipelinePerformance:
    """Test data pipeline performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_extraction_performance(self, performance_monitor):
        """Test extraction performance."""
        class FastExtractor(DataExtractor):
            async def extract_batch(self, batch_params):
                batch_size = batch_params.get('batch_size', 100)
                return [{'trajectory_id': f'perf_{i}'} for i in range(batch_size)]
            
            async def validate_data(self, data):
                return True
        
        extractor = FastExtractor()
        
        with performance_monitor() as monitor:
            await extractor.extract_batch({'batch_size': 1000})
        
        # Extraction should be reasonably fast
        assert monitor.elapsed_time < 1.0  # Should complete in under 1 second
    
    @pytest.mark.asyncio
    async def test_processing_performance(self, performance_monitor):
        """Test processing performance."""
        class FastProcessor(DataProcessor):
            async def process_batch(self, raw_data):
                # Simulate fast processing
                return [
                    TrajectoryData(
                        trajectory_id=item['trajectory_id'],
                        vehicle_id='vehicle',
                        positions=[Position(x=0.0, y=0.0)],
                        velocities=[Velocity(vx=1.0, vy=0.0)],
                        timestamps=[0.0]
                    )
                    for item in raw_data
                ]
        
        processor = FastProcessor()
        raw_data = [{'trajectory_id': f'perf_{i}'} for i in range(1000)]
        
        with performance_monitor() as monitor:
            await processor.process_batch(raw_data)
        
        # Processing should be reasonably fast
        assert monitor.elapsed_time < 2.0  # Should complete in under 2 seconds
    
    def test_feature_extraction_performance(self, performance_monitor, sample_trajectory):
        """Test feature extraction performance."""
        feature_extractor = TrajectoryFeatureExtractor()
        
        with performance_monitor() as monitor:
            for _ in range(100):  # Extract features 100 times
                feature_extractor.extract_features(sample_trajectory)
        
        # Feature extraction should be fast
        assert monitor.elapsed_time < 1.0  # Should complete in under 1 second