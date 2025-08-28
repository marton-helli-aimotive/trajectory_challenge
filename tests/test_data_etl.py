"""Tests for ETL pipeline components."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import shutil
import os

from vehicle_trajectory_prediction.data.etl import AsyncETLPipeline, ETLConfig
from vehicle_trajectory_prediction.data.datasets import (
    NGSIMDataset, DatasetFactory, DatasetConfig, BaseDataset
)
from vehicle_trajectory_prediction.data.storage import ParquetStorage, StorageConfig
from vehicle_trajectory_prediction.data.quality import (
    DataQualityPipeline, QualityConfig, QualityMetrics
)
from vehicle_trajectory_prediction.core.models import TrajectoryPoint, Trajectory, TrajectoryDataset
from vehicle_trajectory_prediction.core.exceptions import (
    ETLPipelineError, DataSourceError, StorageError, DataQualityError
)


class TestETLConfig:
    """Test ETL configuration."""
    
    def test_etl_config_defaults(self):
        """Test ETL config default values."""
        config = ETLConfig()
        
        assert config.max_concurrent_requests == 10
        assert config.request_timeout == 30
        assert config.retry_attempts == 3
        assert config.batch_size == 1000
        assert config.enable_progress_bars is True
    
    def test_etl_config_custom_values(self):
        """Test ETL config with custom values."""
        config = ETLConfig(
            max_concurrent_requests=5,
            batch_size=500,
            enable_progress_bars=False
        )
        
        assert config.max_concurrent_requests == 5
        assert config.batch_size == 500
        assert config.enable_progress_bars is False


class TestAsyncETLPipeline:
    """Test async ETL pipeline."""
    
    @pytest.fixture
    def etl_config(self):
        """Create ETL config for testing."""
        return ETLConfig(
            max_concurrent_requests=2,
            batch_size=100,
            enable_progress_bars=False
        )
    
    @pytest.fixture
    def pipeline(self, etl_config):
        """Create ETL pipeline for testing."""
        return AsyncETLPipeline(etl_config)
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.config.max_concurrent_requests == 2
        assert pipeline.config.batch_size == 100
        assert pipeline.temp_dir.exists()
    
    @pytest.mark.asyncio
    async def test_download_file_success(self, pipeline):
        """Test successful file download."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.content.iter_chunked.return_value = [b"test data"]
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            with patch('aiofiles.open') as mock_file:
                mock_file.return_value.__aenter__.return_value.write = AsyncMock()
                
                result = await pipeline.download_file("http://example.com/test.csv", Path("/tmp/test.csv"))
                
                assert result is True
    
    @pytest.mark.asyncio
    async def test_download_file_failure(self, pipeline):
        """Test file download failure."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = Exception("Network error")
            
            result = await pipeline.download_file("http://example.com/test.csv", Path("/tmp/test.csv"))
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_process_file_batch(self, pipeline):
        """Test file batch processing."""
        # Create test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test CSV file
            test_data = pd.DataFrame({
                'Vehicle_ID': ['1', '2', '3'],
                'Frame_ID': [1, 2, 3],
                'Local_X': [0.0, 1.0, 2.0],
                'Local_Y': [0.0, 1.0, 2.0],
                'v_Vel': [10.0, 11.0, 12.0]
            })
            
            csv_file = temp_path / "test.csv"
            test_data.to_csv(csv_file, index=False)
            
            # Process file batch
            results = await pipeline.process_file_batch([csv_file])
            
            assert len(results) == 1
            assert len(results[0]) == 3
            assert list(results[0].columns) == ['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel']
    
    @pytest.mark.asyncio
    async def test_incremental_load(self, pipeline):
        """Test incremental loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files with different modification times
            file1 = temp_path / "file1.csv"
            file2 = temp_path / "file2.csv"
            
            file1.touch()
            file2.touch()
            
            # Set different modification times
            import time
            old_time = time.time() - 3600  # 1 hour ago
            os.utime(file1, (old_time, old_time))
            
            # Test incremental load
            last_processed = pd.Timestamp.now() - pd.Timedelta(minutes=30)
            files_to_process = await pipeline.incremental_load([file1, file2], last_processed)
            
            # Only file2 should be processed (newer than last_processed)
            assert len(files_to_process) == 1
            assert files_to_process[0] == file2


class TestDatasetConfig:
    """Test dataset configuration."""
    
    def test_dataset_config_defaults(self):
        """Test dataset config default values."""
        config = DatasetConfig()
        
        assert config.min_trajectory_length == 10
        assert config.max_trajectory_length == 1000
        assert config.time_resolution == 0.1
        assert config.max_velocity == 50.0
        assert config.max_acceleration == 10.0
        assert len(config.ngsim_columns) > 0
    
    def test_dataset_config_custom_values(self):
        """Test dataset config with custom values."""
        config = DatasetConfig(
            min_trajectory_length=5,
            max_velocity=30.0,
            min_x=0.0,
            max_x=1000.0
        )
        
        assert config.min_trajectory_length == 5
        assert config.max_velocity == 30.0
        assert config.min_x == 0.0
        assert config.max_x == 1000.0


class TestNGSIMDataset:
    """Test NGSIM dataset loader."""
    
    @pytest.fixture
    def dataset_config(self):
        """Create dataset config for testing."""
        return DatasetConfig(
            min_trajectory_length=3,
            max_trajectory_length=100,
            time_resolution=0.1
        )
    
    @pytest.fixture
    def sample_ngsim_data(self):
        """Create sample NGSIM data."""
        return pd.DataFrame({
            'Vehicle_ID': ['1', '1', '1', '2', '2', '2'],
            'Frame_ID': [1, 2, 3, 1, 2, 3],
            'Total_Frames': [3, 3, 3, 3, 3, 3],
            'Global_Time': [0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
            'Local_X': [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            'Local_Y': [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            'Global_X': [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            'Global_Y': [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            'v_length': [4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
            'v_Width': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            'v_Class': [1, 1, 1, 1, 1, 1],
            'v_Vel': [10.0, 11.0, 12.0, 15.0, 16.0, 17.0],
            'v_Acc': [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'Lane_ID': [1, 1, 1, 2, 2, 2],
            'Preceding': [0, 0, 0, 0, 0, 0],
            'Following': [0, 0, 0, 0, 0, 0],
            'Space_Headway': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'Time_Headway': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
    
    def test_ngsim_dataset_initialization(self, dataset_config):
        """Test NGSIM dataset initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = NGSIMDataset(dataset_config, data_path=temp_dir)
            
            assert dataset.config == dataset_config
            assert dataset.data_path == Path(temp_dir)
            assert dataset.metadata['dataset_type'] == 'NGSIM'
    
    def test_ngsim_dataset_load_data_missing_directory(self, dataset_config):
        """Test NGSIM dataset loading with missing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a non-existent subdirectory
            data_path = Path(temp_dir) / "nonexistent"
            dataset = NGSIMDataset(dataset_config, data_path=data_path)
            
            with pytest.raises(DataSourceError, match="NGSIM data directory does not exist"):
                dataset.load_data()
    
    def test_ngsim_dataset_load_data_no_files(self, dataset_config):
        """Test NGSIM dataset loading with no CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = NGSIMDataset(dataset_config, data_path=temp_dir)
            
            with pytest.raises(DataSourceError, match="No CSV files found"):
                dataset.load_data()
    
    def test_ngsim_dataset_load_data_success(self, dataset_config, sample_ngsim_data):
        """Test successful NGSIM dataset loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test CSV file
            csv_file = Path(temp_dir) / "test_data.csv"
            sample_ngsim_data.to_csv(csv_file, index=False)
            
            dataset = NGSIMDataset(dataset_config, data_path=temp_dir)
            data = dataset.load_data()
            
            assert len(data) == 6
            assert list(data.columns) == dataset_config.ngsim_columns + ['source_file']
            assert data['source_file'].iloc[0] == "test_data.csv"
    
    def test_ngsim_dataset_preprocess_data(self, dataset_config, sample_ngsim_data):
        """Test NGSIM data preprocessing."""
        dataset = NGSIMDataset(dataset_config)
        
        # Add some missing values
        sample_ngsim_data.loc[0, 'v_Acc'] = np.nan
        sample_ngsim_data.loc[1, 'v_Heading'] = np.nan
        
        processed_data = dataset.preprocess_data(sample_ngsim_data)
        
        assert len(processed_data) == 6
        assert processed_data['v_Acc'].isna().sum() == 0  # Should be filled
        assert 'v_Heading' in processed_data.columns  # Should be added if missing
        assert processed_data['Vehicle_ID'].dtype == 'object'
        assert processed_data['Local_X'].dtype == 'float64'
    
    def test_ngsim_dataset_to_trajectories(self, dataset_config, sample_ngsim_data):
        """Test conversion to trajectories."""
        dataset = NGSIMDataset(dataset_config)
        
        # Preprocess data first
        processed_data = dataset.preprocess_data(sample_ngsim_data)
        
        # Convert to trajectories
        trajectory_dataset = dataset.to_trajectories(processed_data)
        
        assert isinstance(trajectory_dataset, TrajectoryDataset)
        assert len(trajectory_dataset.trajectories) == 2  # Two vehicles
        assert trajectory_dataset.name == "NGSIMDataset_Dataset"
        
        # Check first trajectory
        trajectory = trajectory_dataset.trajectories[0]
        assert trajectory.vehicle_id == "1"
        assert len(trajectory.points) == 3
        assert trajectory.points[0].x == 0.0
        assert trajectory.points[0].y == 0.0
        assert trajectory.points[0].velocity == 10.0


class TestDatasetFactory:
    """Test dataset factory."""
    
    def test_factory_list_datasets(self):
        """Test listing available datasets."""
        datasets = DatasetFactory.list_datasets()
        
        assert 'ngsim' in datasets
        assert len(datasets) >= 1
    
    def test_factory_create_dataset_success(self):
        """Test successful dataset creation."""
        config = DatasetConfig()
        dataset = DatasetFactory.create_dataset('ngsim', config)
        
        assert isinstance(dataset, NGSIMDataset)
        assert dataset.config == config
    
    def test_factory_create_dataset_unknown(self):
        """Test creating unknown dataset type."""
        config = DatasetConfig()
        
        with pytest.raises(DataSourceError, match="Unknown dataset type"):
            DatasetFactory.create_dataset('unknown', config)
    
    def test_factory_register_dataset(self):
        """Test registering new dataset type."""
        class TestDataset(BaseDataset):
            def load_data(self):
                return pd.DataFrame()
            
            def preprocess_data(self, data):
                return data
        
        # Register new dataset
        DatasetFactory.register_dataset('test', TestDataset)
        
        # Verify it's available
        datasets = DatasetFactory.list_datasets()
        assert 'test' in datasets
        
        # Test creation
        config = DatasetConfig()
        dataset = DatasetFactory.create_dataset('test', config)
        assert isinstance(dataset, TestDataset)
    
    def test_factory_get_dataset_info(self):
        """Test getting dataset information."""
        info = DatasetFactory.get_dataset_info('ngsim')
        
        assert info['name'] == 'ngsim'
        assert info['class'] == 'NGSIMDataset'
        assert 'description' in info
        assert 'config_fields' in info


class TestStorageConfig:
    """Test storage configuration."""
    
    def test_storage_config_defaults(self):
        """Test storage config default values."""
        config = StorageConfig()
        
        assert config.base_path == "data/storage"
        assert config.compression == "snappy"
        assert config.row_group_size == 100000
        assert config.enable_versioning is True
        assert config.use_threads is True
        assert len(config.partition_by) == 2
        assert len(config.index_columns) == 3
    
    def test_storage_config_custom_values(self):
        """Test storage config with custom values."""
        config = StorageConfig(
            base_path="custom/path",
            compression="gzip",
            enable_versioning=False,
            partition_by=['vehicle_id']
        )
        
        assert config.base_path == "custom/path"
        assert config.compression == "gzip"
        assert config.enable_versioning is False
        assert config.partition_by == ['vehicle_id']


class TestParquetStorage:
    """Test Parquet storage."""
    
    @pytest.fixture
    def storage_config(self):
        """Create storage config for testing."""
        return StorageConfig(
            base_path="test_storage",
            partition_by=['vehicle_id'],
            enable_versioning=False
        )
    
    @pytest.fixture
    def storage(self, storage_config):
        """Create storage instance for testing."""
        return ParquetStorage(storage_config)
    
    @pytest.fixture
    def sample_trajectories(self):
        """Create sample trajectories for testing."""
        points1 = [
            TrajectoryPoint(x=0.0, y=0.0, timestamp=0.0, velocity=10.0, vehicle_id="1"),
            TrajectoryPoint(x=1.0, y=1.0, timestamp=0.1, velocity=11.0, vehicle_id="1"),
            TrajectoryPoint(x=2.0, y=2.0, timestamp=0.2, velocity=12.0, vehicle_id="1"),
        ]
        
        points2 = [
            TrajectoryPoint(x=10.0, y=10.0, timestamp=0.0, velocity=15.0, vehicle_id="2"),
            TrajectoryPoint(x=11.0, y=11.0, timestamp=0.1, velocity=16.0, vehicle_id="2"),
            TrajectoryPoint(x=12.0, y=12.0, timestamp=0.2, velocity=17.0, vehicle_id="2"),
        ]
        
        trajectory1 = Trajectory(points=points1, vehicle_id="1")
        trajectory2 = Trajectory(points=points2, vehicle_id="2")
        
        return [trajectory1, trajectory2]
    
    def test_storage_initialization(self, storage):
        """Test storage initialization."""
        assert storage.config.base_path == Path("test_storage")
        assert storage.data_path.exists()
        assert storage.metadata_path.exists()
        assert storage.index_path.exists()
    
    def test_storage_store_trajectories(self, storage, sample_trajectories):
        """Test storing trajectories."""
        metadata = storage.store_trajectories(sample_trajectories, "test_dataset", "v1.0")
        
        assert metadata['dataset_name'] == "test_dataset"
        assert metadata['version'] == "v1.0"
        assert metadata['total_trajectories'] == 2
        assert metadata['total_points'] == 6
        assert len(metadata['stored_files']) == 2  # One file per vehicle
    
    def test_storage_load_trajectories(self, storage, sample_trajectories):
        """Test loading trajectories."""
        # First store trajectories
        storage.store_trajectories(sample_trajectories, "test_dataset", "v1.0")
        
        # Then load them
        df = storage.load_trajectories("test_dataset", "v1.0")
        
        assert len(df) == 6
        assert 'vehicle_id' in df.columns
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'timestamp' in df.columns
        assert 'velocity' in df.columns
    
    def test_storage_load_trajectories_with_filters(self, storage, sample_trajectories):
        """Test loading trajectories with filters."""
        # First store trajectories
        storage.store_trajectories(sample_trajectories, "test_dataset", "v1.0")
        
        # Load with vehicle filter
        df = storage.load_trajectories("test_dataset", "v1.0", filters={'vehicle_id': '1'})
        
        assert len(df) == 3
        assert all(df['vehicle_id'] == '1')
    
    def test_storage_list_versions(self, storage, sample_trajectories):
        """Test listing versions."""
        # Store multiple versions
        storage.store_trajectories(sample_trajectories, "test_dataset", "v1.0")
        storage.store_trajectories(sample_trajectories, "test_dataset", "v2.0")
        
        versions = storage.list_versions("test_dataset")
        
        assert len(versions) == 2
        version_names = [v['version'] for v in versions]
        assert 'v1.0' in version_names
        assert 'v2.0' in version_names
    
    def test_storage_get_stats(self, storage, sample_trajectories):
        """Test getting storage statistics."""
        # Store some data first
        storage.store_trajectories(sample_trajectories, "test_dataset", "v1.0")
        
        stats = storage.get_storage_stats()
        
        assert 'base_path' in stats
        assert 'total_data_files' in stats
        assert 'total_size_bytes' in stats
        assert 'total_size_mb' in stats
        assert stats['total_data_files'] > 0
    
    def test_storage_optimization(self, storage, sample_trajectories):
        """Test storage optimization."""
        # Store some data first
        storage.store_trajectories(sample_trajectories, "test_dataset", "v1.0")
        
        # Run optimization
        stats = storage.optimize_storage()
        
        assert 'optimized_files' in stats
        assert 'saved_space_bytes' in stats
        assert 'saved_space_mb' in stats


class TestQualityConfig:
    """Test quality configuration."""
    
    def test_quality_config_defaults(self):
        """Test quality config default values."""
        config = QualityConfig()
        
        assert config.min_completeness == 0.8
        assert config.max_velocity == 50.0
        assert config.max_acceleration == 10.0
        assert config.enable_quality_scoring is True
        assert config.enable_cleaning is True
        assert config.generate_reports is True
    
    def test_quality_config_custom_values(self):
        """Test quality config with custom values."""
        config = QualityConfig(
            min_completeness=0.9,
            max_velocity=30.0,
            enable_cleaning=False,
            outlier_threshold=2.5
        )
        
        assert config.min_completeness == 0.9
        assert config.max_velocity == 30.0
        assert config.enable_cleaning is False
        assert config.outlier_threshold == 2.5


class TestDataQualityPipeline:
    """Test data quality pipeline."""
    
    @pytest.fixture
    def quality_config(self):
        """Create quality config for testing."""
        return QualityConfig(
            min_trajectory_length=3,
            max_trajectory_length=100,
            max_velocity=30.0,
            enable_cleaning=True
        )
    
    @pytest.fixture
    def quality_pipeline(self, quality_config):
        """Create quality pipeline for testing."""
        return DataQualityPipeline(quality_config)
    
    @pytest.fixture
    def sample_trajectory_dataset(self):
        """Create sample trajectory dataset for testing."""
        # Create valid trajectory
        valid_points = [
            TrajectoryPoint(x=0.0, y=0.0, timestamp=0.0, velocity=10.0, vehicle_id="1"),
            TrajectoryPoint(x=1.0, y=1.0, timestamp=0.1, velocity=11.0, vehicle_id="1"),
            TrajectoryPoint(x=2.0, y=2.0, timestamp=0.2, velocity=12.0, vehicle_id="1"),
        ]
        
        # Create invalid trajectory (too short)
        invalid_points = [
            TrajectoryPoint(x=0.0, y=0.0, timestamp=0.0, velocity=10.0, vehicle_id="2"),
            TrajectoryPoint(x=1.0, y=1.0, timestamp=0.1, velocity=11.0, vehicle_id="2"),
        ]
        
        trajectory1 = Trajectory(points=valid_points, vehicle_id="1")
        trajectory2 = Trajectory(points=invalid_points, vehicle_id="2")
        
        return TrajectoryDataset(
            trajectories=[trajectory1, trajectory2],
            name="test_dataset",
            description="Test dataset",
            source="test",
            version="1.0.0"
        )
    
    def test_quality_pipeline_initialization(self, quality_pipeline):
        """Test quality pipeline initialization."""
        assert quality_pipeline.config.min_trajectory_length == 3
        assert quality_pipeline.config.max_velocity == 30.0
        assert quality_pipeline.report_path.exists()
    
    def test_validate_trajectory_dataset(self, quality_pipeline, sample_trajectory_dataset):
        """Test trajectory dataset validation."""
        metrics = quality_pipeline.validate_trajectory_dataset(sample_trajectory_dataset)
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.total_trajectories == 2
        assert metrics.valid_trajectories == 1  # Only one trajectory meets length requirement
        assert metrics.completeness_rate == 0.5
        assert metrics.overall_quality_score >= 0.0
        assert metrics.overall_quality_score <= 1.0
    
    def test_clean_trajectory_dataset(self, quality_pipeline, sample_trajectory_dataset):
        """Test trajectory dataset cleaning."""
        cleaned_dataset = quality_pipeline.clean_trajectory_dataset(sample_trajectory_dataset)
        
        assert isinstance(cleaned_dataset, TrajectoryDataset)
        assert len(cleaned_dataset.trajectories) == 1  # Only valid trajectory remains
        assert cleaned_dataset.name == "test_dataset_cleaned"
        assert cleaned_dataset.trajectories[0].vehicle_id == "1"
    
    def test_generate_quality_report(self, quality_pipeline, sample_trajectory_dataset):
        """Test quality report generation."""
        metrics = quality_pipeline.validate_trajectory_dataset(sample_trajectory_dataset)
        report = quality_pipeline.generate_quality_report(metrics, "test_dataset")
        
        assert 'dataset_name' in report
        assert 'summary' in report
        assert 'metrics' in report
        assert 'quality_scores' in report
        assert 'recommendations' in report
        assert report['dataset_name'] == "test_dataset"
        assert report['summary']['total_trajectories'] == 2
        assert report['summary']['valid_trajectories'] == 1


# Cleanup after tests
def teardown_module(module):
    """Clean up test files after tests."""
    import shutil
    
    # Remove test directories
    test_dirs = ["test_storage", "reports"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)