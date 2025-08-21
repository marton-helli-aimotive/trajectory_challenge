"""Tests for Milestone 3: NGSIM Data Integration & ETL Pipeline."""

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.data import (
    DataQualityPipeline,
    ETLConfig,
    IncrementalConfig,
    IncrementalETLPipeline,
    NGSIMDataExtractor,
    NGSIMDataSource,
    NGSIMDataTransformer,
)


def create_sample_ngsim_data() -> pd.DataFrame:
    """Create sample NGSIM-like data for testing."""
    np.random.seed(42)

    # Create data for 3 vehicles with multiple trajectory points
    data = []
    for vehicle_id in [1, 2, 3]:
        for frame in range(10):
            data.append(
                {
                    "Vehicle_ID": vehicle_id,
                    "Frame_ID": frame,
                    "Global_Time": 1000 + frame * 100,  # 0.1 second intervals
                    "Local_X": 10.0 + frame * 2.0 + np.random.normal(0, 0.1),
                    "Local_Y": 5.0 + vehicle_id * 3.0 + np.random.normal(0, 0.1),
                    "v_Vel": 20.0 + np.random.normal(0, 1.0),
                    "v_Acc": np.random.normal(0, 0.5),
                    "v_Class": 2,  # Car
                    "v_Length": 4.5,
                    "v_Width": 1.8,
                    "Lane_ID": vehicle_id,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_ngsim_file():
    """Create temporary NGSIM CSV file for testing."""
    df = create_sample_ngsim_data()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestAsyncETLPipeline:
    """Test async ETL pipeline functionality."""

    def test_ngsim_data_extractor(self, sample_ngsim_file):
        """Test NGSIM data extraction."""
        config = ETLConfig(chunk_size=100)
        extractor = NGSIMDataExtractor(config, [sample_ngsim_file])

        async def test_extraction():
            records = []
            async for record in extractor.extract():
                records.append(record)
            return records

        records = asyncio.run(test_extraction())

        # Should extract all 30 records (3 vehicles * 10 frames)
        assert len(records) == 30
        assert all("Vehicle_ID" in record for record in records)
        assert all("Local_X" in record for record in records)

    def test_ngsim_data_transformer(self, sample_ngsim_file):
        """Test NGSIM data transformation."""
        config = ETLConfig(chunk_size=100)
        extractor = NGSIMDataExtractor(config, [sample_ngsim_file])
        transformer = NGSIMDataTransformer(config, {"min_trajectory_length": 5})

        async def test_transformation():
            # Extract data
            records = []
            async for record in extractor.extract():
                records.append(record)

            # Transform in batch
            trajectories = []
            async for trajectory in transformer._transform_batch(records):
                trajectories.append(trajectory)

            return trajectories

        trajectories = asyncio.run(test_transformation())

        # Should create 3 trajectories (one per vehicle)
        assert len(trajectories) == 3

        # Check trajectory structure
        for trajectory in trajectories:
            assert len(trajectory.points) == 10  # 10 frames per vehicle
            assert trajectory.vehicle.vehicle_id in [1, 2, 3]
            assert trajectory.dataset_name == "ngsim"


class TestNGSIMDataSource:
    """Test NGSIM data source functionality."""

    def test_data_source_validation(self, sample_ngsim_file):
        """Test data source validation."""
        data_source = NGSIMDataSource(sample_ngsim_file)

        # Should validate CSV files
        assert data_source.validate_source(sample_ngsim_file) is True

        # Should reject non-CSV files
        fake_file = Path("test.txt")
        assert data_source.validate_source(fake_file) is False

    def test_data_source_metadata(self, sample_ngsim_file):
        """Test data source metadata."""
        data_source = NGSIMDataSource(sample_ngsim_file)
        metadata = data_source.get_metadata()

        assert metadata["source_type"] == "ngsim"
        assert ".csv" in metadata["supported_formats"]

    def test_etl_pipeline_creation(self, sample_ngsim_file, temp_output_dir):
        """Test ETL pipeline creation."""
        data_source = NGSIMDataSource(sample_ngsim_file)

        async def test_pipeline():
            pipeline = await data_source.create_etl_pipeline(temp_output_dir)
            return pipeline

        pipeline = asyncio.run(test_pipeline())
        assert pipeline is not None


class TestDataQuality:
    """Test data quality pipeline."""

    def test_data_quality_pipeline(self):
        """Test data quality pipeline with rules."""
        from trajectory_prediction.data.models import (
            Trajectory,
            TrajectoryPoint,
            Vehicle,
            VehicleType,
        )
        from trajectory_prediction.data.quality import (
            MinimumPointsRule,
            TemporalConsistencyRule,
        )

        # Create test trajectory
        vehicle = Vehicle(
            vehicle_id=1, vehicle_type=VehicleType.CAR, length=4.5, width=1.8
        )

        points = [
            TrajectoryPoint(
                timestamp=i,
                x=float(i),
                y=1.0,
                speed=10.0,
                velocity_x=10.0,
                velocity_y=0.0,
                acceleration_x=0.0,
                acceleration_y=0.0,
            )
            for i in range(15)  # 15 points - above minimum
        ]

        trajectory = Trajectory(
            trajectory_id="test_traj_1",
            vehicle=vehicle,
            points=points,
            dataset_name="test",
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )

        # Test quality pipeline
        rules = [MinimumPointsRule(min_points=10), TemporalConsistencyRule()]

        pipeline = DataQualityPipeline(rules)
        result = pipeline.process_trajectory(trajectory)

        assert result is not None
        assert result.trajectory_id == "test_traj_1"

        # Check statistics
        stats = pipeline.get_statistics()
        assert stats["total_processed"] == 1
        assert stats["passed_checks"] == 1


class TestIncrementalLoading:
    """Test incremental loading capabilities."""

    def test_checkpoint_manager(self, temp_output_dir):
        """Test checkpoint management."""
        from trajectory_prediction.data.incremental import CheckpointManager

        checkpoint_dir = temp_output_dir / "checkpoints"
        manager = CheckpointManager(checkpoint_dir)

        # Test saving and loading checkpoint
        test_data = {"processed_records": 100, "last_timestamp": "2024-01-01"}
        manager.save_checkpoint("test_source", test_data)

        loaded_data = manager.load_checkpoint("test_source")
        assert loaded_data["processed_records"] == 100

        # Test listing checkpoints
        checkpoints = manager.list_checkpoints()
        assert "test_source" in checkpoints

    def test_file_hash_tracker(self, temp_output_dir, sample_ngsim_file):
        """Test file hash tracking."""
        from trajectory_prediction.data.incremental import FileHashTracker

        tracker_file = temp_output_dir / "hashes.json"
        tracker = FileHashTracker(tracker_file)

        # First check should indicate change (new file)
        assert tracker.has_file_changed(sample_ngsim_file) is True

        # Second check should indicate no change
        assert tracker.has_file_changed(sample_ngsim_file) is False

    def test_incremental_pipeline(self, temp_output_dir, sample_ngsim_file):
        """Test incremental ETL pipeline."""
        etl_config = ETLConfig()
        incremental_config = IncrementalConfig(
            checkpoint_dir=temp_output_dir / "checkpoints",
            hash_tracker_file=temp_output_dir / "hashes.json",
        )

        pipeline = IncrementalETLPipeline(etl_config, incremental_config)

        # Test source processing decision
        source_id = "test_ngsim_source"

        # First time should process
        assert pipeline.should_process_source(sample_ngsim_file, source_id) is True

        # Mark as complete
        pipeline.mark_processing_complete(source_id, sample_ngsim_file, {"records": 30})

        # Second time should skip
        assert pipeline.should_process_source(sample_ngsim_file, source_id) is False


class TestParquetStorage:
    """Test Parquet storage functionality."""

    def test_parquet_partitioning(self, sample_ngsim_file, temp_output_dir):
        """Test Parquet storage with partitioning."""
        # This is a basic test - full implementation would test actual Parquet writing
        config = ETLConfig(
            partition_cols=["dataset_name", "date"], compression="snappy"
        )

        assert config.partition_cols == ["dataset_name", "date"]
        assert config.compression == "snappy"


def test_milestone_3_integration(sample_ngsim_file, temp_output_dir):
    """Integration test for complete Milestone 3 functionality."""
    # Test complete workflow

    # 1. Create data source
    data_source = NGSIMDataSource(sample_ngsim_file, {"min_trajectory_length": 5})

    # 2. Test validation
    assert data_source.validate_source(sample_ngsim_file)

    # 3. Create ETL pipeline
    async def test_full_pipeline():
        pipeline = await data_source.create_etl_pipeline(temp_output_dir)
        assert pipeline is not None
        return True

    result = asyncio.run(test_full_pipeline())
    assert result is True

    # 4. Test data quality
    quality_pipeline = DataQualityPipeline()
    assert quality_pipeline is not None

    # 5. Test incremental loading
    incremental_config = IncrementalConfig(
        checkpoint_dir=temp_output_dir / "checkpoints"
    )
    incremental_pipeline = IncrementalETLPipeline(ETLConfig(), incremental_config)
    assert incremental_pipeline is not None

    print("✅ Milestone 3 integration test passed!")


if __name__ == "__main__":
    # Quick manual test
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample data
        df = create_sample_ngsim_data()
        csv_file = Path(temp_dir) / "sample.csv"
        df.to_csv(csv_file, index=False)

        # Run integration test
        test_milestone_3_integration(csv_file, Path(temp_dir))
        print("Manual test completed successfully!")
