"""Example configuration and usage for Milestone 3 ETL Pipeline.

This demonstrates how to use the complete ETL pipeline with NGSIM data,
quality checks, incremental loading, and Parquet storage.
"""

from pathlib import Path

from trajectory_prediction.data import (
    DataQualityPipeline,
    DataSourceFactory,
    ETLConfig,
    IncrementalConfig,
    IncrementalETLPipeline,
    NGSIMDataSource,
    create_source_id,
)


async def process_ngsim_dataset_example():
    """Example of processing NGSIM dataset with full ETL pipeline."""

    # Configuration
    input_file = Path("data/ngsim/i80_sample.csv")  # Example path
    output_dir = Path("data/processed")

    # ETL Configuration
    etl_config = ETLConfig(
        chunk_size=1000,
        max_concurrent_downloads=5,
        partition_cols=["dataset_name", "date"],
        compression="snappy",
    )

    # Incremental Loading Configuration
    incremental_config = IncrementalConfig(
        checkpoint_dir=Path("checkpoints"),
        hash_tracker_file=Path("checkpoints/file_hashes.json"),
        enable_incremental=True,
        force_full_reload=False,
    )

    # Create incremental pipeline
    incremental_pipeline = IncrementalETLPipeline(etl_config, incremental_config)

    # Create source ID
    source_id = create_source_id(input_file, "ngsim")

    # Check if we should process this source
    if not incremental_pipeline.should_process_source(input_file, source_id):
        print(f"Skipping {input_file} - already processed and unchanged")
        return

    # Create NGSIM data source
    ngsim_config = {
        "min_trajectory_length": 10,
        "chunk_size": 1000,
        "max_concurrent": 5,
    }

    data_source = NGSIMDataSource(input_file, ngsim_config)

    # Validate source
    if not data_source.validate_source(input_file):
        print(f"Invalid NGSIM source: {input_file}")
        return

    # Create ETL pipeline
    pipeline = await data_source.create_etl_pipeline(output_dir)

    # Set up data quality pipeline (would be used during processing)
    # quality_pipeline = DataQualityPipeline()  # Uses default rules

    print(f"Starting ETL processing for {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Source ID: {source_id}")

    try:
        # Save initial progress checkpoint
        incremental_pipeline.save_progress_checkpoint(
            source_id, {"status": "started", "input_file": str(input_file)}
        )

        # Run ETL pipeline (this would be enhanced to include quality checks)
        await pipeline.run()

        # Mark processing complete
        incremental_pipeline.mark_processing_complete(
            source_id,
            input_file,
            {"status": "completed", "output_dir": str(output_dir)},
        )

        print(f"✅ Successfully processed {input_file}")

    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")
        # Could save error checkpoint here
        raise


def demonstrate_data_source_factory():
    """Demonstrate data source factory pattern."""

    # Register sources (done automatically on import)
    sources = DataSourceFactory.list_sources()
    print(f"Available data sources: {sources}")

    # Create source by type
    ngsim_source = DataSourceFactory.create_source(
        "ngsim", "data/sample.csv", {"min_trajectory_length": 5}
    )
    print(f"Created NGSIM source: {ngsim_source}")

    # Auto-detect source type
    try:
        auto_source = DataSourceFactory.auto_detect_source("data/sample.csv")
        print(f"Auto-detected source: {auto_source}")
    except ValueError as e:
        print(f"Auto-detection failed: {e}")


def demonstrate_data_quality():
    """Demonstrate data quality pipeline."""
    from trajectory_prediction.data.quality import (
        MinimumPointsRule,
        OutlierDetectionRule,
        PhysicsConsistencyRule,
        TemporalConsistencyRule,
    )

    # Create custom quality rules
    custom_rules = [
        MinimumPointsRule(min_points=15),  # Stricter minimum
        TemporalConsistencyRule(),
        PhysicsConsistencyRule(
            max_speed=150.0, max_acceleration=15.0
        ),  # Highway limits
        OutlierDetectionRule(position_threshold=500.0),  # 500m jump threshold
    ]

    # Create quality pipeline
    quality_pipeline = DataQualityPipeline(custom_rules)

    print("Custom data quality pipeline created with rules:")
    for rule in custom_rules:
        print(f"  - {rule.name}: {rule.description}")

    return quality_pipeline


def demonstrate_incremental_loading():
    """Demonstrate incremental loading capabilities."""

    # Configuration for incremental loading
    incremental_config = IncrementalConfig(
        checkpoint_dir=Path("checkpoints"), enable_incremental=True
    )

    etl_config = ETLConfig()
    pipeline = IncrementalETLPipeline(etl_config, incremental_config)

    # Get processing summary
    summary = pipeline.get_processing_summary()
    print("Processing Summary:")
    print(f"  Total sources: {summary['total_sources']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  In progress: {summary['in_progress']}")

    return pipeline


if __name__ == "__main__":
    print("=== Milestone 3: NGSIM Data Integration & ETL Pipeline ===")
    print()

    print("1. Data Source Factory:")
    demonstrate_data_source_factory()
    print()

    print("2. Data Quality Pipeline:")
    quality_pipeline = demonstrate_data_quality()
    print()

    print("3. Incremental Loading:")
    incremental_pipeline = demonstrate_incremental_loading()
    print()

    print("4. Full ETL Pipeline Example:")
    print("   (Would process NGSIM data if input file exists)")
    # asyncio.run(process_ngsim_dataset_example())

    print()
    print("✅ Milestone 3 demonstration completed!")
    print()
    print("Key Features Implemented:")
    print("  ✅ Async ETL pipeline using asyncio and aiohttp")
    print("  ✅ NGSIM dataset integration with proper schema mapping")
    print("  ✅ Parquet storage with optimal partitioning strategy")
    print("  ✅ Data source factory pattern for multiple dataset support")
    print("  ✅ Incremental data loading capabilities")
    print("  ✅ Data quality pipeline with cleaning and validation")
