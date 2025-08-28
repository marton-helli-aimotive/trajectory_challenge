"""Integration module demonstrating complete ETL pipeline workflow."""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .etl import AsyncETLPipeline, ETLConfig
from .datasets import DatasetFactory, DatasetConfig, NGSIMDataset
from .storage import ParquetStorage, StorageConfig
from .quality import DataQualityPipeline, QualityConfig
from ..core.logging import get_logger

logger = get_logger(__name__)


class CompleteETLPipeline:
    """Complete ETL pipeline integrating all components."""
    
    def __init__(self, 
                 etl_config: Optional[ETLConfig] = None,
                 dataset_config: Optional[DatasetConfig] = None,
                 storage_config: Optional[StorageConfig] = None,
                 quality_config: Optional[QualityConfig] = None):
        """Initialize complete ETL pipeline.
        
        Args:
            etl_config: ETL configuration
            dataset_config: Dataset configuration
            storage_config: Storage configuration
            quality_config: Quality configuration
        """
        self.etl_config = etl_config or ETLConfig()
        self.dataset_config = dataset_config or DatasetConfig()
        self.storage_config = storage_config or StorageConfig()
        self.quality_config = quality_config or QualityConfig()
        
        # Initialize components
        self.etl_pipeline = AsyncETLPipeline(self.etl_config)
        self.storage = ParquetStorage(self.storage_config)
        self.quality_pipeline = DataQualityPipeline(self.quality_config)
        
        logger.info("Initialized complete ETL pipeline")
    
    async def process_ngsim_dataset(self, 
                                  data_path: str,
                                  dataset_name: str = "ngsim",
                                  version: Optional[str] = None,
                                  enable_quality_validation: bool = True,
                                  enable_cleaning: bool = True) -> Dict[str, Any]:
        """Process NGSIM dataset through complete pipeline.
        
        Args:
            data_path: Path to NGSIM data files
            dataset_name: Name for the dataset
            version: Optional version identifier
            enable_quality_validation: Whether to run quality validation
            enable_cleaning: Whether to clean the data
            
        Returns:
            Pipeline results summary
        """
        logger.info("Starting NGSIM dataset processing", 
                   data_path=data_path,
                   dataset_name=dataset_name,
                   version=version)
        
        results = {
            'dataset_name': dataset_name,
            'version': version or datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().isoformat(),
            'stages': {}
        }
        
        try:
            # Stage 1: Load dataset
            logger.info("Stage 1: Loading NGSIM dataset")
            dataset = await self._load_ngsim_dataset(data_path)
            results['stages']['loading'] = {
                'status': 'completed',
                'trajectories': len(dataset.trajectories),
                'total_points': sum(len(t.points) for t in dataset.trajectories)
            }
            
            # Stage 2: Quality validation
            if enable_quality_validation:
                logger.info("Stage 2: Quality validation")
                quality_metrics = self.quality_pipeline.validate_trajectory_dataset(dataset)
                quality_report = self.quality_pipeline.generate_quality_report(
                    quality_metrics, dataset_name
                )
                results['stages']['quality_validation'] = {
                    'status': 'completed',
                    'overall_quality_score': quality_metrics.overall_quality_score,
                    'completeness_rate': quality_metrics.completeness_rate,
                    'valid_trajectories': quality_metrics.valid_trajectories,
                    'report': quality_report
                }
                
                # Stage 3: Data cleaning
                if enable_cleaning:
                    logger.info("Stage 3: Data cleaning")
                    cleaned_dataset = self.quality_pipeline.clean_trajectory_dataset(dataset)
                    results['stages']['cleaning'] = {
                        'status': 'completed',
                        'original_trajectories': len(dataset.trajectories),
                        'cleaned_trajectories': len(cleaned_dataset.trajectories),
                        'removed_trajectories': len(dataset.trajectories) - len(cleaned_dataset.trajectories)
                    }
                    dataset = cleaned_dataset
            
            # Stage 4: Storage
            logger.info("Stage 4: Storing dataset")
            storage_metadata = self.storage.store_trajectories(
                dataset.trajectories, dataset_name, version
            )
            results['stages']['storage'] = {
                'status': 'completed',
                'stored_files': len(storage_metadata['stored_files']),
                'total_points': storage_metadata['total_points'],
                'storage_metadata': storage_metadata
            }
            
            # Stage 5: Storage optimization
            logger.info("Stage 5: Storage optimization")
            optimization_stats = self.storage.optimize_storage()
            results['stages']['optimization'] = {
                'status': 'completed',
                'optimized_files': optimization_stats['optimized_files'],
                'saved_space_mb': optimization_stats['saved_space_mb']
            }
            
            results['status'] = 'completed'
            results['end_time'] = datetime.now().isoformat()
            
            logger.info("NGSIM dataset processing completed successfully", 
                       dataset_name=dataset_name,
                       final_trajectories=len(dataset.trajectories))
            
        except Exception as e:
            logger.error("NGSIM dataset processing failed", error=str(e))
            results['status'] = 'failed'
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
        
        return results
    
    async def _load_ngsim_dataset(self, data_path: str):
        """Load NGSIM dataset using dataset factory.
        
        Args:
            data_path: Path to NGSIM data files
            
        Returns:
            Loaded TrajectoryDataset
        """
        # Create dataset using factory
        dataset = DatasetFactory.create_dataset(
            'ngsim', 
            self.dataset_config, 
            data_path=data_path
        )
        
        # Load the dataset
        return dataset.load()
    
    def load_processed_dataset(self, 
                             dataset_name: str,
                             version: Optional[str] = None,
                             filters: Optional[Dict[str, Any]] = None):
        """Load processed dataset from storage.
        
        Args:
            dataset_name: Name of the dataset
            version: Optional version identifier
            filters: Optional filters for loading specific data
            
        Returns:
            DataFrame with trajectory data
        """
        logger.info("Loading processed dataset", 
                   dataset_name=dataset_name,
                   version=version)
        
        return self.storage.load_trajectories(dataset_name, version, filters)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Storage statistics
        """
        return self.storage.get_storage_stats()
    
    def list_available_datasets(self) -> Dict[str, Any]:
        """List available datasets and their versions.
        
        Returns:
            Dictionary of available datasets and versions
        """
        # This would need to scan the storage directory
        # For now, return basic information
        return {
            'storage_path': str(self.storage_config.base_path),
            'available_datasets': DatasetFactory.list_datasets(),
            'storage_stats': self.get_storage_stats()
        }


async def run_sample_pipeline():
    """Run a sample ETL pipeline for demonstration."""
    
    # Configuration
    etl_config = ETLConfig(
        max_concurrent_requests=5,
        batch_size=500,
        enable_progress_bars=True
    )
    
    dataset_config = DatasetConfig(
        min_trajectory_length=5,
        max_trajectory_length=500,
        min_velocity=0.0,
        max_velocity=30.0
    )
    
    storage_config = StorageConfig(
        base_path="data/processed",
        partition_by=['vehicle_id', 'date'],
        compression="snappy"
    )
    
    quality_config = QualityConfig(
        min_completeness=0.7,
        enable_cleaning=True,
        outlier_detection=True,
        generate_reports=True
    )
    
    # Create pipeline
    pipeline = CompleteETLPipeline(
        etl_config=etl_config,
        dataset_config=dataset_config,
        storage_config=storage_config,
        quality_config=quality_config
    )
    
    # Run pipeline (assuming NGSIM data is available)
    try:
        results = await pipeline.process_ngsim_dataset(
            data_path="data/ngsim",
            dataset_name="ngsim_sample",
            version="v1.0",
            enable_quality_validation=True,
            enable_cleaning=True
        )
        
        print("Pipeline Results:")
        print(f"Status: {results['status']}")
        print(f"Dataset: {results['dataset_name']}")
        print(f"Version: {results['version']}")
        
        for stage, stage_results in results['stages'].items():
            print(f"\n{stage.upper()}:")
            for key, value in stage_results.items():
                if key != 'report':  # Skip detailed reports
                    print(f"  {key}: {value}")
        
        # Get storage stats
        storage_stats = pipeline.get_storage_stats()
        print(f"\nStorage Statistics:")
        print(f"  Total files: {storage_stats['total_data_files']}")
        print(f"  Total size: {storage_stats['total_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")


if __name__ == "__main__":
    # Run the sample pipeline
    asyncio.run(run_sample_pipeline())