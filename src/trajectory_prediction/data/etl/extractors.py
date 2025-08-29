"""Data extraction components for ETL pipeline."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
import pandas as pd
from omegaconf import DictConfig

from ..sources.base import DataSource
from ..validation.schemas import TrajectoryData

logger = logging.getLogger(__name__)


class DataExtractor:
    """Enhanced data extractor with validation and async processing."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.batch_size = config.get("batch_size", 1000)
        self.max_concurrent = config.get("max_concurrent", 10)
        self.validate_data = config.get("validate_extraction", True)
        
    async def extract(self, source: DataSource) -> List[Dict[str, Any]]:
        """Extract and validate data from a data source."""
        logger.info(f"Extracting data from {source.name}")
        
        # Extract raw data
        raw_data = await source.load_data()
        logger.info(f"Extracted {len(raw_data)} raw records")
        
        if not raw_data:
            logger.warning("No data extracted from source")
            return []
        
        # Optional validation
        if self.validate_data:
            validated_data = await self._validate_extracted_data(raw_data)
            logger.info(f"Validated {len(validated_data)} records")
            return validated_data
        
        return raw_data
    
    async def extract_incremental(
        self, 
        source: DataSource, 
        last_update: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract incremental data updates."""
        logger.info(f"Extracting incremental data from {source.name} since {last_update}")
        
        raw_data = await source.get_incremental_data(last_update)
        logger.info(f"Extracted {len(raw_data)} incremental records")
        
        if self.validate_data and raw_data:
            validated_data = await self._validate_extracted_data(raw_data)
            return validated_data
        
        return raw_data
    
    async def extract_to_file(
        self, 
        source: DataSource, 
        output_path: Union[str, Path],
        file_format: str = "parquet"
    ) -> Dict[str, Any]:
        """Extract data and save directly to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        data = await self.extract(source)
        
        if not data:
            logger.warning("No data to write to file")
            return {"records_written": 0, "file_path": str(output_path)}
        
        # Convert to DataFrame for efficient writing
        df = pd.DataFrame(data)
        
        # Write based on format
        if file_format.lower() == "parquet":
            df.to_parquet(output_path, index=False, engine="pyarrow")
        elif file_format.lower() == "csv":
            df.to_csv(output_path, index=False)
        elif file_format.lower() == "json":
            df.to_json(output_path, orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Wrote {len(data)} records to {output_path}")
        
        return {
            "records_written": len(data),
            "file_path": str(output_path),
            "file_size_bytes": output_path.stat().st_size
        }
    
    async def _validate_extracted_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate extracted data using Pydantic schemas."""
        validated_records = []
        invalid_count = 0
        
        async def validate_batch(batch: List[Dict[str, Any]]) -> tuple:
            batch_valid = []
            batch_invalid = 0
            
            for record in batch:
                try:
                    # Validate using TrajectoryData schema
                    validated_record = TrajectoryData(**record)
                    batch_valid.append(validated_record.dict())
                except Exception as e:
                    logger.debug(f"Invalid record skipped: {e}")
                    batch_invalid += 1
            
            return batch_valid, batch_invalid
        
        # Process in batches for better performance
        batch_size = 1000
        tasks = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            tasks.append(validate_batch(batch))
        
        # Process batches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch validation failed: {result}")
                continue
            
            batch_valid, batch_invalid = result
            validated_records.extend(batch_valid)
            invalid_count += batch_invalid
        
        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid records during validation")
        
        return validated_records


class ParallelDataExtractor(DataExtractor):
    """High-performance parallel data extractor."""
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.max_workers = config.get("max_workers", 4)
        self.chunk_size = config.get("chunk_size", 10000)
    
    async def extract_parallel(
        self, 
        sources: List[DataSource]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract data from multiple sources in parallel."""
        logger.info(f"Extracting data from {len(sources)} sources in parallel")
        
        # Create extraction tasks
        tasks = []
        for source in sources:
            task = asyncio.create_task(self.extract(source))
            tasks.append((source.name, task))
        
        # Execute tasks and collect results
        results = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in tasks], 
            return_exceptions=True
        )
        
        for (source_name, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Failed to extract from {source_name}: {result}")
                results[source_name] = []
            else:
                results[source_name] = result
                logger.info(f"Extracted {len(result)} records from {source_name}")
        
        return results