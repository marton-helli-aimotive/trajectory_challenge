"""Async ETL pipeline for vehicle trajectory data processing."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    try:
        import pandas as pd
        DataFrame = pd.DataFrame
    except ImportError:
        DataFrame = Any
else:
    DataFrame = Any
from dataclasses import dataclass
from datetime import datetime, timedelta

import aiohttp
import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

# Use a simple base class since BaseConfig doesn't exist
class BaseConfig:
    """Base configuration class."""
    pass
from ..core.exceptions import ETLPipelineError, DataSourceError
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ETLConfig(BaseConfig):
    """Configuration for ETL pipeline."""
    
    # Async processing settings
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Batch processing
    batch_size: int = 1000
    max_batch_concurrency: int = 5
    
    # Data processing
    chunk_size: int = 8192
    temp_dir: str = "/tmp/etl"
    
    # Monitoring
    enable_progress_bars: bool = True
    log_batch_progress: bool = True
    
    class Config:
        env_prefix = "ETL_"


class AsyncETLPipeline:
    """Async ETL pipeline for vehicle trajectory data processing."""
    
    def __init__(self, config: ETLConfig):
        """Initialize ETL pipeline with configuration.
        
        Args:
            config: ETL configuration settings
        """
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized async ETL pipeline", 
                   config=config.dict(),
                   temp_dir=str(self.temp_dir))
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def download_file(self, url: str, local_path: Path) -> bool:
        """Download a file from URL with retry logic.
        
        Args:
            url: Source URL
            local_path: Local file path to save to
            
        Returns:
            True if download successful, False otherwise
        """
        async with self.semaphore:
            for attempt in range(self.config.retry_attempts):
                try:
                    async with self.session.get(url) as response:
                        response.raise_for_status()
                        
                        async with aiofiles.open(local_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(
                                self.config.chunk_size
                            ):
                                await f.write(chunk)
                    
                    logger.debug("Downloaded file successfully", 
                               url=url, 
                               local_path=str(local_path),
                               size=local_path.stat().st_size)
                    return True
                    
                except Exception as e:
                    logger.warning("Download attempt failed", 
                                 url=url, 
                                 attempt=attempt + 1,
                                 error=str(e))
                    
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    else:
                        logger.error("All download attempts failed", 
                                   url=url, 
                                   error=str(e))
                        return False
        
        return False
    
    async def download_files(self, urls: List[str], output_dir: Path) -> List[Path]:
        """Download multiple files concurrently.
        
        Args:
            urls: List of URLs to download
            output_dir: Directory to save files
            
        Returns:
            List of successfully downloaded file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        async def download_single(url: str) -> Optional[Path]:
            filename = url.split('/')[-1]
            local_path = output_dir / filename
            
            if await self.download_file(url, local_path):
                return local_path
            return None
        
        tasks = [download_single(url) for url in urls]
        
        if self.config.enable_progress_bars:
            results = await tqdm_asyncio.gather(
                *tasks, 
                desc="Downloading files",
                total=len(tasks)
            )
        else:
            results = await asyncio.gather(*tasks)
        
        downloaded_files = [f for f in results if f is not None]
        
        logger.info("Download completed", 
                   total_urls=len(urls),
                   successful_downloads=len(downloaded_files))
        
        return downloaded_files
    
    async def process_file_batch(self, file_paths: List[Path]) -> List[DataFrame]:
        """Process a batch of files concurrently.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of processed DataFrames
        """
        async def process_single(file_path: Path) -> DataFrame:
            """Process a single file."""
            try:
                # Determine file type and process accordingly
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                elif file_path.suffix.lower() in ['.parquet', '.pq']:
                    df = pd.read_parquet(file_path)
                elif file_path.suffix.lower() in ['.json', '.jsonl']:
                    df = pd.read_json(file_path, lines=True)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
                logger.debug("Processed file", 
                           file_path=str(file_path),
                           rows=len(df),
                           columns=list(df.columns))
                
                return df
                
            except Exception as e:
                logger.error("Failed to process file", 
                           file_path=str(file_path),
                           error=str(e))
                raise
        
        # Process files with limited concurrency
        semaphore = asyncio.Semaphore(self.config.max_batch_concurrency)
        
        async def process_with_semaphore(file_path: Path) -> DataFrame:
            async with semaphore:
                return await asyncio.to_thread(process_single, file_path)
        
        tasks = [process_with_semaphore(fp) for fp in file_paths]
        
        if self.config.enable_progress_bars:
            results = await tqdm_asyncio.gather(
                *tasks,
                desc="Processing files",
                total=len(tasks)
            )
        else:
            results = await asyncio.gather(*tasks)
        
        return results
    
    async def process_files(self, file_paths: List[Path]) -> List[DataFrame]:
        """Process files in batches.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of processed DataFrames
        """
        all_dataframes = []
        
        # Process in batches
        for i in range(0, len(file_paths), self.config.batch_size):
            batch = file_paths[i:i + self.config.batch_size]
            
            if self.config.log_batch_progress:
                logger.info("Processing batch", 
                           batch_num=i // self.config.batch_size + 1,
                           batch_size=len(batch),
                           total_files=len(file_paths))
            
            batch_results = await self.process_file_batch(batch)
            all_dataframes.extend(batch_results)
        
        logger.info("File processing completed", 
                   total_files=len(file_paths),
                   total_dataframes=len(all_dataframes))
        
        return all_dataframes
    
    async def incremental_load(self, 
                             source_paths: List[Path],
                             last_processed_time: Optional[datetime] = None) -> List[Path]:
        """Perform incremental loading based on file modification times.
        
        Args:
            source_paths: List of source file paths
            last_processed_time: Last processing timestamp
            
        Returns:
            List of files that need processing
        """
        if last_processed_time is None:
            return source_paths
        
        files_to_process = []
        
        for file_path in source_paths:
            if not file_path.exists():
                logger.warning("Source file does not exist", 
                             file_path=str(file_path))
                continue
            
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            if mtime > last_processed_time:
                files_to_process.append(file_path)
                logger.debug("File marked for processing", 
                           file_path=str(file_path),
                           modification_time=mtime,
                           last_processed=last_processed_time)
        
        logger.info("Incremental load analysis", 
                   total_files=len(source_paths),
                   files_to_process=len(files_to_process),
                   last_processed_time=last_processed_time)
        
        return files_to_process
    
    async def run_pipeline(self, 
                          source_urls: Optional[List[str]] = None,
                          source_paths: Optional[List[Path]] = None,
                          output_dir: Path = Path("data/processed"),
                          incremental: bool = False,
                                                     last_processed_time: Optional[datetime] = None) -> List[DataFrame]:
        """Run the complete ETL pipeline.
        
        Args:
            source_urls: URLs to download (optional)
            source_paths: Local file paths to process (optional)
            output_dir: Output directory for processed data
            incremental: Whether to perform incremental loading
            last_processed_time: Last processing timestamp for incremental loading
            
        Returns:
            List of processed DataFrames
            
        Raises:
            ETLPipelineError: If pipeline execution fails
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Download files if URLs provided
            if source_urls:
                logger.info("Starting file downloads", 
                           num_urls=len(source_urls))
                downloaded_files = await self.download_files(source_urls, self.temp_dir)
                source_paths = downloaded_files
            
            if not source_paths:
                raise ETLPipelineError("No source files provided")
            
            # Incremental loading
            if incremental:
                source_paths = await self.incremental_load(source_paths, last_processed_time)
                
                if not source_paths:
                    logger.info("No new files to process")
                    return []
            
            # Process files
            logger.info("Starting file processing", 
                       num_files=len(source_paths))
            dataframes = await self.process_files(source_paths)
            
            # Save processed data
            if dataframes:
                output_file = output_dir / f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                
                # Combine all dataframes
                combined_df = pd.concat(dataframes, ignore_index=True)
                combined_df.to_parquet(output_file, index=False)
                
                logger.info("Pipeline completed successfully", 
                           output_file=str(output_file),
                           total_rows=len(combined_df),
                           total_columns=len(combined_df.columns))
            
            return dataframes
            
        except Exception as e:
            logger.error("ETL pipeline failed", error=str(e))
            raise ETLPipelineError(f"Pipeline execution failed: {str(e)}") from e
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "temp_dir": str(self.temp_dir),
            "max_concurrent_requests": self.config.max_concurrent_requests,
            "batch_size": self.config.batch_size,
            "retry_attempts": self.config.retry_attempts,
        }