"""NGSIM data access and caching.

Scope: this module is responsible only for obtaining the raw dataset and returning a
single pandas DataFrame with canonical column names. No additional transformations or
analysis happen here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import io
import os
import zipfile
import logging
import json
import time
from urllib.parse import urlencode

import pandas as pd
import requests
from .utils import DatasetName, canonicalize_columns
from .trajectory import filter_by_vehicle as _filter_by_vehicle
from .trajectory import filter_by_frame_range as _filter_by_frame_range
from .trajectory import filter_by_lane as _filter_by_lane

LOGGER = logging.getLogger(__name__)

# Socrata (US DOT ITS Public Data Hub) dataset id for the merged NGSIM trajectories dataset.
NGSIM_PORTAL_DATASET_ID = "8ect-6jqj"

# Known dataset URL registry (raw trajectory files). These are representative; users may need
# to override if hosting changes.
# Each entry maps to a list of candidate URLs (tried in order until one succeeds).
_DATASET_URLS: Mapping[DatasetName, list[str]] = {
    "us_101": [
        "https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm",  # landing page (not direct)
    ],
    "i_80": [
        "https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm",
    ],
    "lankershim": [
        "https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm",
    ],
}

# Column name canonicalization map based on dataset schema
# Maps various input column names to standardized names from dataset instructions
_CANONICAL_COLUMNS = None  # kept for backward-compat import paths


@dataclass(frozen=True)
class NGSIMLoadResult:
    """Holds a loaded trajectory DataFrame plus metadata."""

    dataset: DatasetName
    dataframe: pd.DataFrame
    source: str  # URL or local path
    cached: bool


class NGSIMDownloader:
    """Handles retrieval and caching of NGSIM datasets.

    Parameters:
        cache_dir: Directory where downloaded files will be stored. Created if needed.
        timeout: HTTP timeout (seconds) for each URL attempt.
    """

    def __init__(self, cache_dir: str | Path = ".ngsim_cache", timeout: float = 30.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def _candidate_filenames(self, dataset: DatasetName) -> list[str]:
        # Many NGSIM distributions: zipped or plain CSV; we accept either naming pattern.
        return [
            f"{dataset}.csv",
            f"{dataset}.zip",
        ]

    def _find_cached(self, dataset: DatasetName) -> Path | None:
        for name in self._candidate_filenames(dataset):
            path = self.cache_dir / name
            if path.exists():
                return path
        return None

    def download(self, dataset: DatasetName, force: bool = False) -> Path:
        """Download dataset into cache if not present.

        Currently placeholder: real direct file URLs should replace landing pages.
        If only a landing page is available, user must supply local file. For now we
        raise to signal missing direct link.
        """
        cached = self._find_cached(dataset)
        if cached and not force:
            LOGGER.info("Using cached file %s", cached)
            return cached

        urls = _DATASET_URLS[dataset]
        last_exc: Exception | None = None
        for url in urls:
            try:
                LOGGER.info("Attempting download: %s", url)
                resp = requests.get(url, timeout=self.timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"Unexpected status {resp.status_code} for {url}")
                # Heuristic: if response is not CSV or zip, treat as failure (likely HTML landing page).
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    raise RuntimeError("Received HTML page instead of dataset file; supply local path.")
                # Decide filename by content-type or fallback.
                filename = self.cache_dir / (f"{dataset}.csv" if "csv" in content_type else f"{dataset}.bin")
                filename.write_bytes(resp.content)
                LOGGER.info("Saved %s (%d bytes)", filename, len(resp.content))
                return filename
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Download failed for %s: %s", url, exc)
                last_exc = exc
        raise RuntimeError(f"All download attempts failed for {dataset}: {last_exc}")

    def load(self, dataset: DatasetName, local_path: str | Path | None = None, force_download: bool = False) -> NGSIMLoadResult:
        """Load a dataset as a DataFrame.

        If local_path is provided, it's used directly (and may be a zip or csv). Otherwise
        we attempt to use a cached file or download.
        """
        if local_path:
            path = Path(local_path)
            source = str(path)
            cached = False
        else:
            path = self.download(dataset, force=force_download)
            source = str(path)
            cached = True

        df = _read_dataset_file(path)
        df = canonicalize_columns(df)
        return NGSIMLoadResult(dataset=dataset, dataframe=df, source=source, cached=cached)


def _read_dataset_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            # Pick first CSV inside zip.
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    LOGGER.debug("Reading %s from archive %s", name, path)
                    with zf.open(name) as f:  # type: ignore[arg-type]
                        data = f.read()
                        return pd.read_csv(io.BytesIO(data))
            raise ValueError(f"No CSV file found inside zip archive {path}")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:  # backwards compatibility for tests
    return canonicalize_columns(df)


def load_ngsim_portal(limit: int | None = 1000, select: Sequence[str] | None = None, where: str | None = None, app_token: str | None = None, session: requests.Session | None = None, json_endpoint: bool = True) -> pd.DataFrame:
    """Load trajectory records directly from the official US DOT ITS data portal (Socrata).

    This uses the public dataset: https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj

    Parameters:
        limit: Maximum number of rows to return. None => portal default (may be capped).
        select: Optional list of column names to restrict returned columns (Socrata $select).
        where: Optional Socrata $where expression for server-side filtering.
        app_token: Optional Socrata app token. If omitted, env var SOCRATA_APP_TOKEN is used if set.
        session: Optional pre-configured requests.Session.

    Returns:
        pandas.DataFrame of the requested slice.

    Notes:
        - This function pulls data over the network; for repeated large queries consider local caching.
        - For very large extracts, use the bulk export interface outside this helper.
    """
    # Use JSON endpoint by default (user requested): returns list of JSON objects.
    extension = "json" if json_endpoint else "csv"
    base_url = f"https://data.transportation.gov/resource/{NGSIM_PORTAL_DATASET_ID}.{extension}"
    params: dict[str, str] = {}
    if limit is not None:
        params["$limit"] = str(limit)
    if select:
        params["$select"] = ",".join(select)
    if where:
        params["$where"] = where

    headers: dict[str, str] = {"User-Agent": "trajectory-challenge-ngsim-loader/0.1"}
    token = app_token or os.getenv("SOCRATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token

    query_url = base_url
    if params:
        query_url += "?" + urlencode(params)
    sess = session or requests.Session()
    LOGGER.info("Fetching NGSIM portal data: %s", query_url)
    try:
        resp = sess.get(query_url, headers=headers, timeout=60)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to fetch NGSIM portal data: {exc}") from exc

    if json_endpoint:
        # Socrata JSON returns list[dict]; build DataFrame directly.
        try:
            records = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Invalid JSON returned from portal: {exc}") from exc
        if not isinstance(records, list):
            raise RuntimeError("Unexpected JSON structure (expected list of records)")
        df = pd.DataFrame.from_records(records)
    else:
        df = pd.read_csv(io.BytesIO(resp.content))
    df = canonicalize_columns(df)
    return df


def load_ngsim_cached(cache_dir: str | Path = ".ngsim_cache", force_refresh: bool = False, chunk_size: int = 50000) -> pd.DataFrame:
    """Load the complete NGSIM dataset with local caching.
    
    This function will:
    1. Check for a cached complete dataset file
    2. If not found or force_refresh=True, download the complete dataset in chunks
    3. Save the combined data to cache for future use
    4. Return the complete dataset as a pandas DataFrame
    
    Parameters:
        cache_dir: Directory to store cached data
        force_refresh: If True, download fresh data even if cache exists
        chunk_size: Number of records to fetch per API request
        
    Returns:
        pandas.DataFrame containing the complete NGSIM dataset
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_path / "ngsim_complete_dataset.parquet"
    cache_metadata_file = cache_path / "ngsim_cache_metadata.json"
    
    # Check if cached data exists and is recent
    if cache_file.exists() and not force_refresh:
        print("🔍 Checking for existing cache...")
        if cache_metadata_file.exists():
            with open(cache_metadata_file, 'r') as f:
                metadata = json.load(f)
                cache_age_days = (time.time() - metadata.get('timestamp', 0)) / (24 * 3600)
                if cache_age_days < 30:  # Use cache if less than 30 days old
                    LOGGER.info(f"Using cached dataset from {cache_file} (age: {cache_age_days:.1f} days)")
                    print(f"✅ Found valid cache (age: {cache_age_days:.1f} days)")
                    print(f"📁 Loading from: {cache_file}")
                    print(f"📊 Cached records: {metadata.get('total_records', 'Unknown'):,}")
                    if metadata.get('unique_vehicles'):
                        print(f"🚗 Unique vehicles: {metadata['unique_vehicles']:,}")
                    print("⚡ Loading cached data...")
                    df = pd.read_parquet(cache_file)
                    print("✅ Cache loaded successfully!")
                    return _canonicalize_columns(df)
                else:
                    print(f"⚠️  Cache is stale (age: {cache_age_days:.1f} days), downloading fresh data...")
        else:
            print("⚠️  Cache file exists but metadata is missing, downloading fresh data...")
    
    LOGGER.info("Downloading complete NGSIM dataset...")
    print("🔄 Starting download of complete NGSIM dataset...")
    print("📊 This may take several minutes depending on dataset size and connection speed")
    
    # Download complete dataset in chunks
    all_data = []
    offset = 0
    total_fetched = 0
    
    while True:
        LOGGER.info(f"Fetching chunk starting at offset {offset}...")
        print(f"📥 Fetching chunk {len(all_data) + 1} (offset: {offset:,})...")
        
        # Build query with offset and limit
        params = {
            "$limit": str(chunk_size),
            "$offset": str(offset),
            "location": "us-101",
            "$order": "vehicle_id,frame_id"  # Ensure consistent ordering
        }
        
        chunk_url = f"https://data.transportation.gov/resource/{NGSIM_PORTAL_DATASET_ID}.json?" + urlencode(params)
        
        headers = {"User-Agent": "trajectory-challenge-ngsim-loader/0.1"}
        token = os.getenv("SOCRATA_APP_TOKEN")
        if token:
            headers["X-App-Token"] = token
            
        try:
            resp = requests.get(chunk_url, headers=headers, timeout=120)
            resp.raise_for_status()
            records = resp.json()
            
            if not records:  # Empty response means we've reached the end
                LOGGER.info("Reached end of dataset")
                print("✅ Reached end of dataset - download complete!")
                break
                
            chunk_df = pd.DataFrame.from_records(records)
            all_data.append(chunk_df)
            
            chunk_size_actual = len(records)
            total_fetched += chunk_size_actual
            LOGGER.info(f"Fetched {chunk_size_actual} records (total: {total_fetched})")
            print(f"📈 Chunk {len(all_data)}: {chunk_size_actual:,} records (total: {total_fetched:,})")
            
            # If we got fewer records than requested, we've reached the end
            if chunk_size_actual < chunk_size:
                LOGGER.info(f"Got {chunk_size_actual} < {chunk_size} records, assuming end of dataset")
                print(f"✅ Final chunk received ({chunk_size_actual:,} < {chunk_size:,} records) - download complete!")
                break
                
            offset += chunk_size
            
            if total_fetched >= 1000000:  # Arbitrary limit to avoid too large downloads
                break
            # Add a small delay to be respectful to the API
            # time.sleep(0.5)
            
        except Exception as exc:
            LOGGER.error(f"Failed to fetch chunk at offset {offset}: {exc}")
            print(f"❌ Error fetching chunk at offset {offset:,}: {exc}")
            if not all_data:  # If we haven't fetched any data yet, re-raise
                raise
            else:  # If we have some data, break and use what we have
                LOGGER.warning("Stopping download due to error, using partial dataset")
                print("⚠️  Stopping download due to error, using partial dataset")
                break
    
    if not all_data:
        print("❌ No data was successfully downloaded")
        raise RuntimeError("No data was successfully downloaded")
    
    # Combine all chunks
    LOGGER.info(f"Combining {len(all_data)} chunks with total {total_fetched} records...")
    print(f"🔧 Processing data: combining {len(all_data)} chunks...")
    complete_df = pd.concat(all_data, ignore_index=True)
    
    # # Remove duplicates that might occur due to API pagination overlap
    # if 'vehicle_id' in complete_df.columns and 'frame_id' in complete_df.columns:
    #     initial_size = len(complete_df)
    #     print(f"🧹 Removing duplicates from {initial_size:,} records...")
    #     complete_df = complete_df.drop_duplicates(subset=['vehicle_id', 'frame_id'])
    #     final_size = len(complete_df)
    #     if initial_size != final_size:
    #         LOGGER.info(f"Removed {initial_size - final_size} duplicate records")
    #         print(f"✅ Removed {initial_size - final_size:,} duplicates, {final_size:,} unique records remain")
    #     else:
    #         print(f"✅ No duplicates found, {final_size:,} records")
    
    # Canonicalize column names
    print("🏷️  Standardizing column names...")
    complete_df = canonicalize_columns(complete_df)
    
    # Save to cache
    print(f"💾 Saving dataset to cache ({len(complete_df):,} records)...")
    try:
        complete_df.to_parquet(cache_file, index=False)
        
        # Save metadata
        metadata = {
            'timestamp': time.time(),
            'total_records': len(complete_df),
            'columns': list(complete_df.columns),
            'unique_vehicles': complete_df['vehicle_id'].nunique() if 'vehicle_id' in complete_df.columns else None
        }
        with open(cache_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        LOGGER.info(f"Saved complete dataset to cache: {cache_file} ({len(complete_df)} records)")
        print(f"✅ Cache saved successfully!")
        print(f"📁 Cache location: {cache_file}")
        print(f"📊 Total records: {len(complete_df):,}")
        if 'vehicle_id' in complete_df.columns:
            print(f"🚗 Unique vehicles: {complete_df['vehicle_id'].nunique():,}")
        print(f"📅 Cache created: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as exc:
        LOGGER.warning(f"Failed to save cache: {exc}")
        print(f"⚠️  Warning: Failed to save cache: {exc}")
        print("📝 Data will still be returned but won't be cached for future use")
    
    print("🎉 Dataset download and processing complete!")
    return complete_df


# Backwards-compatible re-exports of common filters so existing imports keep working.
def filter_by_vehicle(df: pd.DataFrame, vehicle_ids: int | Sequence[int]) -> pd.DataFrame:
    return _filter_by_vehicle(df, vehicle_ids)


def filter_by_frame_range(df: pd.DataFrame, start: int | None = None, end: int | None = None) -> pd.DataFrame:
    return _filter_by_frame_range(df, start=start, end=end)


def filter_by_lane(df: pd.DataFrame, lanes: int | Sequence[int]) -> pd.DataFrame:
    return _filter_by_lane(df, lanes)


__all__ = [
    "NGSIMDownloader",
    "NGSIMLoadResult",
    "NGSIM_PORTAL_DATASET_ID",
    "load_ngsim_portal",
    "load_ngsim_cached",
    "filter_by_vehicle",
    "filter_by_frame_range",
    "filter_by_lane",
]
