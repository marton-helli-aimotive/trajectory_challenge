"""Tests for NGSIM data downloading functionality with JSON/CSV API."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from trajectory_prediction.data import (
    DownloadError,
    NGSIMDownloader,
    download_all_ngsim_datasets,
    download_ngsim_dataset,
)


class AsyncChunkIterator:
    """Async iterator for mocking chunked content."""

    def __init__(self, data):
        self.data = data
        self.called = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.called:
            raise StopAsyncIteration
        self.called = True
        return self.data


@pytest.fixture
def temp_download_dir():
    """Create temporary download directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_json_data():
    """Create mock JSON trajectory data."""
    return [
        {
            "vehicle_id": "1",
            "frame_id": "1",
            "global_time": "946684800000",
            "local_x": "10.0",
            "local_y": "5.0",
            "v_vel": "20.0",
            "v_acc": "0.0",
            "v_class": "2",
            "v_length": "4.5",
            "v_width": "1.8",
            "lane_id": "1",
            "location": "us-101",
        },
        {
            "vehicle_id": "1",
            "frame_id": "2",
            "global_time": "946684801000",
            "local_x": "12.0",
            "local_y": "5.0",
            "v_vel": "20.0",
            "v_acc": "0.0",
            "v_class": "2",
            "v_length": "4.5",
            "v_width": "1.8",
            "lane_id": "1",
            "location": "us-101",
        },
    ]


@pytest.fixture
def mock_csv_data():
    """Create mock CSV trajectory data."""
    return """vehicle_id,frame_id,global_time,local_x,local_y,v_vel,v_acc,v_class,v_length,v_width,lane_id,location
1,1,946684800000,10.0,5.0,20.0,0.0,2,4.5,1.8,1,us-101
1,2,946684801000,12.0,5.0,20.0,0.0,2,4.5,1.8,1,us-101
"""


class TestNGSIMDownloader:
    """Test NGSIM downloader functionality."""

    def test_init(self, temp_download_dir):
        """Test downloader initialization."""
        downloader = NGSIMDownloader(temp_download_dir)

        assert downloader.download_dir == temp_download_dir
        assert downloader.max_concurrent_downloads == 3
        assert downloader.chunk_size == 8192
        assert downloader.timeout == 300
        assert temp_download_dir.exists()

    def test_init_with_custom_params(self, temp_download_dir):
        """Test downloader initialization with custom parameters."""
        downloader = NGSIMDownloader(
            temp_download_dir,
            max_concurrent_downloads=5,
            chunk_size=16384,
            timeout=600,
        )

        assert downloader.max_concurrent_downloads == 5
        assert downloader.chunk_size == 16384
        assert downloader.timeout == 600

    def test_list_available_datasets(self, temp_download_dir):
        """Test listing available datasets."""
        downloader = NGSIMDownloader(temp_download_dir)
        datasets = downloader.list_available_datasets()

        assert isinstance(datasets, dict)
        assert "us101" in datasets
        assert "i80" in datasets
        assert "lankershim" in datasets
        assert "peachtree" in datasets

        for _dataset_name, dataset_info in datasets.items():
            assert "base_url" in dataset_info
            assert "location_filter" in dataset_info
            assert "description" in dataset_info
            assert "format" in dataset_info

    @pytest.mark.asyncio
    async def test_get_dataset_info(self, temp_download_dir):
        """Test getting dataset information."""
        downloader = NGSIMDownloader(temp_download_dir)

        info = await downloader.get_dataset_info("us101")

        assert info["description"] == "US-101 Los Angeles, CA trajectory data"
        assert info["location_filter"] == "us-101"
        assert info["downloaded"] is False
        assert "local_files" in info
        assert info["local_files"] == {}

    @pytest.mark.asyncio
    async def test_get_dataset_info_with_existing_files(self, temp_download_dir):
        """Test getting dataset info when files exist."""
        downloader = NGSIMDownloader(temp_download_dir)

        # Create mock downloaded files
        json_file = temp_download_dir / "us101.json"
        csv_file = temp_download_dir / "us101.csv"
        json_file.write_text('{"test": "data"}')
        csv_file.write_text("test,data\n1,2")

        info = await downloader.get_dataset_info("us101")

        assert info["downloaded"] is True
        assert "json" in info["local_files"]
        assert "csv" in info["local_files"]
        assert info["local_files"]["json"]["path"] == str(json_file)
        assert info["local_files"]["csv"]["path"] == str(csv_file)
        assert info["local_files"]["json"]["size"] > 0
        assert info["local_files"]["csv"]["size"] > 0

    @pytest.mark.asyncio
    async def test_get_dataset_info_invalid_dataset(self, temp_download_dir):
        """Test getting info for invalid dataset."""
        downloader = NGSIMDownloader(temp_download_dir)

        with pytest.raises(ValueError, match="Unknown dataset"):
            await downloader.get_dataset_info("invalid_dataset")

    @pytest.mark.asyncio
    async def test_download_dataset_json_success(
        self, temp_download_dir, mock_json_data
    ):
        """Test successful JSON dataset download."""
        downloader = NGSIMDownloader(temp_download_dir)

        # Mock the HTTP response
        with patch("aiohttp.ClientSession.get") as mock_get:
            # Mock response context manager
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {"content-length": "1000"}

            # Create a mock content object with iter_chunked method that returns the async iterator directly
            mock_content = AsyncMock()
            # Make iter_chunked a simple method that returns the async iterator, not an AsyncMock
            mock_content.iter_chunked = lambda chunk_size: AsyncChunkIterator(
                json.dumps(mock_json_data).encode()
            )
            mock_response.content = mock_content

            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            # Download dataset
            result_path = await downloader.download_dataset(
                "us101", output_format="json"
            )

            # Verify result
            assert result_path.exists()
            assert result_path.name == "us101.json"
            assert result_path.parent == temp_download_dir

    @pytest.mark.asyncio
    async def test_download_dataset_csv_success(self, temp_download_dir, mock_csv_data):
        """Test successful CSV dataset download."""
        downloader = NGSIMDownloader(temp_download_dir)

        # Mock the HTTP response
        with patch("aiohttp.ClientSession.get") as mock_get:
            # Mock response context manager
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {"content-length": "500"}

            # Create a mock content object with iter_chunked method that returns the async iterator directly
            mock_content = AsyncMock()
            # Make iter_chunked a simple method that returns the async iterator, not an AsyncMock
            mock_content.iter_chunked = lambda chunk_size: AsyncChunkIterator(
                mock_csv_data.encode()
            )
            mock_response.content = mock_content

            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            # Download dataset
            result_path = await downloader.download_dataset(
                "us101", output_format="csv"
            )

            # Verify result
            assert result_path.exists()
            assert result_path.name == "us101.csv"
            assert result_path.parent == temp_download_dir

    @pytest.mark.asyncio
    async def test_download_dataset_with_limit(self, temp_download_dir, mock_json_data):
        """Test dataset download with record limit."""
        downloader = NGSIMDownloader(temp_download_dir)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {"content-length": "500"}

            # Create a mock content object with iter_chunked method that returns the async iterator directly
            mock_content = AsyncMock()
            # Make iter_chunked a simple method that returns the async iterator, not an AsyncMock
            mock_content.iter_chunked = lambda chunk_size: AsyncChunkIterator(
                json.dumps(mock_json_data[:1]).encode()
            )
            mock_response.content = mock_content

            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            # Download with limit
            await downloader.download_dataset("us101", limit=1)

            # Verify the API was called with correct parameters
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "$limit" in call_args[1]["params"]
            assert call_args[1]["params"]["$limit"] == "1"

    @pytest.mark.asyncio
    async def test_download_dataset_already_exists(self, temp_download_dir):
        """Test download when file already exists."""
        downloader = NGSIMDownloader(temp_download_dir)

        # Create existing file
        existing_file = temp_download_dir / "us101.json"
        existing_file.write_text('{"existing": "data"}')

        # Download should return existing file without calling API
        with patch("aiohttp.ClientSession.get") as mock_get:
            result_path = await downloader.download_dataset("us101")

            # Should not call the API
            mock_get.assert_not_called()
            assert result_path == existing_file

    @pytest.mark.asyncio
    async def test_download_dataset_force_redownload(
        self, temp_download_dir, mock_json_data
    ):
        """Test force re-download of existing file."""
        downloader = NGSIMDownloader(temp_download_dir)

        # Create existing file
        existing_file = temp_download_dir / "us101.json"
        existing_file.write_text('{"old": "data"}')

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {"content-length": "1000"}

            # Create a mock content object with iter_chunked method that returns the async iterator directly
            mock_content = AsyncMock()
            # Make iter_chunked a simple method that returns the async iterator, not an AsyncMock
            mock_content.iter_chunked = lambda chunk_size: AsyncChunkIterator(
                json.dumps(mock_json_data).encode()
            )
            mock_response.content = mock_content

            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            # Force download
            result_path = await downloader.download_dataset(
                "us101", force_download=True
            )

            # Should call the API even though file exists
            mock_get.assert_called_once()
            assert result_path == existing_file

    @pytest.mark.asyncio
    async def test_download_dataset_invalid_dataset(self, temp_download_dir):
        """Test download with invalid dataset name."""
        downloader = NGSIMDownloader(temp_download_dir)

        with pytest.raises(ValueError, match="Unknown dataset"):
            await downloader.download_dataset("invalid_dataset")

    @pytest.mark.asyncio
    async def test_download_dataset_network_error(self, temp_download_dir):
        """Test download with network error."""
        downloader = NGSIMDownloader(temp_download_dir)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.side_effect = aiohttp.ClientError("Network error")

            with pytest.raises(DownloadError, match="Network error"):
                await downloader.download_dataset("us101")

    @pytest.mark.asyncio
    async def test_download_all_datasets(self, temp_download_dir, mock_json_data):
        """Test downloading all datasets."""
        downloader = NGSIMDownloader(temp_download_dir)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {"content-length": "1000"}

            # Create a mock content object with iter_chunked method that returns the async iterator directly
            mock_content = AsyncMock()
            # Make iter_chunked a simple method that returns the async iterator, not an AsyncMock
            mock_content.iter_chunked = lambda chunk_size: AsyncChunkIterator(
                json.dumps(mock_json_data).encode()
            )
            mock_response.content = mock_content

            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            # Download all datasets
            results = await downloader.download_all_datasets()

            # Verify all datasets were downloaded
            assert len(results) == 4  # us101, i80, lankershim, peachtree
            assert "us101" in results
            assert "i80" in results
            assert "lankershim" in results
            assert "peachtree" in results

            # Check files exist
            for dataset_name, path in results.items():
                assert path.exists()
                assert path.name == f"{dataset_name}.json"


class TestConvenienceFunctions:
    """Test convenience functions for backward compatibility."""

    @pytest.mark.asyncio
    async def test_download_ngsim_dataset(self, temp_download_dir):
        """Test convenience function for downloading single dataset."""
        mock_path = temp_download_dir / "us101.json"

        with patch(
            "trajectory_prediction.data.downloaders.NGSIMDownloader"
        ) as mock_downloader_class:
            mock_downloader = AsyncMock()
            mock_downloader_class.return_value = mock_downloader
            mock_downloader.download_dataset.return_value = mock_path

            result = await download_ngsim_dataset("us101", temp_download_dir, limit=100)

            assert result == mock_path
            mock_downloader_class.assert_called_once_with(temp_download_dir)
            mock_downloader.download_dataset.assert_called_once_with(
                "us101", False, 100, 0, "json"
            )

    @pytest.mark.asyncio
    async def test_download_all_ngsim_datasets(self, temp_download_dir):
        """Test convenience function for downloading all datasets."""
        mock_results = {"us101": temp_download_dir / "us101.json"}

        with patch(
            "trajectory_prediction.data.downloaders.NGSIMDownloader"
        ) as mock_downloader_class:
            mock_downloader = AsyncMock()
            mock_downloader_class.return_value = mock_downloader
            mock_downloader.download_all_datasets.return_value = mock_results

            results = await download_all_ngsim_datasets(
                temp_download_dir, output_format="csv"
            )

            assert results == mock_results
            mock_downloader_class.assert_called_once_with(temp_download_dir)
            mock_downloader.download_all_datasets.assert_called_once_with(
                False, None, "csv"
            )


@pytest.mark.integration
class TestRealDownload:
    """Integration tests with real network calls."""

    @pytest.fixture
    def cache_download_dir(self):
        """Create persistent cache directory for integration tests."""
        cache_dir = Path("test_data_cache")
        cache_dir.mkdir(exist_ok=True)
        return cache_dir

    @pytest.mark.asyncio
    async def test_real_ngsim_download(self, cache_download_dir):
        """Test actual download from NGSIM data portal."""
        downloader = NGSIMDownloader(cache_download_dir, timeout=600)

        # Download a small sample with limit
        result_path = await downloader.download_dataset(
            "us101", limit=10, output_format="json"
        )

        # Verify download
        assert result_path.exists()
        assert result_path.name == "us101.json"

        # Basic validation of JSON structure
        with result_path.open() as f:
            data = json.load(f)
            assert isinstance(data, list)
            if data:  # If we got data
                assert "vehicle_id" in data[0]
                assert "location" in data[0]
                assert data[0]["location"] == "us-101"

        # Test CSV download as well
        csv_result = await downloader.download_dataset(
            "us101", limit=5, output_format="csv"
        )
        assert csv_result.exists()
        assert csv_result.name == "us101.csv"

        # Verify CSV content
        csv_content = csv_result.read_text()
        assert "vehicle_id" in csv_content
        assert "location" in csv_content
