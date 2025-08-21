#!/usr/bin/env python3
"""
Demo script for the refactored NGSIM downloader.

This script demonstrates how to use the new API-based downloader to fetch
NGSIM trajectory data from the Socrata API in JSON or CSV format.
"""

import asyncio
import logging
from pathlib import Path

from trajectory_prediction.data.downloaders import NGSIMDownloader


async def main():
    """Main demo function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create download directory
    download_dir = Path("./demo_downloads")
    download_dir.mkdir(exist_ok=True)

    # Initialize downloader
    downloader = NGSIMDownloader(download_dir)

    print("NGSIM Downloader Demo")
    print("=" * 50)

    # List available datasets
    print("\n1. Available datasets:")
    datasets = downloader.list_available_datasets()
    for name in datasets:
        print(f"   - {name}")

    # Get dataset information
    print("\n2. Dataset information for 'us101':")
    info = await downloader.get_dataset_info("us101")
    print(f"   Description: {info['description']}")
    print(f"   Base URL: {info['base_url']}")
    print(f"   Location filter: {info['location_filter']}")
    print(f"   Format: {info['format']}")

    # Download a small sample (first 10 records) in JSON format
    print("\n3. Downloading small sample from US-101 dataset (JSON)...")
    try:
        json_path = await downloader.download_dataset(
            "us101", output_format="json", limit=10
        )
        print(f"   Downloaded to: {json_path}")
        print(f"   File size: {json_path.stat().st_size} bytes")

        # Show first few lines
        content = json_path.read_text()[:200]
        print(f"   Content preview: {content}...")

    except Exception as e:
        print(f"   Error: {e}")

    # Download the same sample in CSV format
    print("\n4. Downloading small sample from US-101 dataset (CSV)...")
    try:
        csv_path = await downloader.download_dataset(
            "us101", output_format="csv", limit=10
        )
        print(f"   Downloaded to: {csv_path}")
        print(f"   File size: {csv_path.stat().st_size} bytes")

        # Show first few lines
        lines = csv_path.read_text().split("\n")[:3]
        print("   Content preview:")
        for line in lines:
            print(f"     {line}")

    except Exception as e:
        print(f"   Error: {e}")

    print("\n5. Demo completed!")
    print(f"   Files saved to: {download_dir.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())
