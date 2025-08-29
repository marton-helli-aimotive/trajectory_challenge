"""Command-line interface for trajectory prediction pipeline."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler

from .data import DataSourceFactory, TrajectoryETLPipeline

app = typer.Typer(name="trajectory-predict")
console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Setup logging with Rich handler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@app.command()
def init(
    config_path: str = typer.Option("configs/config.yaml", help="Configuration file path"),
    force: bool = typer.Option(False, help="Force initialization even if data exists")
) -> None:
    """Initialize the trajectory prediction pipeline."""
    console.print("[bold green]Initializing Trajectory Prediction Pipeline[/bold green]")
    
    # Load configuration
    if not Path(config_path).exists():
        console.print(f"[red]Configuration file not found: {config_path}[/red]")
        raise typer.Exit(1)
    
    config = OmegaConf.load(config_path)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    
    # Run initialization
    asyncio.run(_run_init(config, force))


@app.command()
def validate_data(
    config_path: str = typer.Option("configs/config.yaml", help="Configuration file path"),
    source: str = typer.Option("ngsim", help="Data source to validate")
) -> None:
    """Validate trajectory data quality."""
    console.print(f"[bold blue]Validating {source} data source[/bold blue]")
    
    config = OmegaConf.load(config_path)
    setup_logging()
    
    asyncio.run(_run_validation(config, source))


async def _run_init(config: DictConfig, force: bool) -> None:
    """Run pipeline initialization."""
    try:
        # Create data source
        source_type = config.get("data", {}).get("source", "ngsim")
        data_source = DataSourceFactory.create(source_type, config)
        
        # Initialize ETL pipeline
        pipeline = TrajectoryETLPipeline(config, data_source)
        
        # Run pipeline
        result = await pipeline.run(force_refresh=force)
        
        if result["status"] == "success":
            console.print(f"[green]✓ Pipeline initialized successfully![/green]")
            console.print(f"Records processed: {result['records_processed']}")
            console.print(f"Duration: {result['duration_seconds']:.2f}s")
        else:
            console.print(f"[red]✗ Pipeline failed: {result['error']}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]✗ Initialization failed: {e}[/red]")
        raise typer.Exit(1)


async def _run_validation(config: DictConfig, source: str) -> None:
    """Run data source validation."""
    try:
        data_source = DataSourceFactory.create(source, config)
        validation_result = await data_source.validate_source()
        
        console.print(f"Source Available: {validation_result['source_available']}")
        console.print(f"Data Integrity: {validation_result['data_integrity']}")
        console.print(f"Schema Compliance: {validation_result['schema_compliance']}")
        
        if validation_result["errors"]:
            console.print("[red]Errors:[/red]")
            for error in validation_result["errors"]:
                console.print(f"  - {error}")
        else:
            console.print("[green]✓ All validations passed![/green]")
            
    except Exception as e:
        console.print(f"[red]✗ Validation failed: {e}[/red]")
        raise typer.Exit(1)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()