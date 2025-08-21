"""Command line interface for trajectory prediction."""

from pathlib import Path

import typer

app = typer.Typer(
    name="trajectory-predict",
    help="Advanced vehicle trajectory prediction pipeline",
)


@app.command()
def train_command(config_file: Path) -> None:
    """Train trajectory prediction models."""
    typer.echo(f"Training models with config: {config_file}")
    # Implementation will be added in future milestones


@app.command()
def predict_command(
    model_path: Path,
    input_data: Path,
    output_path: Path,
) -> None:
    """Make trajectory predictions."""
    typer.echo(f"Making predictions using model: {model_path}")
    typer.echo(f"Input data: {input_data}")
    typer.echo(f"Output path: {output_path}")
    # Implementation will be added in future milestones


@app.command()
def evaluate_command(
    results_path: Path,
    ground_truth: Path,
    output_path: Path,
) -> None:
    """Evaluate trajectory prediction results."""
    typer.echo(f"Evaluating results from: {results_path}")
    typer.echo(f"Ground truth: {ground_truth}")
    typer.echo(f"Output path: {output_path}")
    # Implementation will be added in future milestones


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
