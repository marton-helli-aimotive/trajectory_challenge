"""
CLI command for running the interactive dashboard.
"""

import click
import streamlit.web.cli as stcli
import sys
import os
from pathlib import Path


@click.command()
@click.option('--port', default=8501, help='Port to run the dashboard on')
@click.option('--host', default='localhost', help='Host to run the dashboard on')
@click.option('--browser', is_flag=True, help='Automatically open browser')
@click.option('--config', type=click.Path(exists=True), help='Path to configuration file')
def dashboard(port: int, host: str, browser: bool, config: str):
    """
    Run the interactive vehicle trajectory prediction dashboard.
    
    This command starts a Streamlit web application that provides:
    - Interactive trajectory visualization
    - Model comparison and evaluation
    - Dataset exploration tools
    - Model explainability features
    """
    # Get the path to the dashboard script
    dashboard_path = Path(__file__).parent.parent / "visualization" / "dashboard.py"
    
    if not dashboard_path.exists():
        click.echo(f"Error: Dashboard script not found at {dashboard_path}")
        sys.exit(1)
    
    # Set up Streamlit arguments
    sys.argv = [
        "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true" if not browser else "false"
    ]
    
    # Set environment variables for configuration
    if config:
        os.environ['DASHBOARD_CONFIG'] = config
    
    click.echo(f"Starting dashboard on http://{host}:{port}")
    click.echo("Press Ctrl+C to stop the dashboard")
    
    try:
        # Run the Streamlit app
        sys.exit(stcli.main())
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped by user")
        sys.exit(0)
    except Exception as e:
        click.echo(f"Error starting dashboard: {e}")
        sys.exit(1)


if __name__ == '__main__':
    dashboard()