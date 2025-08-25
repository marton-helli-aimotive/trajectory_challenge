# trajectory-challenge

Python project using `uv`, Python 3.12, Ruff, Pyright, Pytest, focused on NGSIM trajectory data utilities.

## Setup

1. Install uv (see https://github.com/astral-sh/uv):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Sync the environment (installs dependencies):
   ```bash
   uv sync --all-extras --dev
   ```

## Run
Run via module or installed script:
```bash
uv run python -m trajectory_challenge
uv run trajectory-challenge  # console script
```

## GUI (Trajectory Explorer)
Install GUI extras then launch Streamlit app:
```bash
uv sync --extra gui
# Preferred: console script (sets up proper ScriptRunContext)
uv run trajectory-gui
# Alternate manual launch:
uv run streamlit run -m trajectory_challenge.gui

If you see a 'missing ScriptRunContext' warning, use the console script above.
```

## Test
```bash
uv run pytest -q
```

## Lint & Format
Run Ruff (lint + fix) and Pyright:
```bash
uv run ruff check --fix .
uv run ruff format .
uv run pyright
```
