# Scripts

This directory contains utility scripts for:

- Data preprocessing and ETL
- Model training and evaluation
- Deployment and infrastructure
- Maintenance and monitoring

## Structure

```
scripts/
├── setup_data.py          # Download and prepare datasets
├── train_models.py        # Batch model training
├── evaluate_models.py     # Model comparison and validation
├── deploy.py              # Deployment automation
└── README.md              # This file
```

## Usage

Scripts can be run from the project root with the virtual environment activated:

```bash
source .venv/bin/activate
python scripts/setup_data.py --help
```

## Guidelines

- Scripts should use the CLI interface where possible
- Include proper argument parsing with `typer` or `argparse`
- Add logging and error handling
- Make scripts idempotent where possible
