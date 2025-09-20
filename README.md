# California Housing Price Prediction with MLflow

A minimal MLOps template following Cookiecutter Data Science (CCDS) V2 conventions with src/ layout, integrated MLflow experiment tracking, and GitHub Actions CI for linting and testing. This project predicts California housing prices using a Random Forest Regressor and tracks experiments with MLflow.

## Quickstart

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   make install
   ```

3. Run tests:
   ```bash
   make test
   ```

4. Train the model:
   ```bash
   make train
   ```

5. View results in MLflow UI:
   ```bash
   make ui
   ```
   Open http://127.0.0.1:5000 in your browser.

## Project Structure

This project follows CCDS V2 conventions with a src/ layout for explicit imports:

- `data/`: Raw and processed data (gitignored).
- `models/`: Saved models.
- `notebooks/`: Jupyter notebooks for exploration.
- `reports/`: Generated reports and figures.
- `src/`: Source code with `train.py` for training.
- `tests/`: Unit tests.

## MLflow Integration

The training script (`src/train.py`) logs experiments to MLflow:

- **Experiment**: "california-housing-regression"
- **Run**: A new run with a descriptive name.
- **Parameters**: Hyperparameters like n_estimators and max_depth.
- **Metrics**: Mean Absolute Error (MAE) and R² Score.
- **Artifacts**: The trained model.

View logged data in the MLflow UI at http://127.0.0.1:5000.

## CI/CD

GitHub Actions runs on push/PR to main:

- Lints with Ruff.
- Runs tests with Pytest.

Add a CI status badge once pushed to GitHub.

## Requirements

- Python 3.11
- CPU-only, fast demo (<30s training).