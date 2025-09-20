import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Set experiment
mlflow.set_experiment("california-housing-regression")

# Start run
with mlflow.start_run(run_name="RandomForest-Regressor-Run") as run:
    # Load data
    housing = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )

    # Hyperparams from env
    n_estimators = int(os.getenv("N_ESTIMATORS", 100))
    max_depth = int(os.getenv("MAX_DEPTH", 10))

    # Log params
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Predict and metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Print summary
    print(f"Training complete. MAE: {mae:.4f}, R2 Score: {r2:.4f}, Run ID: {run.info.run_id}")
