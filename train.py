import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Model parameters
n_estimators = 50
max_depth = 3

# Create model
model = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth, random_state=42
)

# Recent MLflow versions require an explicit opt-in for the local file store.
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Keep MLflow temp artifacts inside the workspace to avoid OS temp permission issues.
temp_dir = Path(".tmp").resolve()
temp_dir.mkdir(exist_ok=True)
os.environ["TMP"] = str(temp_dir)
os.environ["TEMP"] = str(temp_dir)
tempfile.tempdir = str(temp_dir)

# Keep tracking data in a clean repo-local folder instead of relying on stale state.
mlflow.set_tracking_uri(Path("mlruns_local").resolve().as_uri())

# Start MLflow tracking
with mlflow.start_run():

    # Train model
    model.fit(X_train, y_train)

    # Calculate accuracy
    accuracy = model.score(X_test, y_test)

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Save the model in a workspace temp folder, then log it as an artifact.
    run_id = mlflow.active_run().info.run_id
    model_dir = temp_dir / run_id / "model"
    mlflow.sklearn.save_model(model, str(model_dir))
    mlflow.log_artifacts(str(model_dir), artifact_path="model")

    print("Training completed")
    print("Accuracy:", accuracy)

    if accuracy < 0.90:
        raise Exception("Model accuracy is below 90%. Pipeline failed.")
