# Day 2 MLOps Pipeline: ML Model Training with MLflow and GitHub Actions


This repository contains a basic MLOps pipeline built using Python, Scikit-learn, MLflow, and GitHub Actions.

The project trains a machine learning model on the Iris dataset, tracks model parameters and accuracy using MLflow, and runs the training pipeline automatically through GitHub Actions whenever code is pushed to the main branch.

## Project Overview

This project demonstrates a simple end-to-end MLOps workflow:

* Load the Iris dataset
* Split the dataset into training and testing data
* Train a Random Forest Classifier
* Track parameters using MLflow
* Track model accuracy using MLflow
* Save and log the trained model
* Run the training script automatically using GitHub Actions CI pipeline

## Tech Stack

* Python
* Scikit-learn
* MLflow
* Pandas
* NumPy
* GitHub Actions

## Project Structure

```text
day2_mlops/
│
├── .github/
│   └── workflows/
│       └── test.yml
│
├── mlruns/
├── mlflow.db
├── requirement.txt
├── train.py
├── .gitignore
└── README.md
```

## How the Project Works

The `train.py` file performs the following steps:

1. Loads the Iris dataset.
2. Splits the data into training and testing sets.
3. Creates a Random Forest Classifier model.
4. Trains the model.
5. Calculates model accuracy.
6. Logs parameters and metrics using MLflow.
7. Saves the trained model as an MLflow artifact.
8. Fails the pipeline if model accuracy is below 90%.

## Installation

Clone the repository:

```bash
git clone https://github.com/Mandeep2670/day2_mlops.git
cd day2_mlops
```

Install the required dependencies:

```bash
pip install -r requirement.txt
```

## Run the Project

To run the training script locally:

```bash
python train.py
```

After running the script, you should see output similar to:

```text
Training completed
Accuracy: <model_accuracy>
```

## MLflow Tracking

This project uses MLflow to track:

* Model parameters
* Model accuracy
* Trained model artifacts

Logged parameters include:

* `n_estimators`
* `max_depth`

Logged metric:

* `accuracy`

## GitHub Actions CI Pipeline

This repository includes a GitHub Actions workflow located at:

```text
.github/workflows/test.yml
```

The workflow runs automatically whenever code is pushed to the `main` branch.

The CI pipeline performs these steps:

1. Fetches the repository code.
2. Sets up Python 3.10.
3. Installs dependencies from `requirement.txt`.
4. Runs the training script using:

```bash
python train.py
```

## Purpose of This Project

The main purpose of this project is to understand the basics of MLOps, including:

* Machine learning model training
* Experiment tracking
* Model logging
* CI pipeline automation
* GitHub Actions workflow setup

## Author

Made by Mandeep Malik
