import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

from ml_utils import eval_metrics

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

DATA_URL = os.environ["CSV_URL"]
TARGET_VAR = os.environ["TARGET_VAR"]
TRAINING_PARAMS = {
    "alpha": float(os.environ["ALPHA"]),
    "l1_ratio": float(os.environ["L1_RATIO"]),
    "random_state": int(os.environ["RANDOM_STATE"])
}

def ingest_data():
    """Read the wine-quality csv file from env variable"""
    try:
        data = pd.read_csv(DATA_URL, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    return data

def split_data(data, target):
    """Assumes data is a pandas dataframe and target_var is a str"""
    np.random.seed(40)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    data_out = {
        "train_x": train.drop([target], axis=1),
        "test_x": test.drop([target], axis=1),
        "train_y": train[[target]],
        "test_y": test[[target]]
    }

    return data_out

def train_model(data: dict):
    params = TRAINING_PARAMS
    clf = ElasticNet(**params)
    clf.fit(data['train_x'], data['train_y'])
    return clf

def get_prediction(data: dict, model: ElasticNet):
    return model.predict(data["test_x"])

def evaluate_model(data: dict, prediction):
    (rmse, mae, r2) = eval_metrics(data['test_y'], prediction)
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    return metrics

def train():
    print(f"Starting MLflow run with tracking URI: {mlflow.get_tracking_uri()}")
    mlflow.start_run()
    try:
        df = ingest_data()
        data = split_data(df, TARGET_VAR)
        model = train_model(data)
        predictions = model.predict(data["test_x"])
        (rmse, mae, r2) = eval_metrics(data["test_y"], predictions)

        print(f"ElasticNet model (alpha={TRAINING_PARAMS['alpha']}, l1_ratio={TRAINING_PARAMS['l1_ratio']}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", TRAINING_PARAMS['alpha'])
        mlflow.log_param("l1_ratio", TRAINING_PARAMS['l1_ratio'])
        mlflow.log_param("target_var", TARGET_VAR)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticNetWineModel")
        else:
            mlflow.sklearn.log_model(model, "model")
    finally:
        mlflow.end_run()
        print("MLflow run ended.")

if __name__ == "__main__":
    train()