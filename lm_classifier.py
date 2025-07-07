# lm_plain_classifier.py

import pandas as pd
import numpy as np
import time
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

def linear_means_train(X_train, y_train):
    """
    Trains the Linear Means Classifier: computes the class means.
    """
    class_means = {}
    for cls in np.unique(y_train):
        class_means[cls] = X_train[y_train == cls].mean(axis=0)
    return class_means

def linear_means_predict(X_test, class_means):
    """
    Predicts class by finding the nearest class mean in Euclidean space.
    """
    predictions = []
    for x in X_test:
        min_dist = float('inf')
        pred_cls = None
        for cls, mean_vec in class_means.items():
            dist = np.linalg.norm(x - mean_vec)
            if dist < min_dist:
                min_dist = dist
                pred_cls = cls
        predictions.append(pred_cls)
    return np.array(predictions)

def run_plain_lmc_benchmark(csv_path="boston.csv", output_path="metrics_plain_lmc.csv"):
    df = pd.read_csv(csv_path)

    # Assume the last column is the target for classification (convert regression to classification)
    if df.iloc[:, -1].nunique() > 10:
        df.iloc[:, -1] = pd.qcut(df.iloc[:, -1], q=2, labels=[0, 1])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training
    train_start = time.time()
    class_means = linear_means_train(X_train, y_train)
    train_end = time.time()

    # Inference
    infer_start = time.time()
    predictions = linear_means_predict(X_test, class_means)
    infer_end = time.time()

    accuracy = accuracy_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # Metrics
    metrics = {
        "training_time": train_end - train_start,
        "inference_time": infer_end - infer_start,
        "accuracy": accuracy,
        "mse": mse,
        "model_size_bytes": len(class_means) * X.shape[1] * 8  # float64 = 8 bytes
    }

    # Save to CSV
    df_out = pd.DataFrame([metrics])
    df_out.to_csv(output_path, index=False)
    print("Benchmark metrics written to", output_path)

if __name__ == "__main__":
    run_plain_lmc_benchmark()
