## -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from tenseal_utils import TenSealManager
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from client import prepare_and_send_data_toserver
from server import server_train_linear_mean_classifier
from predictions import predict_linear_mean_model


CSV_FILE_PATH = "boston.csv"  # Default dataset file
OUTPUT_DIR = "results"  # Default output directory for metrics



# Run the benchmark for Linear Means Classifier by CKKS scheme in TenSEAL
# This function benchmarks the Linear Means Classifier using the CKKS scheme in TenSEAL.
# It prepares the data, trains the model, performs inference, and evaluates metrics.
# It supports both plaintext and encrypted training modes, and saves the results to a CSV file. 
def benchmark_tenseal_ckks_for_lmc(csv_path=CSV_FILE_PATH, output_dir=OUTPUT_DIR, scheme="ckks", training_mode="plaintext", verbose=False):
    """
    Run the benchmark for Linear Means Classifier using CKKS scheme in TenSEAL.
    Parameters:
        csv_path (str): Path to the CSV dataset file.
        output_dir (str): Directory to save the output metrics.
        scheme (str): Encryption scheme to use, default is 'ckks'.
        training_mode (str): 'plaintext' or 'encrypted' to select training approach.
        verbose (bool): If True, print detailed logs.
    Returns:
        None: Saves metrics to a CSV file in the specified output directory.
    """

    os.makedirs(output_dir, exist_ok=True)

    if training_mode == "plaintext":
        output_path = os.path.join(output_dir, "metrics_tenseal_ckks_lmc_plaintext.csv")
    elif training_mode == "encrypted":
        output_path = os.path.join(output_dir, "metrics_tenseal_ckks_lmc_encrypted.csv")
    else:
        raise ValueError("Invalid training_mode. Use 'plaintext' or 'encrypted'.")

    if verbose:
        print(f"Loading dataset from {csv_path}...")

    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Convert regression target to binary classification if needed
    if df.iloc[:, -1].nunique() > 10:
        df.iloc[:, -1] = pd.qcut(df.iloc[:, -1], q=2, labels=[0, 1])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----------------------------------------
    # Data preparation and server-side training
    # ----------------------------------------
    preparation_start = time.time()
    prepare_and_send_data_toserver(X_train, y_train, training_mode=training_mode, scheme_type=scheme, verbose=verbose)
    preparation_end = time.time()

    train_start = time.time()
    class_means = server_train_linear_mean_classifier(training_mode=training_mode, scheme=scheme, verbose=verbose)
    train_end = time.time()

    # ----------------------------------------
    # Inference
    # ----------------------------------------
    tenseal_manager = TenSealManager(scheme=scheme)

    # Encrypt the test data
    enc_X_test = tenseal_manager.encrypt(X_test)

    if verbose:
        print(f"Encrypted {len(enc_X_test)} test vectors using {scheme} scheme.")

    predictions = []
    infer_start = time.time()
    for enc_vector in enc_X_test:
        pred = predict_linear_mean_model(enc_vector, class_means, tenseal_manager, training_mod=training_mode, scheme_type=scheme, verbose=verbose)
        predictions.append(pred)
    infer_end = time.time()

    # ----------------------------------------
    # Metrics
    # ----------------------------------------
    accuracy = accuracy_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    metrics = {
        "training_time": train_end - train_start,
        "preparation_time": preparation_end - preparation_start,
        "inference_time": infer_end - infer_start,
        "accuracy": accuracy,
        "mse": mse,
        "scheme": scheme,
        "training_mode": training_mode
    }

    # Save metrics to CSV (overwrite mode)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path, index=False)

    if verbose:
        print(f"Benchmark completed. Metrics saved to {output_path}.")
        print(f"Preparation time: {metrics['preparation_time']:.4f} seconds")
        print(f"Training time: {metrics['training_time']:.4f} seconds")
        print(f"Inference time: {metrics['inference_time']:.4f} seconds")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")





## Run the benchmark for Linear Means Classifier by BFV scheme in TenSEAL
# This function benchmarks the Linear Means Classifier using the BFV scheme in TenSEAL.
# It prepares the data, trains the model, performs inference, and evaluates metrics.
# It supports both plaintext and encrypted training modes, and saves the results to a CSV file.
def benchmark_tenseal_bfv_for_lmc(csv_path="boston.csv", output_dir="results", scheme="bfv", training_mode="plaintext", verbose=False):
    """
    Run the benchmark for Linear Means Classifier using BFV scheme in TenSEAL.
    
    Parameters:
        csv_path (str): Path to the CSV dataset file.
        output_dir (str): Directory to save the output metrics.
        scheme (str): Encryption scheme to use, default is 'bfv'.
        training_mode (str): 'plaintext' or 'encrypted' to select training approach.
        verbose (bool): If True, print detailed logs.
    returns:
        None: Saves metrics to a CSV file in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set output filename based on training mode
    if training_mode == "plaintext":
        output_path = os.path.join(output_dir, "metrics_tenseal_bfv_lmc_plaintext.csv")
    elif training_mode == "encrypted":
        output_path = os.path.join(output_dir, "metrics_tenseal_bfv_lmc_encrypted.csv")
    else:
        raise ValueError("Invalid training_mode. Use 'plaintext' or 'encrypted'.")

    if verbose:
        print(f"Loading dataset from {csv_path}...")

    # Load dataset
    df = pd.read_csv(csv_path)

    # Convert regression to binary classification if needed
    if df.iloc[:, -1].nunique() > 10:
        df.iloc[:, -1] = pd.qcut(df.iloc[:, -1], q=2, labels=[0, 1])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----------------------------
    # Data preparation and server training
    # ----------------------------
    preparation_start = time.time()
    prepare_and_send_data_toserver(X_train, y_train, training_mode=training_mode, scheme_type=scheme, verbose=verbose)
    preparation_end = time.time()

    train_start = time.time()
    class_means = server_train_linear_mean_classifier(training_mode=training_mode, scheme=scheme,  verbose=verbose)
    train_end = time.time()

    # ----------------------------
    # Inference
    # ----------------------------
    tenseal_manager = TenSealManager(scheme=scheme)
    enc_X_test = []

    for x in X_test:
        x_int = (x * 1000).astype(int)  # Scale and convert to int for BFV
        enc_vec = tenseal_manager.encrypt(x_int)
        enc_X_test.append(enc_vec)

    if verbose:
        print(f"Encrypted {len(enc_X_test)} test vectors using {scheme} scheme.")

    predictions = []
    infer_start = time.time()
    for enc_vector in enc_X_test:
        pred = predict_linear_mean_model(enc_vector, class_means, tenseal_manager, training_mod=training_mode, scheme_type=scheme, verbose=verbose)
        predictions.append(pred)
    infer_end = time.time()

    if verbose:
        print(f"Inference completed for {len(predictions)} test vectors.")

    # ----------------------------
    # Evaluation Metrics
    # ----------------------------
    accuracy = accuracy_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    metrics = {
        "training_time": train_end - train_start,
        "preparation_time": preparation_end - preparation_start,
        "inference_time": infer_end - infer_start,
        "accuracy": accuracy,
        "mse": mse,
        "scheme": scheme,
        "training_mode": training_mode
    }

    # Save metrics (overwrite)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path, index=False)

    if verbose:
        print(f"Benchmark completed. Metrics saved to {output_path}.")
        print(f"Preparation time: {metrics['preparation_time']:.4f} seconds")
        print(f"Training time: {metrics['training_time']:.4f} seconds")
        print(f"Inference time: {metrics['inference_time']:.4f} seconds")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")












