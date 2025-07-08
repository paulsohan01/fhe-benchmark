# lm_tenseal.py

import pandas as pd
import numpy as np
import time
import os
import tenseal as ts

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from tenseal_utils import TenSealManager

# Linear Means Classifier training on plaintext data
def linear_means_train(X_train, y_train,verbose=False):
    """
    Parameters:
        X_train: np.ndarray, shape (n_samples, n_features)
        y_train: np.ndarray, shape (n_samples,)
    Returns:
        dict of class_id: mean_vector
    """
    if verbose:
        print("Training Linear Means Classifier...")

    class_means = {}
    for cls in np.unique(y_train):
        class_means[cls] = X_train[y_train == cls].mean(axis=0)

    if verbose:
        print("Training completed. Class means computed.")
    return class_means




# Linear Means Classifier prediction function (input: encrypted data)
def linear_means_predict_ckks(enc_X_test, class_means, tenseal_manager, verbose=False):
    """
    Predicts class by finding the nearest class mean in Euclidean space.
    
    Parameters:
        enc_X_test (ts.CKKSVector): Encrypted test vector
        class_means (dict): Dictionary of class label -> mean vector
        tenseal_manager (TenSealManager): TenSEAL encryption manager
        
    Returns:
        pred_class (int): Predicted class label
    """
    # Check if enc_X_test is a CKKS encrypted vector
    assert isinstance(enc_X_test, ts.CKKSVector), "Expected CKKS encrypted vector"

    # Initialize minimum distance and predicted class
    min_dist = float('inf')
    pred_class = None

    for cls, mean_vec in class_means.items():
        # Encrypt the class mean vector
        enc_mean_vec = tenseal_manager.encrypt(mean_vec)
        # Compute the distance between the encrypted test vector and the encrypted class mean
        enc_dist = enc_X_test - enc_mean_vec
        diff_sq = enc_dist* enc_dist
        dist = diff_sq.sum()

        # Decrypt the squared distance
        decrypted_dist = tenseal_manager.decrypt(dist) # it returns a list
        decrypted_dist = decrypted_dist[0]
        # Find the minimum distance and corresponding class
        if decrypted_dist < min_dist:
            min_dist = decrypted_dist
            pred_class = cls

    if verbose:
        print(f"Predicted class: {pred_class} with distance: {min_dist}")
    return pred_class





# Linear Means Classifier prediction function for BFV encrypted data
def linear_means_predict_bfv(enc_X_test, class_means, tenseal_manager, verbose=False):
    """
    Predicts class by finding the nearest class mean in Euclidean space for BFV encrypted data.
    Parameters:
        enc_X_test (ts.BFVVector): Encrypted test vector
        class_means (dict): Dictionary of class label -> mean vector
        tenseal_manager (TenSealManager): TenSEAL encryption manager
    Returns:
        pred_class (int): Predicted class label
    Note:
        This function assumes that the class means are already computed and stored as plaintext vectors.
        The input enc_X_test is expected to be a BFV encrypted vector.
        The class means should be integers since BFV only supports integer vectors.    
    """

    # Check if enc_X_test is a BFV encrypted vector
    assert isinstance(enc_X_test, ts.BFVVector), "Expected BFV encrypted vector"

    # Initialize minimum distance and predicted class
    min_dist = float('inf')
    pred_class = None

    for cls, mean_vec in class_means.items():
        # BFV only supports integer vectors, so we convert the mean vector to integers
        mean_vec_int = np.round(mean_vec).astype(int)
        # Encrypt the class mean vector
        enc_mean_vec = tenseal_manager.encrypt(mean_vec_int)
        # Compute the distance between the encrypted test vector and the encrypted class mean
        enc_dist = enc_X_test - enc_mean_vec
        diff_sq = enc_dist * enc_dist
        dist = diff_sq.sum()

        # Decrypt the squared distance
        decrypted_dist = tenseal_manager.decrypt(dist) # it returns a list
        decrypted_dist = decrypted_dist[0]  # Get the first element since it's a single value
        # Find the minimum distance and corresponding class
        if decrypted_dist < min_dist:
            min_dist = decrypted_dist
            pred_class = cls
    
    if verbose:
        print(f"Predicted class: {pred_class} with distance: {min_dist}")
    # Return the predicted class
    return pred_class




# Run the benchmark for Linear Means Classifier on plaintext data by ckks scheme in TenSEAl
def benchmark_tenseal_ckks_lmc(csv_path="boston.csv", output_dir="results", scheme="ckks", verbose=False):

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metrics_tenseal_ckks_lmc_plaintext.csv")

    if verbose:
        print(f"Loading dataset from {csv_path}...")

    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Assume the last column is the target for classification (convert regression to classification)
    if df.iloc[:, -1].nunique() > 10:
        df.iloc[:, -1] = pd.qcut(df.iloc[:, -1], q=2, labels=[0, 1])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training on plaintext data
    train_start = time.time()
    class_means = linear_means_train(X_train, y_train, verbose=verbose)
    train_end = time.time()

    # Initialize TenSealManager
    tenseal_manager = TenSealManager(scheme=scheme)
    # Encrypt the test data "X_test"
    enc_X_test = tenseal_manager.encrypt(X_test)
    
    if verbose:
        print(f"Encrypted {len(enc_X_test)} test vectors using {scheme} scheme.")

    # Inference
    infer_start = time.time()
    predictions = [linear_means_predict_ckks(enc_X_test[i], class_means, tenseal_manager) for i in range(len(enc_X_test))]
    infer_end = time.time()

    accuracy = accuracy_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # Metrics
    metrics = {
        "training_time": train_end - train_start,
        "inference_time": infer_end - infer_start,
        "accuracy": accuracy,
        "mse": mse,
        "scheme": scheme
    }

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    if os.path.exists(output_path):
        metrics_df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(output_path, index=False)

    if verbose:
        print(f"Benchmark completed. Metrics saved to {output_path}.")
        print(f"Training time: {metrics['training_time']:.4f} seconds")
        print(f"Inference time: {metrics['inference_time']:.4f} seconds")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")




# Run the benchmark for Linear Means Classifier on 
def benchmark_tenseal_bfv_lmc(csv_path="boston.csv", output_dir="results", scheme="bfv", verbose=False):
    """
    Run the benchmark for Linear Means Classifier on plaintext data by bfv scheme in TenSEAL.
    Parameters:
        csv_path (str): Path to the CSV dataset file.
        output_dir (str): Directory to save the output metrics.
        scheme (str): Encryption scheme to use, either 'ckks' or 'bfv'.
        verbose (bool): If True, print detailed logs.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metrics_tenseal_bfv_lmc_plaintext.csv")

    if verbose:
        print(f"Loading dataset from {csv_path}...")

    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Assume the last column is the target for classification (convert regression to classification)
    if df.iloc[:, -1].nunique() > 10:
        df.iloc[:, -1] = pd.qcut(df.iloc[:, -1], q=2, labels=[0, 1])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training on plaintext data "X_train"
    train_start = time.time()
    class_means = linear_means_train(X_train, y_train, verbose=verbose)
    train_end = time.time()

    # Initialize TenSealManager
    tenseal_manager = TenSealManager(scheme=scheme)
    
    if verbose:
        print(f"Initialized TenSealManager with {scheme} scheme.")
    # Encrypt the test data "X_test"
    enc_X_test = tenseal_manager.encrypt(X_test)

    if verbose:
        print(f"Encrypted {len(enc_X_test)} test vectors using {scheme} scheme.")

    # Inference
    predictions = []
    infer_start = time.time()
    for x in X_test:
        x__int = (x*1000).astype(int)  #Scale and Convert to integer for BFV
        enc_x = tenseal_manager.encrypt(x__int) # then encrypt
        pred = linear_means_predict_bfv(enc_x, class_means, tenseal_manager)
        predictions.append(pred)
    infer_end = time.time()

    if verbose:
        print(f"Inference completed for {len(predictions)} test vectors.")

    accuracy = accuracy_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # Metrics
    metrics = {
        "training_time": train_end - train_start,
        "inference_time": infer_end - infer_start,
        "accuracy": accuracy,
        "mse": mse,
        "scheme": scheme
    }

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    if os.path.exists(output_path):
        metrics_df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(output_path, index=False)

    if verbose:
        print(f"Benchmark completed. Metrics saved to {output_path}.")
        print(f"Training time: {metrics['training_time']:.4f} seconds")
        print(f"Inference time: {metrics['inference_time']:.4f} seconds")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")