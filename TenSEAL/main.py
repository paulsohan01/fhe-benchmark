## -*- coding: utf-8 -*-
from benchmark import benchmark_tenseal_ckks_for_lmc
from benchmark import benchmark_tenseal_bfv_for_lmc

if __name__ == "__main__":
    print("<==========Starting benchmark for Linear Means Classifier with TenSEAL==========>\n")
    # Benchmark using CKKS scheme
    print("Generating benchmark metrics for Linear Means Classifier using CKKS scheme in TenSEAL...\n")
    print("<========== First case: Training on plaintext data and inference on encrypted data ==========>\n")
    benchmark_tenseal_ckks_for_lmc(csv_path="boston.csv", output_dir="results_ckks", scheme="ckks", training_mode="plaintext", verbose=True)
    print("<========== Second case: Training on encrypted data and inference on encrypted data ==========>\n")
    benchmark_tenseal_ckks_for_lmc(csv_path="boston.csv", output_dir="results_ckks", scheme="ckks", training_mode="encrypted", verbose=True)

    print("Benchmarking completed for Linear Means Classifier with TenSEAL using CKKS scheme.\n")
    # Benchmark using BFV scheme
    print("Generating benchmark metrics for Linear Means Classifier using BFV scheme in TenSEAL...\n")
    print("<========== First case: Training on plaintext data and inference on encrypted data ==========>\n")   
    benchmark_tenseal_bfv_for_lmc(csv_path="boston.csv", output_dir="results_bfv", scheme="bfv", training_mode="plaintext", verbose=True)
    print("<========== Second case: Training on encrypted data and inference on encrypted data ==========>\n")
    benchmark_tenseal_bfv_for_lmc(csv_path="boston.csv", output_dir="results_bfv", scheme="bfv", training_mode="encrypted", verbose=True)
    # Print completion message
    print("Benchmarking completed for Linear Means Classifier with TenSEAL.")


