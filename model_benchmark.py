'''import pandas as pd
import time
from seal_wrapper import SEALWrapper

def run_benchmark_on_dataset(input_csv="boston.csv", output_csv="metrics.csv"):
    df = pd.read_csv(input_csv)
    feature_cols = df.columns[:-1]  # Last column is target
    target_col = df.columns[-1]

    fhe = SEALWrapper()

    metrics = []

    for index, row in df.iterrows():
        features = row[feature_cols].values.astype(float).tolist()
        label = row[target_col]

        # Encrypt features
        ct, enc_time, ct_size, pt_size, expansion = fhe.encrypt(features)

        # Simulate a linear inference (e.g., dot product with unit weights)
        weights = [1.0] * len(features)
        ct_result, op_time = fhe.linear_inference(ct, weights)

        # Decrypt and measure noise
        decrypted, dec_time, noise = fhe.decrypt(ct_result, [sum([a*b for a, b in zip(features, weights)])])

        metrics.append({
            "encryption_time": enc_time,
            "operation_time": op_time,
            "decryption_time": dec_time,
            "noise": noise,
            "plaintext_size": pt_size,
            "ciphertext_size": ct_size,
            "expansion_ratio": expansion
        })

    pd.DataFrame(metrics).to_csv(output_csv, index=False)
    print(f"✅ Benchmarking complete. Results saved to {output_csv}")'''
