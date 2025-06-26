import pandas as pd
import csv
from seal_wrapper import SimulatedFHE

def run_fhe_benchmark(input_csv, output_csv):
    fhe = SimulatedFHE()
    df = pd.read_csv(input_csv)
    results = []

    for row_index, row in df.iterrows():
        try:
            # Convert each row of dataset to float vector (plaintext vector)
            plaintext_vector = list(map(float, row.values))

            # Encrypt the same row twice to simulate homomorphic addition
            ct1, enc_time1, ct1_size, pt1_size, expansion1, noise1 = fhe.encrypt(plaintext_vector)
            ct2, enc_time2, ct2_size, pt2_size, expansion2, noise2 = fhe.encrypt(plaintext_vector)

            # Homomorphic addition of ciphertexts
            added_ct, add_time, noise_after_add = fhe.add(ct1, ct2)

            # Decryption
            decrypted_result, dec_time = fhe.decrypt(added_ct)

            # Record metrics
            results.append({
                "RowIndex": row_index,
                "EncryptTimeTotal": enc_time1 + enc_time2,
                "AdditionTime": add_time,
                "DecryptionTime": dec_time,
                "PlaintextSize": pt1_size,
                "CiphertextSize": ct1_size,
                "ExpansionRatio": expansion1,
                "InitialNoise": noise1,
                "NoiseAfterAddition": noise_after_add
            })

        except Exception as e:
            print(f"Error processing row {row_index}: {e}")

    # Save metrics to CSV
    with open(output_csv, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("FHE benchmarking complete. Metrics saved.")
