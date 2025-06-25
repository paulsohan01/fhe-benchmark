from seal_wrapper import DummySEAL
import pandas as pd
import csv

def run_benchmark_on_dataset(csv_path, output_csv):
    fhe = DummySEAL()
    df = pd.read_csv(csv_path)

    metrics = []
    for idx, row in df.iterrows():
        plain1 = list(map(float, row.values))
        plain2 = list(map(float, row.values))# simulate repeated values

        ct1, enc_time1, ct_size1, pt_size1, expansion1, noise1 = fhe.encrypt(plain1)
        ct2, enc_time2, ct_size2, pt_size2, expansion2, noise2 = fhe.encrypt(plain2)

        ct_result, add_time, noise_add = fhe.add(ct1, ct2)

        pt_result, dec_time = fhe.decrypt(ct_result)

        metrics.append({
            'Row': idx,
            'Plaintext Size (bytes)': pt_size1,
            'Ciphertext Size (bytes)': ct_size1,
            'Ciphertext Expansion': expansion1,
            'Encryption Time (s)': enc_time1 + enc_time2,
            'Addition Time (s)': add_time,
            'Decryption Time (s)': dec_time,
            'Noise Before Add': noise1,
            'Noise After Add': noise_add
        })

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    print(f"✅ Benchmark complete. Results saved to {output_csv}")
