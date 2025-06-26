from test_addition import run_fhe_benchmark

if __name__ == "__main__":
    print("Running FHE Micro-Benchmark on Dataset...")
    run_fhe_benchmark("clean_boston.csv", "metrics.csv")
    print("Metrics written to metrics.csv")
