#from lm_classifier import run_plain_lmc_benchmark
from lm_tenseal import benchmark_tenseal_ckks_lmc
from lm_tenseal import benchmark_tenseal_bfv_lmc 

if __name__ == "__main__":
    # Run the benchmark for Linear Means Classifier on plaintext data by ckks scheme in TenSEAL
    benchmark_tenseal_ckks_lmc(verbose=True)
    # Run the benchmark for Linear Means Classifier on plaintext data by bfv scheme in TenSEAL
    benchmark_tenseal_bfv_lmc()
