import sys
from timer import Timer

class SimulatedFHE:
    def __init__(self):
        self.timer = Timer()

    def encrypt(self, plaintext_vector):
        """
        Simulates encryption and estimates latency, ciphertext size, expansion, and noise.
        """
        self.timer.start()
        ciphertext_vector = [x * 2 for x in plaintext_vector]  # mock encryption
        encryption_time = self.timer.stop()

        ciphertext_size = sys.getsizeof(ciphertext_vector)
        plaintext_size = sys.getsizeof(plaintext_vector)
        expansion_ratio = ciphertext_size / plaintext_size if plaintext_size else 0

        # Simpler noise model: linearly proportional to total input
        noise_estimate = 0.01 * sum(plaintext_vector)

        return ciphertext_vector, encryption_time, ciphertext_size, plaintext_size, expansion_ratio, noise_estimate

    def decrypt(self, ciphertext_vector):
        """
        Simulates decryption and returns time taken.
        """
        self.timer.start()
        decrypted = [x // 2 for x in ciphertext_vector]  # reverse of encryption
        decryption_time = self.timer.stop()
        return decrypted, decryption_time

    def add(self, ciphertext_vector1, ciphertext_vector2):
        """
        Simulates homomorphic addition and estimates new noise.
        """
        self.timer.start()
        result_ciphertext = [a + b for a, b in zip(ciphertext_vector1, ciphertext_vector2)]
        addition_time = self.timer.stop()

        # Additive noise: sum proportional again
        noise_estimate = 0.01 * sum(result_ciphertext)

        return result_ciphertext, addition_time, noise_estimate
