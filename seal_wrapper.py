from timer import Timer
import sys

class DummySEAL:
    def __init__(self):
        self.timer = Timer()

    def encrypt(self, plaintext):
        self.timer.start()
        ct = [x * 2 for x in plaintext]
        time_taken = self.timer.stop()
        ct_size = sys.getsizeof(ct)
        pt_size = sys.getsizeof(plaintext)
        expansion = ct_size / pt_size if pt_size else 0
        noise = 0.01 * sum(plaintext)  # mock noise
        return ct, time_taken, ct_size, pt_size, expansion, noise

    def decrypt(self, ciphertext):
        self.timer.start()
        pt = [x // 2 for x in ciphertext]
        time_taken = self.timer.stop()
        return pt, time_taken

    def add(self, ct1, ct2):
        self.timer.start()
        result = [a + b for a, b in zip(ct1, ct2)]
        time_taken = self.timer.stop()
        noise = 0.005 * sum(result)  # mock noise growth
        return result, time_taken, noise
