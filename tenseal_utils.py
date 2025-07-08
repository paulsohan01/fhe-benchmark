# tenseal.py
import tenseal as ts
import numpy as np

# TenSEAL Manager for CKKS and BFV schemes
# This class manages the creation of TenSEAL contexts, encryption, decryption, and serialization
# It supports both CKKS and BFV schemes, allowing for flexible use in homomorphic encryption
# The class provides methods to encrypt and decrypt vectors, rotate encrypted vectors (only for CKKS),
# and serialize the context for saving/loading. It also includes error handling for unsupported operations.
class TenSealManager:
    def __init__(self, scheme="ckks", scale=2**40, poly_modulus_degree=8192, plain_modulus=1032193):
        """
        Create a TenSEAL context for either CKKS or BFV scheme.
        """
        if scheme.lower() == "ckks":
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            self.context.global_scale = scale
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()
            self.scheme = "ckks"
            print("[TenSEAL CKKS] Context created successfully.")
        elif scheme.lower() == "bfv":
            self.context = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=poly_modulus_degree,
                plain_modulus=plain_modulus
            )
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()
            self.scheme = "bfv"
            print("[TenSEAL BFV] Context created successfully.")
        else:
            raise ValueError("Unsupported scheme. Use 'ckks' or 'bfv'.")

    def _encrypt_vector(self, vector):
        if self.scheme == "ckks":
            enc = ts.ckks_vector(self.context, vector)
            #print("[Encryption] CKKS vector encrypted.")
            return enc
        elif self.scheme == "bfv":
            vector_int = np.round(vector).astype(int)
            enc = ts.bfv_vector(self.context, vector_int)
            #print("[Encryption] BFV vector encrypted (int-converted).")
            return enc
        else:
            raise ValueError(f"Unsupported scheme: {self.scheme}")

    def encrypt(self, data):
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                #print("[Encrypt] Encrypting 1D vector...")
                return self._encrypt_vector(data)
            elif data.ndim == 2:
                #print(f"[Encrypt] Encrypting batch of {len(data)} vectors...")
                return [self._encrypt_vector(row) for row in data]
            else:
                raise ValueError("Only 1D or 2D numpy arrays are supported.")
        else:
            raise TypeError("Input must be a numpy array.")

    def decrypt(self, encrypted_vector):
        if not hasattr(encrypted_vector, "decrypt"):
            raise TypeError("Object does not support decryption.")
        decrypted = encrypted_vector.decrypt()
        #print("[Decrypt] Vector decrypted.")
        return decrypted

    def rotate(self, encrypted_vector, steps):
        if self.scheme == "ckks":
            rotated = encrypted_vector.rotate(steps)
           # print(f"[Rotate] CKKS vector rotated by {steps} steps.")
            return rotated
        elif self.scheme == "bfv":
            raise NotImplementedError("Rotation is not supported in BFV scheme.")

    def serialize(self):
        #print("[Serialize] Context serialized.")
        return self.context.serialize()

    def save(self, path):
        with open(path, 'wb') as f:
            f.write(self.context.serialize())
        print(f"[Save] Context saved to {path}.")

    def load(self, path):
        with open(path, 'rb') as f:
            self.context = ts.context_from(f.read())
        print(f"[Load] Context loaded from {path}.")
