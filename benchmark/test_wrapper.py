import seal_wrapper as sw

def test():
    parms = sw.EncryptionParameters(sw.scheme_type.BFV)
    # Set parameters
    parms.set_poly_modulus_degree(4096)
    parms.set_coeff_modulus(sw.CoeffModulus.BFVDefault(4096,sw.sec_level_type.tc128))
    parms.set_plain_modulus(1032193)
    
    #Create SEALContext
    context = sw.SEALContext(parms)

    # Generate keys
    keygen = sw.KeyGenerator(context)
    sk = keygen.secret_key()
    pk = keygen.create_public_key()
    gk = keygen.create_galois_keys()
    rk = keygen.create_relin_keys()

    print("Public key:", pk)
    print("Secret key:", sk)
    print("Galois keys:", gk)
    print("Relin keys:", rk)
    
    print("<====================================================>")
    #Public key encryption
    encryptor = sw.Encryptor(context, pk)
    plain_text = sw.Plaintext("1")
    cipher_text = encryptor.encrypt(plain_text)
    print("Ciphertext:", cipher_text)
    print("Test passed: All Upto public key encryption are valid.")

test()