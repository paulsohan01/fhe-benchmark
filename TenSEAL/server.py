# -*- coding: utf-8 -*-
# server.py
# This script implements the server-side training of a Linear Means Classifier (LMC) on encrypted data using the TenSEAL library.
# The training data is encrypted and sent from the client side, and the server decrypts it to train the model.
# The trained model is then saved to a file for future use or evaluation.
import os
import pickle
from tenseal_utils import TenSealManager 
from lm_training import train_linear_mean_classifier



ENCRYPTED_DATA_PATH = "encrypted_data.pkl"  # Path to the encrypted data file
PUBLIC_CONTEXT_PATH = "public_context.ctx"  # Path to save the public context file
OUTPUT_MODEL_PATH = "encrypted_lmc_model.pkl"  # Path to save the trained model



def server_train_plaintext_data(verbose=False):
    """
    Server side function to train a Linear Means Classifier on plaintext data.
    This function receives the plaintext training data, trains the model, and saves it.
    
    Parameters:
        verbose (bool): If True, print additional information during training.
    
    Returns:
        dict: A dictionary containing the class means for each class.
    """
    if verbose:
        print("[Server] Loading plaintext training data...")

    # Check input files
    if not os.path.exists(ENCRYPTED_DATA_PATH):
        raise FileNotFoundError(f"[Error] Required file not found: {ENCRYPTED_DATA_PATH}")

    # Load the plaintext data from file
    if verbose:
        print(f"[Server] Loading plaintext training data from {ENCRYPTED_DATA_PATH}...")
    with open(ENCRYPTED_DATA_PATH, 'rb') as f:
        X_train, y_train = pickle.load(f)

    if verbose:
        print(f"[Server] Loaded {len(X_train)} plaintext training vectors and calling 'train_linear_mean_classifier'...")

    # Train the Linear Means Classifier on plaintext data
    lmc_model = train_linear_mean_classifier(X_train, y_train, training_mode='plaintext')

    if verbose:
        print("[Server] Training completed. Saving the trained model...")

    # Save the trained model to a file
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump(lmc_model, f)

    if verbose:
        print(f"[Server] Trained model saved to {OUTPUT_MODEL_PATH}.")
    
    return lmc_model  # Return the trained model for further use or evaluation




def server_train_encrypted_data_byckks( verbose=False):
    tenseal_manager = TenSealManager(verbose=verbose)
    tenseal_manager.load(PUBLIC_CONTEXT_PATH)  # Load the public context

    if verbose:
        print("[Server] Loading encrypted training data...")



    # Check input files
    for file_path in [ENCRYPTED_DATA_PATH, PUBLIC_CONTEXT_PATH]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Error] Required file not found: {file_path}")


    # Load the encrypted data from file
    with open(ENCRYPTED_DATA_PATH, 'rb') as f:
        serialized_enc_X_train, y_train = pickle.load(f)


    # Deserialize each encrypted vector
    enc_X_train = [tenseal_manager.deserialize(vec_bytes) for vec_bytes in serialized_enc_X_train]
    if verbose:
        print(f"[Server] Loaded {len(enc_X_train)} encrypted training vectors.")



    lmc_model = train_linear_mean_classifier(enc_X_train, y_train, training_mode="encrypted", scheme="ckks")
    if verbose:
        print("[Server] Training completed. Saving the trained model...")



    serialized_lmc_model = {cls: enc_mean.serialize() for cls, enc_mean in lmc_model.items()}
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump(serialized_lmc_model, f)
    if verbose:
        print(f"[Server] Trained model saved to {OUTPUT_MODEL_PATH}.")

    return serialized_lmc_model  # Return the trained model for further use or evaluation 






def server_train_encrypted_data_bybfv(verbose=False):
    tenseal_manager = TenSealManager(verbose=verbose)
    tenseal_manager.load(PUBLIC_CONTEXT_PATH)  # Load the public context

    if verbose:
        print("[Server] Loading encrypted training data...")

    # Check input files
    for file_path in [ENCRYPTED_DATA_PATH, PUBLIC_CONTEXT_PATH]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Error] Required file not found: {file_path}")

    # Load the encrypted data from file
    with open(ENCRYPTED_DATA_PATH, 'rb') as f:
        serialized_enc_X_train, y_train = pickle.load(f)

    # Deserialize each encrypted vector
    enc_X_train = [tenseal_manager.deserialize(vec_bytes) for vec_bytes in serialized_enc_X_train]
    if verbose:
        print(f"[Server] Loaded {len(enc_X_train)} encrypted training vectors.")

    lmc_model = train_linear_mean_classifier(enc_X_train, y_train, training_mode="encrypted", scheme="bfv")
    if verbose:
        print("[Server] Training completed. Saving the trained model...")
    
 
    # lmc_model is a tuple: (class_sums: dict[int, BFVVector], class_counts: dict[int, int])
    class_sums, class_counts = lmc_model
    # Serialize the encrypted vectors before pickling
    serialized_class_sums = {cls: vec.serialize() for cls, vec in class_sums.items()}
    serialized_lmc_model = (serialized_class_sums, class_counts)

    # Save the trained model to a file
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump(serialized_lmc_model, f)

    if verbose:
        print(f"[Server] Trained model saved to {OUTPUT_MODEL_PATH}.")

    return serialized_lmc_model  # Return the serialized model










def server_train_linear_mean_classifier(training_mode='plaintext', scheme='ckks', verbose=False):
    """
    Server side function to train a Linear Means Classifier (LMC) on encrypted data.
    
    Parameters:
        training_mode (str): 'plaintext' for training on plaintext data, 'encrypted' for encrypted data.
        scheme (str): 'ckks' or 'bfv' for the encryption scheme.
        verbose (bool): If True, print additional information during training.
    
    Returns:
        dict: A dictionary containing the class means for each class.
    """
    if training_mode.lower() == 'plaintext':
        return server_train_plaintext_data(verbose)
    elif training_mode.lower() == 'encrypted':
        if scheme.lower() == 'ckks':
            return server_train_encrypted_data_byckks(verbose)
        elif scheme.lower() == 'bfv':
            return server_train_encrypted_data_bybfv(verbose)
        else:
            raise ValueError("Unsupported scheme type. Use 'ckks' or 'bfv'.")
    else:
        raise ValueError("Unsupported training mode. Use 'plaintext' or 'encrypted'.")