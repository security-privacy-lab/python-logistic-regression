import math

from logistic_regression import LogisticRegression as CustomLogisticRegression
# from data import x_train, y_train, x_test, y_test
from data import x_scaled_train, y_train, x_scaled_test, y_test
import os
import pickle
import socket
import tenseal as ts
import numpy as np
import pandas
import sys

MODEL_DIRECTORY = '../../models'
MODEL_PATH = '../../models/Custom_Logistic_Regression2.pkl'

SIGMOID_POLY = [0.5, 0.197, 0, -0.004]

# Step 1: Create a TenSeal Context
cntx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192 * 2,  # MUST be 8192 * 2^x, where x is a non-negative integer
    # coeff_mod_bit_sizes = [60,40,40,60]
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60]
)
cntx.global_scale = 2 ** 40
cntx.generate_galois_keys()


print(os.path.exists(MODEL_DIRECTORY))

if not os.path.exists(MODEL_DIRECTORY):
    os.mkdir(MODEL_DIRECTORY)

if not os.path.exists(MODEL_PATH):
    # if True:
    print(f"[MODEL NOT FOUND] No model at {MODEL_PATH}")
    lr = CustomLogisticRegression()
    lr.fit(x_scaled_train, y_train, epochs=150)

    with open(MODEL_PATH, 'wb+') as file:
        print(f"[SAVING MODEL] Saving model at {MODEL_PATH}")
        pickle.dump(lr, file, pickle.HIGHEST_PROTOCOL)
        print(f"[SAVING MODEL] Model successfully saved locally")
else:
    print(f"[MODEL FOUND] Found model at {MODEL_PATH}")
    with open(MODEL_PATH, 'rb+') as file:
        print(f"[LOADING MODEL] Attempting to load model from {MODEL_PATH}")
        lr = pickle.load(file)
        print(f"[LOADING MODEL] Load successful")

print(lr.weights)

print("Automatic relinearization is:", ("on" if cntx.auto_relin else "off"))
print("Automatic rescaling is:", ("on" if cntx.auto_rescale else "off"))
print("Automatic modulus switching is:", ("on" if cntx.auto_mod_switch else "off"))

matches = 0

for i in range(114):
    plain_input = x_scaled_test.iloc[i]
    # plain_input = plain_input.to_numpy()
    encrypted_input = ts.ckks_vector(cntx, plain_input)
    enc_probability = lr.predictEncryptedSingle(encrypted_input)
    dec_probability = enc_probability.decrypt()[0]

    plain_probability = lr.predictSingleProb(x_scaled_test.iloc[i])

    dec_true_false = 1 if dec_probability > 0.5 else 0
    plain_true_false = 1 if plain_probability > 0.5 else 0

    print("TenSEAL probability:", dec_probability)
    print("plantext probability:", plain_probability)

    print("TenSEAL method:  ", dec_true_false)
    print("plaintext method:", plain_true_false)

    if dec_true_false == plain_true_false:
        matches += 1

print(f"Total matches: {matches}/114")
print(f"About {(matches / 114) * 100}% similar")

# transposed_x = x_scaled_test.transpose().to_numpy()

# encrypted_matrix = ts.ckks_tensor(cntx, x_scaled_test)
# print("encrypted entire test set")
# encrypted_matrix_batched = ts.ckks_tensor(cntx, x_scaled_test, batch=True)
# print("encrypted batched test set")
# # transp_enc_matrix_batched = ts.ckks_tensor(cntx, np.array(x_scaled_test).transpose(), batch=True)
#
# x_dot_weights = encrypted_matrix_batched.dot(lr.weights.transpose()) + lr.bias
# dec = np.array(x_dot_weights.decrypt().tolist())
# q = 1



# batched_serialized_encrypted_matrix = encrypted_matrix_batched.serialize()
# print("serialized entire test set")
# print(f"len of serialized matrix:                   {len(batched_serialized_encrypted_matrix)}")
# print(f"getSizeOf()'s size of non-serialized matrix:{sys.getsizeof(encrypted_matrix)}")
# print(f"getSizeOf()'s size of serialized matrix:    {sys.getsizeof(serialized_encrypted_matrix)}")
# pickled_serialized_encrypted_matrix = pickle.dumps(serialized_encrypted_matrix)
# print(f"pickled version of serialized matrix:       {len(pickled_serialized_encrypted_matrix)}")


# deserialized_encrypted_matrix = ts.ckks_tensor_from(cntx, serialized_encrypted_matrix)
# decrypted_deserialized_matrix = deserialized_encrypted_matrix.decrypt()
# transp_weights = lr.weights.transpose()
# transp_weights = ts.ckks_tensor(cntx, lr.weights.transpose(), batch=True)
# # transp_x_dot_weights = transp_enc_matrix_batched.dot(transp_weights) + lr.bias
# x_dot_weights = encrypted_matrix_batched.dot(transp_weights) + lr.bias
# dec_dot_weights = np.array(x_dot_weights.decrypt().tolist())
# print(dec_dot_weights)



# # juice = encrypted_matrix
# juice = encrypted_matrix.transpose()
# dec_juice = np.array(juice.decrypt().tolist())
# dec_matrix = np.array(encrypted_matrix.decrypt().tolist())
# print(dec_juice)
# print(dec_juice)

# test1 = encrypted_matrix_batched.ciphertext()
# print("test1 complete")
# test2 = encrypted_matrix_batched.reshape([30])
# print("test2 complete")
#
#
# print(np.array(test2.decrypt().tolist()))