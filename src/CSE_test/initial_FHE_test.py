from logistic_regression import LogisticRegression as CustomLogisticRegression
# from data import x_train, y_train, x_test, y_test
from data import x_scaled_train, y_train, x_scaled_test, y_test
import os
import pickle
import socket
import tenseal as ts
import numpy as np
import pandas

MODEL_DIRECTORY = '../../models'
MODEL_PATH = '../../models/Custom_Logistic_Regression2.pkl'

SIGMOID_POLY = [0.5, 0.197, 0, -0.004]

# Step 1: Create a TenSeal Context
cntx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree = 8192 * 2,             # MUST be 8192 * 2^x, where x is a non-negative integer
    # coeff_mod_bit_sizes = [60,40,40,60]
    coeff_mod_bit_sizes = [60,40,40,40,40,40,40,60]
)
cntx.global_scale = 2**40

cntx.generate_galois_keys()


def encrypt_data(cntx, data):               # Encryption Function using CKKS
    encrypted_data = [ts.ckks_vector(cntx, datum.tolist()) for datum in data]
    return encrypted_data


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


# print(lr.weights)

print("Automatic relinearization is:", ("on" if cntx.auto_relin else "off"))
print("Automatic rescaling is:", ("on" if cntx.auto_rescale else "off"))
print("Automatic modulus switching is:", ("on" if cntx.auto_mod_switch else "off"))



# plain_input = np.array([                           # Please input normalized and preprocessed data!!
#     [0.5, 1.2, 0.3],                        # Random example nums
#     [0.1, 0.4, 0.5]                         # random example nums
# ])

matches = 0

for i in range(114):

    plain_input = x_scaled_test.iloc[i]
    plain_input = plain_input.to_numpy()

    encrypted_input = ts.ckks_vector(cntx, plain_input)

    x_dot_weight = encrypted_input.dot(lr.weights.transpose()) + lr.bias
    enc_probability = x_dot_weight.polyval(SIGMOID_POLY)

    dec_probability = enc_probability.decrypt()[0]
    plain_probability = lr.predictSingle(x_scaled_test.iloc[i])

    dec_true_false = 1 if dec_probability > 0.5 else 0
    plain_true_false = 1 if plain_probability > 0.5 else 0

    print("TenSEAL probability:", dec_probability)
    print("plantext probability:", plain_probability)

    print("TenSEAL method:  ", dec_true_false)
    print("plaintext method:", plain_true_false)

    if dec_true_false == plain_true_false:
        matches += 1

print(f"Total matches: {matches}/114")
print(f"About {(matches/114) * 100}% similar")









