from data import x_scaled_test, y_test
import pickle
import socket
import tenseal as ts
import numpy as np
import time

PORT = 5050
SERVER_IP = socket.gethostbyname(socket.gethostname())
ADDRESS_TUPLE = (SERVER_IP, PORT)
FORMAT = "utf-8"


# Step 1: Create a TenSeal Context
cntx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192 * 2,  # MUST be 8192 * 2^x, where x is a non-negative integer
    # coeff_mod_bit_sizes = [60,40,40,60]
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60]
)
cntx.global_scale = 2 ** 40
cntx.generate_galois_keys()

server_cntx = cntx.copy()
server_cntx.make_context_public()

def send_all(client, serialized_object):
    object_byte_length = len(serialized_object)
    print(f"object_byte_length: {object_byte_length}")
    client.send(pickle.dumps(object_byte_length))

    server_confirmation = client.recv(4096).decode(FORMAT)
    print(f"[SERVER MESSAGE]: {server_confirmation}")

    client.send(serialized_object)

    server_confirmation2 = client.recv(4096).decode(FORMAT)
    print(f"[SERVER MESSAGE]: {server_confirmation2}")

def recv_all(client, object_name, dataType):
    client.send(f"Ready to receive {object_name}'s byte size...".encode(FORMAT))
    pickled_object_length = pickle.loads(client.recv(4096))

    client.send(f"Received {object_name}'s byte size. Ready to receive {object_name}".encode(FORMAT))
    serialized_object = client.recv(pickled_object_length)

    if dataType == "vector":
        object = ts.ckks_vector_from(cntx, serialized_object)
    elif dataType == "pickle":
        object = pickle.loads(serialized_object)
    else:
        return "BIG ERROR"

    return object




client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDRESS_TUPLE)

send_all(client, server_cntx.serialize())

df_length = len(x_scaled_test.index)
send_all(client, pickle.dumps(df_length))


for i in range(df_length):
    current_row = x_scaled_test.iloc[i]
    encrypted_row = ts.ckks_vector(cntx, current_row)
    serialized_row = encrypted_row.serialize()
    print(f"row {i + 1}'s byte size is {len(serialized_row)}")
    send_all(client, serialized_row)

    print(f"row {i + 1} sent")

    # time.sleep(0.05)



send_all(client, pickle.dumps(x_scaled_test))
# send_all(client, pickle.dumps(y_test))

enc_output = []
for i in range(df_length):
    encrypted_row = recv_all(client, f"row {i + 1}", "vector")
    decrypted_row = encrypted_row.decrypt()[0]
    enc_output.append(1 if decrypted_row > .5 else 0)
plain_output = recv_all(client, "unencrypted custom model", "pickle")
built_in_model = recv_all(client, "built-in model", "pickle")

client.close()

for i in range(df_length):
    print(f"ROW {i + 1:3}:\tenc: {enc_output[i]}\tplain: {plain_output[i]}\tbuilt-in: {built_in_model[i]}\t ACTUAL: {y_test.iloc[i]}")

