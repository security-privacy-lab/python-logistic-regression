from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from data import x_scaled_train, y_train
from models.BreastCancer import lr as custom_regression
import pickle
import socket
import tenseal as ts

PORT = 5050
SERVER_IP = socket.gethostbyname(socket.gethostname())
ADDRESS_TUPLE = (SERVER_IP, PORT)
FORMAT = "utf-8"


def recv_all(conn, object_name, dataType):
    ser_pickled_object_length = conn.recv(4096)
    pickled_object_length = pickle.loads(ser_pickled_object_length)
    server_message = f"Received byte size of {pickled_object_length}. length of {len(ser_pickled_object_length)}"
    print(server_message)
    conn.send(server_message.encode(FORMAT))

    serialized_object = conn.recv(pickled_object_length)
    server_message2 = f"Received {object_name} of size {len(serialized_object)}."
    print(server_message2)
    conn.send(server_message2.encode(FORMAT))

    if dataType == "vector":
        object = ts.ckks_vector_from(cntx, serialized_object)
    elif dataType == "context":
        object = ts.context_from(serialized_object)
    elif dataType == "pickle":
        object = pickle.loads(serialized_object)
    else:
        return "BIG ERROR"

    return object

def send_all(conn, serialized_object):
    client_signal = conn.recv(4096).decode(FORMAT)
    print(client_signal)


    object_byte_length = len(serialized_object)
    conn.send(pickle.dumps(object_byte_length))


    client_signal2 = conn.recv(4096).decode(FORMAT)
    print(client_signal2)

    conn.send(serialized_object)


model = LogisticRegression(solver='newton-cg', max_iter=150)
model.fit(x_scaled_train, y_train)


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDRESS_TUPLE)
server.listen()
print(f"Server listening on {SERVER_IP}:{PORT}")

conn, addr = server.accept()
print(f"Connected by {addr}")


cntx = recv_all(conn, "context", "context")

encrypted_x_test = []
x_test_size = recv_all(conn, "x_test size", "pickle")
print(x_test_size)

for i in range(x_test_size):
    current_row = recv_all(conn, f"row {i + 1}", "vector")
    encrypted_x_test.append(current_row)


x_scaled_test = recv_all(conn, "x_scaled_test", "pickle")
# y_test = recv_all(conn, "y_test", False)


enc_cust_probabilities = custom_regression.predictEncrypted(encrypted_x_test)
custom_predictions = custom_regression.predict(x_scaled_test)
builtin_predictions = model.predict(x_scaled_test)

serialized_enc_cust_prob = [(i.serialize()) for i in enc_cust_probabilities]


for i in range(x_test_size):
    send_all(conn, serialized_enc_cust_prob[i])
send_all(conn, pickle.dumps(custom_predictions))
send_all(conn, pickle.dumps(builtin_predictions))

conn.close()
