import pickle
import socket
from models.BreastCancer import lr as custom_regression
from data.constants import *


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDRESS_TUPLE)
server.listen()

print(f"Server listening on {SERVER_IP}:{PORT}")

conn, addr = server.accept()
print(f"Connected by {addr}")


for i in range(10):
    x_message = conn.recv(4096)
    x_series = pickle.loads(x_message)

    prediction = custom_regression.predictSingle(x_series)
    encrypted_prediction = pickle.dumps(prediction)
    conn.send(encrypted_prediction)
