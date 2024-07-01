# from sklearn.metrics import accuracy_score                # implement storage of sklearn trained model
# from sklearn.linear_model import LogisticRegression
from logistic_regression import LogisticRegression as CustomLogisticRegression
from data import x_train, y_train
from data import x_scaled_train, y_train
import os
import pickle
import socket


MODEL_DIRECTORY = '../../models'
MODEL_PATH = '../../models/Custom_Logistic_Regression2.pkl'

PORT = 5050
SERVER_IP = socket.gethostbyname(socket.gethostname())
ADDRESS_TUPLE = (SERVER_IP, PORT)
FORMAT = "utf-8"





print(os.path.exists(MODEL_DIRECTORY))

if not os.path.exists(MODEL_PATH):
# if True:
    print(f"[MODEL NOT FOUND] No model at {MODEL_PATH}")
    lr = CustomLogisticRegression()
    lr.fit(x_train, y_train, epochs=150)

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


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDRESS_TUPLE)
server.listen()

print(f"Server listening on {SERVER_IP}:{PORT}")

conn, addr = server.accept()
print(f"Connected by {addr}")






for i in range(10):
    x_message = conn.recv(4096)
    x_series = pickle.loads(x_message)

    prediction = lr.predictSingle(x_series)
    encrypted_prediction = pickle.dumps(prediction)
    conn.send(encrypted_prediction)


