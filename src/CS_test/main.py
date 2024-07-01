from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from logistic_regression import LogisticRegression as CustomLogisticRegression
# from data import x_train, y_train
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

if not os.path.exists(MODEL_DIRECTORY):
    os.mkdir(MODEL_DIRECTORY)

# if not os.path.exists(MODEL_PATH):
if True:
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


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDRESS_TUPLE)
server.listen()

print(f"Server listening on {SERVER_IP}:{PORT}")

conn, addr = server.accept()
print(f"Connected by {addr}")

def recv_all(conn, size):
    """Helper function to receive exactly 'size' bytes from 'conn'."""
    data = b''
    while len(data) < size:
        print(f"Attempting to receive {size - len(data)} more bytes")
        packet = conn.recv(size - len(data))
        if not packet:
            # Connection closed by the client
            print("Connection closed by the client")
            return None
        data += packet
        print(f"Received {len(packet)} bytes, total received: {len(data)} bytes")
    if size > len(data):
        print(f"missing {size - len(data)} bytes")
    else:
        print("obtained all bytes")
    return data


# Receive the size of the incoming DataFrame
xMessageSize = conn.recv(4096)
xClientByteSize = pickle.loads(xMessageSize)
print(f"Expecting {xClientByteSize} bytes for the DataFrame")
conn.send("received x byte size".encode(FORMAT))

# Ensure we receive the complete DataFrame data
xMessage = recv_all(conn, xClientByteSize)
if xMessage is not None:
    xClient = pickle.loads(xMessage)
    print(f"Received DataFrame")
    conn.send("Received DataFrame".encode(FORMAT))
else:
    print("Failed to receive the complete DataFrame")
    conn.send("Failed to receive the complete DataFrame".encode(FORMAT))



# Receive the size of the incoming Series
yMessageSize = conn.recv(4096)
yClientByteSize = pickle.loads(yMessageSize)
print(f"Expecting {yClientByteSize} bytes for the Series")
conn.send("received y byte size".encode(FORMAT))

# Ensure we receive the complete Series data
yMessage = recv_all(conn, yClientByteSize)
if yMessage is not None:
    yClient = pickle.loads(yMessage)
    print(f"Received Series")
else:
    print("Failed to receive the complete Series")




pred = lr.predict(xClient)
accuracy = accuracy_score(yClient, pred)
customAccuracyMessage = "Custom made Regression Accuracy: " + str(accuracy)
print(customAccuracyMessage)

conn.send(customAccuracyMessage.encode(FORMAT))
clientConfirmation = conn.recv(4096).decode()
print(clientConfirmation)

model = LogisticRegression(solver='newton-cg', max_iter=150)
model.fit(x_scaled_train, y_train)
pred2 = model.predict(xClient)
accuracy2 = accuracy_score(yClient, pred2)
controlAccuracyMessage = "Built-in Regression Accuracy: " + str(accuracy2)
print(controlAccuracyMessage)

conn.send(controlAccuracyMessage.encode(FORMAT))


conn.close()



"hello world"

"good mornning"


