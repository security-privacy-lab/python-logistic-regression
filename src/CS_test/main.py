from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from data.breastCancer import x_scaled_train, y_train
from data.constants import *
from models.BreastCancer import lr as custom_regression
import pickle
import socket

# PORT = 5050
# SERVER_IP = socket.gethostbyname(socket.gethostname())
# ADDRESS_TUPLE = (SERVER_IP, PORT)
# FORMAT = "utf-8"

# creates the server socket object, using TCP over IPv4.
# bind the server to the port set in data.constants
# opens the server to accept connections
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDRESS_TUPLE)
server.listen()
print(f"Server listening on {SERVER_IP}:{PORT}")

# halts the program until a client socket connects to the server
# pastes the client's info afterwards
conn, addr = server.accept()
print(f"Connected by {addr}")

def recv_all(conn, size):
    # """Helper function to receive exactly 'size' bytes from 'conn'."""
    """
    Helper function to receive exactly 'size' bytes from 'conn'.

    Repeats a while loop until received bytes matches inputted 'size'

    Args:
        conn (socket): socket connection
        size (int): number of bytes to receive

    Returns:
        bytes: received data
    """
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

# The client will send over the testing dataset in the form of a DataFrame.
# First, the client will send over the byte size of the DataFrame, so that the server can have a reasonable value to
#   pass to conn.recv() within recv_all()
#  The alternative is having an arbitrarily large number of bytes to receive, instead of a specific amount set by the
#   client. Hard-coding it like this isn't ideal, as it leaves network efficiency on the table.
# Next, the byte size is sent to recv_all()

# Receive the size of the incoming DataFrame, which
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




pred = custom_regression.predict(xClient)
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


