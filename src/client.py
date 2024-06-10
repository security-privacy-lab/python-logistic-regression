import time

from data import x_train, x_test, y_train, y_test
# x_test = {'a': 1, 'b': 2}
# y_test = [1, 2, 3, 4]
import pickle
import socket


PORT = 5050
SERVER_IP = socket.gethostbyname(socket.gethostname())
ADDRESS_TUPLE = (SERVER_IP, PORT)
FORMAT = "utf-8"




xPickle = pickle.dumps(x_test)
xByteSize = len(xPickle)
print(xByteSize)

yPickle = pickle.dumps(y_test)
yByteSize = len(yPickle)
print(yByteSize)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDRESS_TUPLE)



# Send the size of the pickled DataFrame
client.send(pickle.dumps(xByteSize))
print(f"Sent size: {xByteSize} bytes")

print(client.recv(4096).decode(FORMAT))

# Send the actual pickled DataFrame
client.sendall(xPickle)
print(f"Sent DataFrame")

print(client.recv(4096).decode(FORMAT))

# Send the size of the pickled Series
client.send(pickle.dumps(yByteSize))
print(f"Sent size: {yByteSize} bytes")

print(client.recv(4096).decode(FORMAT))

# Send the actual pickled Series
client.sendall(yPickle)
print(f"Sent Series")

# Ensure the data is fully sent before closing the connection
time.sleep(1)


print(client.recv(4096).decode(FORMAT))
client.send("Message received.".encode(FORMAT))

print(client.recv(4096).decode(FORMAT))

client.close()
print("Client connection closed")
