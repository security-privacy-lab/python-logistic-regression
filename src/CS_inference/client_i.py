from data import x_train, x_test, y_train, y_test
import pickle
import socket


PORT = 5050
SERVER_IP = socket.gethostbyname(socket.gethostname())
ADDRESS_TUPLE = (SERVER_IP, PORT)
FORMAT = "utf-8"


client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDRESS_TUPLE)



for i in range(10):
    x_message = x_test.iloc[i]
    x_pickled = pickle.dumps(x_message)

    client.send(x_pickled)

    y_pickled = client.recv(4096)
    y_predicted = pickle.loads(y_pickled)

    print(f"Prediction: {y_predicted}\tAnswer: {y_test.iloc[i]}", "\t",
          "CORRECT" if y_predicted == y_test.iloc[i] else "WRONG")


