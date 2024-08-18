from data.breastCancer import x_scaled_test, y_test
from data.constants import *
import pickle
import socket


client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDRESS_TUPLE)


for i in range(10):
    x_message = x_scaled_test.iloc[i]
    x_pickled = pickle.dumps(x_message)

    client.send(x_pickled)

    y_pickled = client.recv(4096)
    y_predicted = pickle.loads(y_pickled)

    print(f"Prediction: {y_predicted}\tAnswer: {y_test.iloc[i]}", "\t",
          "CORRECT" if y_predicted == y_test.iloc[i] else "WRONG")
