import socket

PORT = 5050
SERVER_IP = socket.gethostbyname(socket.gethostname())
ADDRESS_TUPLE = (SERVER_IP, PORT)
FORMAT = "utf-8"

