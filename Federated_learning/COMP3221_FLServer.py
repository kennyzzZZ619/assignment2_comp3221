import json
import pickle
import socket
import threading
import time
import torch
import sys

from LinearRegression import LinearRegressionModel


class FLServer:
    def __init__(self, port_s, num_clients, subsample_s):
        self.s = None
        self.server_port = port_s  # the port number of server
        self.num_clients = num_clients  # we have 5 client
        self.subsample = subsample_s
        self.global_model = LinearRegressionModel()  # actually we have to initial a model
        self.client_models = {}
        self.client_data = {}
        self.lock = threading.Lock()
        self.wait_time = 30
        self.global_round = 10

    def start_server(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(('localhost', self.server_port))
        self.s.listen(self.num_clients)
        print(f"Server started on port {self.server_port}. Waiting for clients...")
        # start global communication
        for t in range(self.global_round):
            print(f"Global Iteration {t + 1}:")
            # for param_tensor in self.global_model.state_dict():
            #     print(param_tensor, "\t", self.global_model.state_dict()[param_tensor])
            # try to do the model fitting and so on
            self.accept_clients(t)
            if t == 0:
                self.broadcast_model()
            # else:
            #     self.aggregate_models()
        # send the ending message
        # self.broadcast_finish_message()
        self.s.close()

    def accept_clients(self, t):
        start_time = time.time()
        connected_clients = 0  # track the number of connections
        print(f"Total Number of clients: {self.num_clients}")
        while connected_clients < self.num_clients:
            # Accept a new connection
            conn, addr = self.s.accept()
            print(f"Getting local model from client {connected_clients + 1}")

            # Increment the counter for connected clients
            connected_clients += 1

            # Start a new thread to handle the client
            client_thread = threading.Thread(target=self.handle_handshake, args=(conn, addr, t))
            client_thread.start()

            # Check if the initial wait time has passed
            if connected_clients == 1:
                time_remaining = self.wait_time - (time.time() - start_time)
                if time_remaining > 0:
                    print(f"Waiting for additional clients to connect for {time_remaining} seconds.")
                    time.sleep(time_remaining)

    def handle_handshake(self, conn, addr, t):
        if t == 0:
            handshake_data = conn.recv(4096)
            if handshake_data:
                handshake_msg = pickle.loads(handshake_data)
                client_id = handshake_msg.get('client_id')
                data_size = handshake_msg.get('data_size')
                print(f"Handshake received from {client_id} with data size {data_size}.")
                self.client_data[client_id] = {'train_samples': data_size}
        else:
            global_model_data = conn.recv(4096)
            if global_model_data:
                global_model = pickle.loads(global_model_data)
                self.process_received_data(global_model)

    def process_received_data(self, data_packet):  # conn
        # decode the data pack, 'local_model' is form client model
        # client_id, local_model = data_packet['client_id'], data_packet['model']
        client_id = data_packet.get('client_id')
        local_model = data_packet.get('model')
        # local_model = data_packet['model']
        with self.lock:
            self.lock.acquire()
            # refresh the model from client
            self.client_models[client_id] = local_model
            self.lock.release()
            # when all the client was sent models
        if len(self.client_models) == self.num_clients:
            # Aggregate the model
            self.global_model = self.aggregate_models()
            # broadcast the model back to client
            self.broadcast_model()

    def aggregate_models(self):
        # Initialize a dictionary to store the accumulated weights
        total_weights = {}

        # Initialize total_weights with zeros
        for key, param in self.global_model.state_dict().items():
            total_weights[key] = torch.zeros_like(param)

        # Compute the total number of training samples from all clients
        total_train_samples = sum(client_data['train_samples'] for client_data in self.client_data.values())

        # Sum the weighted parameters of each client's model
        for client_id, model in self.client_models.items():
            client_samples = self.client_data[client_id]['train_samples']
            user_weights = model.state_dict()
            for key, param in user_weights.items():
                weight = param * client_samples / total_train_samples
                total_weights[key] += weight

        # Load the aggregated weights into the global model
        self.global_model.load_state_dict(total_weights)

        return self.global_model

    def broadcast_model(self):
        print("Broadcasting new global model")
        send_model = json.dumps(self.global_model)
        for client_port in range(6001, 6005, 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as SERVER:
                try:
                    SERVER.connect(('localhost', client_port))
                    SERVER.sendall(send_model.encode('utf-8'))
                except Exception as e:
                    print(f"Can not connect to the client {client_port}, error is {e}")
        print("sender stop")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python COMP3221_FLServer.py <Port-Server> <Sub-Client>")
        sys.exit(1)

    port = int(sys.argv[1])
    subsample = int(sys.argv[2])

    # We should add a function to check the argument can be use
    server = FLServer(port, num_clients=5, subsample_s=subsample)
    server.start_server()
