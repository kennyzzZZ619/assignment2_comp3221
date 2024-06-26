import pickle
import random
import socket
import threading
import time
import torch
import sys
import numpy as np
from LinearRegression import LinearRegressionModel


class FLServer:
    def __init__(self, port_s, num_clients, subsample_s, seed=False):
        self.server_socket = None
        self.s = None
        self.server_port = port_s  # the port number of server
        self.num_clients = num_clients  # we have 5 client
        self.subsample = subsample_s
        if seed:
            self.global_model = LinearRegressionModel(seed=True)  # actually we have to initial a model
        elif not seed:
            self.global_model = LinearRegressionModel()
        self.client_models = {}
        self.client_data = {}
        self.lock = threading.Lock()
        self.wait_time = 30
        self.global_round = 10
        self.connections = []
        #self.all_connections = []

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('localhost', self.server_port))
        self.server_socket.listen(self.num_clients)
        print(f"Server started on port {self.server_port}. Waiting for clients...")
        self.accept_clients()
        print("Broadcast the initial model to client...")
        self.broadcast_model()

    def global_iteration(self):
        for t in range(self.global_round):
            print(f"Global Iteration {t + 1}:")
            self.handle_models()
            self.broadcast_model()

    def accept_clients(self):
        start_time = time.time()
        while len(self.connections) < self.num_clients:
            conn, addr = self.server_socket.accept()
            print(f"Connected to client at {addr}")
            self.connections.append(conn)
            if len(self.connections) == 1:
                time_remaining = self.wait_time - (time.time() - start_time)
                if time_remaining > 0:
                    print(f"Waiting for additional clients to connect for {time_remaining} seconds.")
                    time.sleep(time_remaining)
            threading.Thread(target=self.handle_handshake, args=(conn,)).start()

    def handle_handshake(self, conn):
        try:
            handshake_data = conn.recv(40960)
            if handshake_data:
                handshake_msg = pickle.loads(handshake_data)
                client_id = handshake_msg.get('client_id')
                data_size = handshake_msg.get('data_size')
                print(f"Handshake received from {client_id} with data size {data_size}.")
                self.client_data[client_id] = {'train_samples': data_size}
        except Exception as e:
            print(f"Failed to receive or process handshake data: {e}")

    def handle_models(self):
        self.client_models.clear()
        #selected_connections = self.select_random_clients()
        for conn in self.connections:
            try:
                model_data = conn.recv(40960)
                if model_data:
                    self.process_received_data(pickle.loads(model_data))
                    #time.sleep(2)
            except socket.error as e:
                print(f"Error receiving data: {e}")
        #if self.subsample == len(self.client_models):
        #    self.global_model = self.aggregate_models()
        if len(self.client_models) == self.num_clients:
            self.global_model = self.aggregate_models()

    def select_random_clients(self):
        if self.subsample == 0 or self.subsample >= len(self.connections):
            return self.connections  
        else:

            return random.sample(self.connections, self.subsample)

    def process_received_data(self, data_packet):  # conn
        # decode the data pack, 'local_model' is form client model
        client_id = data_packet.get('client_id')
        print(f"Getting local model from client {client_id}")
        local_model = data_packet.get('model')
        with self.lock:
            # self.lock.acquire()

            self.client_models[client_id] = local_model

    # def aggregate_models(self):
    #     # Initialize a dictionary to store the accumulated weights
    #     total_weights = {}
    #
    #     # Initialize total_weights with zeros
    #     for key, param in self.global_model.state_dict().items():
    #         total_weights[key] = torch.zeros_like(param)
    #
    #     # Compute the total number of training samples from all clients
    #     total_train_samples = sum(client_data['train_samples'] for client_data in self.client_data.values())
    #
    #     # Sum the weighted parameters of each client's model
    #     for client_id, user_weights in self.client_models.items():
    #         client_samples = self.client_data[client_id]['train_samples']
    #         for key, param in user_weights.items():
    #             weight = param * client_samples / total_train_samples
    #             total_weights[key] += weight
    #
    #     # Load the aggregated weights into the global model
    #     self.global_model.load_state_dict(total_weights)
    #     return self.global_model
    def aggregate_models(self):
        # Initialize a dictionary to store the accumulated weights
        total_weights = {}

        # Initialize total_weights with zeros
        for key, param in self.global_model.state_dict().items():
            total_weights[key] = torch.zeros_like(param)

        # Determine the client models to use based on subsampling
        if 0 < self.subsample < self.num_clients:
            # If subsampling is enabled, randomly pick a subset of clients
            selected_client_ids = random.sample(list(self.client_models.keys()), self.subsample)
            selected_models = {client_id: self.client_models[client_id] for client_id in selected_client_ids}
        else:
            # Use all client models if subsampling is not enabled
            selected_models = self.client_models

        # Compute the total number of training samples from the selected clients
        total_train_samples = sum(self.client_data[client_id]['train_samples'] for client_id in selected_models)

        # Sum the weighted parameters of each selected client's model
        for client_id, user_weights in selected_models.items():
            client_samples = self.client_data[client_id]['train_samples']
            for key, param in user_weights.items():
                weight = param * client_samples / total_train_samples
                total_weights[key] += weight

        # Load the aggregated weights into the global model
        self.global_model.load_state_dict(total_weights)
        return self.global_model

    def broadcast_model(self):
        model_state_dict = self.global_model.state_dict()
        for name, param in self.global_model.named_parameters():
            print(f"Parameter name: {name}")
            print(f"Shape: {param.size()}")
            print(f"Type: {param.dtype}")
            print("Values:\n", param.data)
        global_model_data = pickle.dumps(model_state_dict)
        for conn in self.connections:
            conn.sendall(global_model_data)


if __name__ == "__main__":

    if len(sys.argv) == 3:
        port = int(sys.argv[1])
        subsample = int(sys.argv[2])
        # We should add a function to check the argument can be use
        server = FLServer(port, num_clients=5, subsample_s=subsample)
        server.start_server()
        server.global_iteration()
    elif len(sys.argv) != 3:
        if len(sys.argv) == 4 and sys.argv[3] == 'demo':
            random.seed(10)
            port = int(sys.argv[1])
            subsample = int(sys.argv[2])
            # We should add a function to check the argument can be use
            server = FLServer(port, num_clients=5, subsample_s=subsample, seed=True)
            server.start_server()
            server.global_iteration()
        else:
            print("Usage: python COMP3221_FLServer.py <Port-Server> <Sub-Client>")
            sys.exit(1)