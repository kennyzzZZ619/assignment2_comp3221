import pickle
import socket
import sys
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from LinearRegression import LinearRegressionModel


def load_dataset(client_id):
    base_path = os.path.dirname(__file__)  # Gets the directory in which the script is located
    train_filename = os.path.join(base_path, "FLData", f"calhousing_train_{client_id}.csv")
    test_filename = os.path.join(base_path, "FLData", f"calhousing_test_{client_id}.csv")
    train_data = pd.read_csv(train_filename)
    test_data = pd.read_csv(test_filename)

    X_train = train_data.iloc[:, :-1].values  # Select all columns except the last one
    y_train = train_data.iloc[:, -1].values  # Select the last column as the target

    X_test = test_data.iloc[:, :-1].values  # Select all columns except the last one
    y_test = test_data.iloc[:, -1].values  # Select the last column as the target

    train_samples, test_samples = len(y_train), len(y_test)

    return X_train, y_train, X_test, y_test, train_samples, test_samples


class FLClient:
    def __init__(self, client_id, client_port, opt_method):
        self.client_id = client_id
        self.client_port = client_port
        self.opt_method = opt_method
        self.model = LinearRegressionModel()
        # self.model = copy.deepcopy(model)

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = load_dataset(
            self.client_id)

        # self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        # self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)

        # Create TensorDataset objects
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        batch_size = self.train_samples if self.opt_method == 0 else 64

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.test_samples)

    def register_from_server(self):
        handshake_msg = {
            'data_size': len(self.X_train),
            'client_id': self.client_id
        }
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as Client_c:
            Client_c.connect(('localhost', 6000))
            Client_c.sendall(pickle.dumps(handshake_msg))

    def handle_model(self):
        # receive the global model from server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('localhost', 6000))
            global_model_data = s.recv(2048)
            global_model = pickle.loads(global_model_data)
            self.model.load_state_dict(global_model)
            print(f"I am client {self.client_id}")
            print("Received new global model")
            # Evaluate the global model using local test data
            test_mse = self.evaluate_model()
            print(f"Testing MSE: {test_mse}")
            # Training the global model in local training
            print("Local training...")
            train_mse = self.train_model()
            print(f"Training MSE: {train_mse}")
            # Send the model to server
            updated_model_data = pickle.dumps(self.model.state_dict())
            s.sendall(updated_model_data)
            print("Sending new local model")

    def evaluate_model(self):
        # ... evaluate model，return test MSE ...
        pass

    def train_model(self):
        # ... training model，return train MSE ...
        pass

    def log_results(self, train_mse, test_mse):
        # The log to record all the MSE detail
        with open(f"{self.client_id}_log.txt", "a") as log_file:
            log_file.write(f"Train MSE: {train_mse}, Test MSE: {test_mse}\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>")
        sys.exit(1)

    client_id = str(sys.argv[1])
    server_port = int(sys.argv[2])
    opt_method = int(sys.argv[3])

    client = FLClient(client_id, server_port, opt_method)
    client.register_from_server()
    while True:
        client.handle_model()

