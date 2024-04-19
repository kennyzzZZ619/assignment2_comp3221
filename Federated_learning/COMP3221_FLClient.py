import pickle
import socket
import sys
import os

import pandas as pd
from torch.utils.data import DataLoader

from Federated_learning.LinearRegression import LinearRegressionModel


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
    def __init__(self, client_id, server_port, opt_method):
        self.client_id = client_id
        self.server_port = server_port

        self.model = LinearRegressionModel()

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = load_dataset(
            self.client_id)

        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]

        batch_size = self.train_samples if opt_method == 0 else 64

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_data, batch_size=self.test_samples)

    def register_from_server(self):
        handshake_msg = {
            'data_size': len(self.X_train),
            'client_id': self.client_id
        }
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as Client_c:
            Client_c.connect(('localhost', self.server_port))
            Client_c.sendall(pickle.dumps(handshake_msg))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>")
        sys.exit(1)

    client_id = str(sys.argv[1])
    server_port = int(sys.argv[2])
    opt_method = int(sys.argv[3])

    client = FLClient(client_id, server_port, opt_method)
