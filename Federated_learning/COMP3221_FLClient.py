import pickle
import socket
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
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
    def __init__(self, client_id, client_port, opt_method, seed=False):
        self.connection = None
        self.client_id = client_id
        self.client_port = client_port
        self.opt_method = opt_method
        if seed:
            self.model = LinearRegressionModel(seed=True)
        elif not seed:
            self.model = LinearRegressionModel()
        #self.model = None
        self.loss = nn.MSELoss()
        # self.model = copy.deepcopy(model)
        self.server_host = 'localhost'
        self.server_port = 6000
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.numeric_id = self.client_id.replace("client", "")

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

        self.train_losses = []
        self.test_losses = []

    def connect_to_server(self):
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.settimeout(10)
        try:
            self.connection.connect((self.server_host, self.server_port))
            print(f"Connected to server at ({self.server_host}, {self.server_port})")
        except (ConnectionRefusedError, socket.timeout) as e:
            print(f"Connection failed: {e}")
            sys.exit(1)

    def register_with_server(self):
        handshake_msg = {'data_size': len(self.X_train), 'client_id': self.numeric_id}
        self.connection.sendall(pickle.dumps(handshake_msg))
        print("Handshake message sent to server.")

    def maintain_connection(self):
        ft = True
        try:
            while True:  # This loop will continue until the server closes the connection or an error occurs
                #print("Waiting 20 seconds for model")
                if ft:
                    ft = False
                    time.sleep(20)
                else:
                    time.sleep(10)
                global_model_data = self.connection.recv(4096)
                if not global_model_data:
                    print("No more data from server.")
                    break

                global_model = pickle.loads(global_model_data)
                # for param_tensor in global_model.state_dict():
                #     print(param_tensor, "\t", global_model.state_dict()[param_tensor])
                #if isinstance(global_model, dict):
                #self.model.load_state_dict(global_model)
                #else:
                #raise ValueError("Received data is not a state_dict.")
                self.model.load_state_dict(global_model)
                print(f"I am client {self.client_id}")
                print("Received new global model")
                test_mse = self.evaluate_model()

                print(f"Testing MSE: {test_mse}")
                # Training the global model in local training
                print("Local training...")
                train_mse = self.train_model(50)
                print(f"Training MSE: {train_mse}")
                self.log_results(train_mse, test_mse)
                # Logic to update and evaluate the model goes here (omitted for brevity)
                updated_model_data = pickle.dumps({'client_id': self.numeric_id, 'model': self.model.state_dict()})

                self.connection.sendall(updated_model_data)
                print("Sending new local model")

                self.train_losses.append(test_mse.detach().numpy())
                self.test_losses.append(train_mse.detach().numpy())
                #print("self.trainlosses: {}".format(self.train_losses))
                #print("self.testlosses: {}".format(self.test_losses))

        except Exception as e:
            print(f"Error during model receive/send: {e}")

    def evaluate_model(self):
        # ... Evaluate model，return test MSE ...
        self.model.eval()

        mse = 0
        for x, y in self.test_loader:
            y_pred = self.model(x)
            # Calculate evaluation metrics

            mse += self.loss(y_pred, y)
            # print(str(self.id) + ", MSE of client ",self.id, " is: ", mse)

        return mse

    def train_model(self, epochs):
        # ... Training model，return train MSE ...
        loss = 0
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        return loss.data

    def plot(self):
        #print("Plotted")
        count = list(range(len(self.train_losses)))  # Assuming both lists are of the same length

        # Create a plot
        plt.figure(figsize=(10, 5))

        # Plotting both feature lists using scatter
        plt.scatter(count, self.train_losses, s=0.5, label='training losses', color='blue', marker='o')
        plt.scatter(count, self.test_losses, s=0.5, label='testing losses', color='red', marker='o')
        for i, value in enumerate(self.train_losses):
            plt.text(count[i], self.train_losses[i], '{0:.2f}'.format(value), color='blue', fontsize='small', ha='right', va='bottom')
        for i, value in enumerate(self.test_losses):
            plt.text(count[i], self.test_losses[i], '{0:.2f}'.format(value), color='red', fontsize='small', ha='left', va='top')
        # Adding titles and labels
        plt.title('Convergence of losses across training iterations for {}'.format(self.client_id))
        plt.xlabel('Iterations')
        plt.ylabel('Losses')
        plt.legend()
        plt.savefig('{}.png'.format(self.client_id))


        plt.close()
    def close_connection(self):
        if self.connection:
            self.plot()
            self.connection.close()
            print("Connection closed.")

    def log_results(self, train_mse, test_mse):
        # The log to record all the MSE detail
        with open(f"{self.client_id}_log.txt", "a") as log_file:
            log_file.write(f"Train MSE: {train_mse}, Test MSE: {test_mse}\n")


if __name__ == "__main__":
    #np.random.seed(0)
    #torch.manual_seed(0)

    if len(sys.argv) == 4:
        client_id = str(sys.argv[1])
        server_port = int(sys.argv[2])
        opt_method = int(sys.argv[3])

        client = FLClient(client_id, server_port, opt_method)
        client.connect_to_server()
        client.register_with_server()
        client.maintain_connection()
        client.close_connection()
    elif len(sys.argv) != 4:
        if len(sys.argv) == 5 and sys.argv[4] == 'demo':
            client_id = str(sys.argv[1])
            server_port = int(sys.argv[2])
            opt_method = int(sys.argv[3])

            client = FLClient(client_id, server_port, opt_method, seed=True)
            client.connect_to_server()
            client.register_with_server()
            client.maintain_connection()
            client.close_connection()
        else:
            print("Usage: python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>")
            sys.exit(1)

