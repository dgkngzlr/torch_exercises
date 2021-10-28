import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix
import time
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):

    def __init__(self, input_size, n_layer, n_neuron):

        super(Net, self).__init__()

        self.layers = nn.ModuleDict()
        self.n_layer = n_layer
        self.n_neuron = n_neuron

        # input layer
        self.layers['input'] = nn.Linear(input_size,n_neuron)
        
        # hidden layers
        for i in range(n_layer - 2):
            self.layers[f'hidden{i}'] = nn.Linear(n_neuron,n_neuron)

        # output layer
        self.layers['output'] = nn.Linear(n_neuron,1)

    def forward(self, x):

        out = F.relu(self.layers["input"](x))

        for i in range(self.n_layer - 2):
            out = F.relu(self.layers[f"hidden{i}"](out))

        out = self.layers['output'](out)

        return out

class ExampleDataset(Dataset):

    def __init__(self, X, y):

        xy = np.hstack((X, y.reshape(-1, 1)))
        self.x = torch.tensor(xy[:, :8], dtype=torch.float32, device=device)  # torch.from_numpy(xy[:,:2].astype(np.float32))
        self.y = torch.tensor(xy[:, -1], dtype=torch.float32, device=device)  # torch.from_numpy(xy[:,-1].astype(np.int_))
        self.n_samples = xy.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.n_samples

def train_the_model(model: Net, dataset: ExampleDataset,dataloader: DataLoader):
    lr = 0.001
    n_epoch = 5000
    n_iter = len(dataset) // batch_size

    # Loss
    criterion = nn.MSELoss()

    #Optim
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    
    print(50 * "=")
    print(f"Total Epoch : {n_epoch} Iter per Epoch : {n_iter}")
    print(50 * "=")
    losses = []
    summary(model, (1,8))
    for ep in range(n_epoch):

        for i ,(inputs, outputs) in enumerate(dataloader):

            # --forward
            y_hat = model.forward(inputs)

            # --loss
            loss = criterion(y_hat, outputs.view(-1,1))
            losses.append(loss.item())

            # --backward
            optimizer.zero_grad()
            loss.backward()

            # --update
            optimizer.step()

            if (ep + 1 ) % 100 == 0 and (i + 1) % n_iter == 0:

                print(f"Epoch {ep + 1} Step : {i + 1} loss : {loss.item():.3f}")

    model.eval()

    y_hat = model.forward(torch.tensor(X_test, dtype=torch.float32, device=device))
    y_hat = y_hat.to("cpu").detach().numpy()
    print("y_test shape", y_test.shape, "y_hat shape", y_hat.shape)
    mae = (abs(y_test.reshape(-1,1) - y_hat)).mean()

    return model, losses[-1], mae 

# Prepare regression data
X, y = datasets.make_regression(n_samples = 1000, n_features = 8, noise=10, random_state=1)

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Dataloader instantiate
batch_size = 32
dataset = ExampleDataset(X_train, y_train)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Init model
model = Net(input_size=8, n_layer=2, n_neuron=512)
model.to(device)
model.train()

trained_model, loss, mae = train_the_model(model, dataset, dataloader)

print("LOSS:", loss, "MAE:", mae)





