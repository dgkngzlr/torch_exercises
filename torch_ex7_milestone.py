import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix
import time
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):

    def __init__(self, input_size):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.act1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):

        z1 = self.fc1(x)
        a1 = self.act1(z1)

        out = self.fc2(a1)

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

# Prepare regression data
X, y = datasets.make_regression(n_samples = 1000, n_features = 8, noise=10, random_state=1)

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Dataloader instantiate
batch_size = 128
dataset = ExampleDataset(X_train, y_train)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Get first 32 batch from train data
dataiter = iter(dataloader)
batch_feature, batch_label = next(dataiter)
print(50 * "=")
print("First batch feature size and label size :")
print(batch_feature.shape, batch_label.shape)
print(50 * "=")

# Init model
model = Net(8)
model.to(device)
model.train()

# Loss
criterion = nn.MSELoss()

#Optim
lr = 0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

n_epoch = 10000
n_iter = len(dataset) // batch_size
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

torch.save(model.state_dict(), "./model.pth")
model.eval()

# -- Print loss graph
plt.xlabel("Step")
plt.ylabel("Loss")
plt.plot(losses)
plt.show()

y_hat = model.forward(torch.tensor(X_test, dtype=torch.float32, device=device))
y_hat = y_hat.to("cpu").detach().numpy()

print(y_hat.shape)
print(y_test.shape)
print(y_hat[:5,:])
print(y_test.reshape(-1,1)[:5,:])
mae = (abs(y_test.reshape(-1,1) - y_hat)).mean()
print("MAE:", mae)
