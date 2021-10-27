from torch import nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# -- Load data

df = pd.read_csv("dataset/student.csv")

data = df.to_numpy()
M = data.shape[0]

x = data[:, :2].reshape(M, 2)
y = data[:, -1].reshape(M, 1)

x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
y = torch.tensor(y, dtype=torch.float32, requires_grad=False)


# -----------------------

class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, 2)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(2, 2)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(2, 1)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.act1(z1)

        z2 = self.fc2(a1)
        a2 = self.act2(z2)

        z3 = self.fc3(a2)
        out = self.act3(z3)

        return out


# -- Init model
model = Net(input_size=2)

lr = 0.01

loss_func = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

epoch = 1000

losses = []

for ep in range(epoch):

    # -- forward
    y_hat = model.forward(x)

    # -- loss
    loss = loss_func(y_hat, y)
    losses.append(loss.item())

    # -- backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if ep % 10 == 0:
        print(f"Epoch {ep} loss : {round(loss.item(), 2)}")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.plot(losses)
plt.show()