import numpy
from torch import nn
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def get_bin_coded(df, _dict):
    result = df.replace(_dict, inplace=False)

    return result


def data_preprocess(data):
    rec = pd.get_dummies(data.iloc[:, 0]).to_numpy()
    age = pd.get_dummies(data.iloc[:, 1]).to_numpy()
    meno = pd.get_dummies(data.iloc[:, 2]).to_numpy()
    tumor_size = pd.get_dummies(data.iloc[:, 3]).to_numpy()
    deg_malig = data.iloc[:, 6].to_numpy().reshape(-1,1)
    breast = pd.get_dummies(data.iloc[:, 7]).to_numpy()

    x = numpy.hstack((rec, age, meno, tumor_size,deg_malig, breast))
    y = get_bin_coded(data.iloc[:, 9], {"no": 0, "yes": 1}).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=13)

    return x_train, x_test, y_train, y_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- Load data
df = pd.read_csv("dataset/breast-cancer.csv")

x_train, x_test, y_train, y_test = data_preprocess(df)

M = df.shape[0]

x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=False, device=device).view(-1,1)
# -----------------------

# -- Arch
class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(32, 16)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(16, 1)
        self.act4 = nn.Sigmoid()

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.act1(z1)

        z2 = self.fc2(a1)
        a2 = self.act2(z2)

        z3 = self.fc3(a2)
        a3 = self.act3(z3)

        z4 = self.fc4(a3)
        out = self.act4(z4)

        return out


# -- Init model
model = Net(input_size=25)
model.to(device)

model.train()

# -- Show layer[0] weights before training
print("Layer[0] weights before training :")
for name, para in model.named_parameters():
    print('{}: {}'.format(name, para.shape))
    print(para)
    break
# -- Print arch
print(50 * "=")
print(model)
print(50 * "=")

# learning rate
lr = 0.01

# loss function
loss_func = nn.BCELoss()

# Optimizer type
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# num of epochs
epoch = 5000

losses = []

for ep in range(epoch):

    # -- forward
    y_hat = model.forward(x_train)

    # -- loss
    loss = loss_func(y_hat, y_train)
    losses.append(loss.item())

    # -- backward
    optimizer.zero_grad()
    loss.backward()

    # -- update weights
    optimizer.step()

    if (ep + 1) % 1000 == 0:
        print(f"Epoch {ep+1} loss : {round(loss.item(), 2)}")

# -- Test mode
model.eval()

# -- After training layer[0] weights
print("Layer[0] weights after training :")
for name, para in model.named_parameters():
    print('{}: {}'.format(name, para.shape))
    print(para.max(),para.min())
    break

# -- Print loss graph
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(losses)
plt.show()

# -- Get prediction for test set
y_prd = model.forward(torch.tensor(x_test, dtype=torch.float32, device=device))
y_prd = (y_prd > 0.5).int().to("cpu").detach().numpy()

# -- Confusion Matrix
cm = confusion_matrix(y_test, y_prd)
print("Confusion Matrix :",cm, sep="\n")

# -- Get score for test set
score = (np.eye(2) * cm).sum() / cm.sum()
print("Score (%):", score*100)
