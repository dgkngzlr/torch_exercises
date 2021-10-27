import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"

iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species")
plt.show()

# organize the data
# convert from pandas dataframe to tensor
# sepal_length, sepal_width, petal_length, petal_width
data = torch.tensor(iris[iris.columns[0:4]].values).float()

# transform species to number
# 0 -> setosa, 1 -> versicolor, 2 -> virginica
labels = torch.zeros(len(data), dtype=torch.long)
labels[iris.species == 'setosa'] = 0  # don't need actually !
labels[iris.species == 'versicolor'] = 1
labels[iris.species == 'virginica'] = 2

# Finally dataset
x = torch.tensor(data.numpy(), dtype=torch.float32, requires_grad=True, device=device)
y = torch.tensor(labels.numpy(), dtype=torch.long, device=device)

print("# Features: ", x.shape[1], "# Samples :", x.shape[0])
print("X shape :", x.shape, "\n", "\bY shape :", y.shape)
print(y.shape)

# -- Arch
class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 64)
        self.act2 = nn.ReLU()

        # Trick is here because we will use softmax !
        # 3 because there are 3 Class
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):

        z1 = self.fc1(x)
        a1 = self.act1(z1)

        z2 = self.fc2(a1)
        a2 = self.act2(z2)

        out = self.fc3(a2)

        return out

model = Net(4)
model.to(device)
model.train()

# Loss decleration
# Use CrossEntropyLoss for multi-class
loss_func = nn.CrossEntropyLoss()

# Optimizer
lr = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)

epochs = 10000
losses = []

for ep in range(epochs):

    # --forward
    y_hat = model.forward(x)

    # --loss where y should be 1D tensor includes class idxs
    loss = loss_func(y_hat, y)
    losses.append(loss.item())

    # --backward
    optimizer.zero_grad()
    loss.backward()

    # --update
    optimizer.step()

    if (ep + 1) % 100 == 0:
        print(f"Epoch {ep+1} loss : {loss.item()}")

model.eval()

# -- Print loss graph
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(losses)
plt.show()


# -- yhat = 150 x 3 tensor with no softmax
y_hat = model.forward(x)
print(y_hat[0,:])

# -- with softmax probablity sum is 1
sm = nn.Softmax(1)
y_hat = sm(y_hat)
print(y_hat[0,:])

# -- Get max idx for each row
y_hat = torch.argmax(y_hat, axis=1)
print(y_hat)


# -- Get finally prediction
y = y.to("cpu").detach().numpy().reshape(-1,1)
y_hat = y_hat.to("cpu").detach().numpy().reshape(-1,1)

# -- Confusion Matrix
cm = confusion_matrix(y, y_hat)
print("Confusion Matrix :",cm, sep="\n")





