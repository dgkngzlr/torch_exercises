# import libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# create data

nPerClust = 100
blur = 1

A = [  1, 1 ]
B = [  5, 1 ]

# generate data
a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur ]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur ]

# true labels
labels_np = np.vstack((np.zeros((nPerClust,1)),np.ones((nPerClust,1))))

# concatanate into a matrix
data_np = np.hstack((a,b)).T

# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

# show the data
fig = plt.figure(figsize=(5,5))
plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs')
plt.plot(data[np.where(labels==1)[0],0],data[np.where(labels==1)[0],1],'ko')
plt.title('The qwerties!')
plt.xlabel('qwerty dimension 1')
plt.ylabel('qwerty dimension 2')
plt.show()

# Torch data
x = torch.tensor(data_np , dtype=torch.float32, requires_grad=True, device=device)
y = torch.tensor(labels_np, dtype=torch.float32, device=device).view(-1,1)

class Net(nn.Module):

    def __init__(self, input_size):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, 4)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(4, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):

        z1 = self.fc1(x)
        a1 = self.act1(z1)

        z2 = self.fc2(a1)
        out = self.act2(z2)

        return out

model = Net(2)
model.to(device)

model.train()

loss_func = nn.BCELoss()

lr = 0.08
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

y_hat = 0
losses = []
epoch = 10000
for ep in range(epoch):

    # --forward
    y_hat = model.forward(x)

    # --loss
    loss = loss_func(y_hat, y)
    losses.append(loss.item())
    # --backward
    optimizer.zero_grad()
    loss.backward()

    # --update
    optimizer.step()

    if (ep + 1) % 100 == 0:
        print(f"Epoch {ep + 1} loss : {loss.item()}")

model.eval()

# -- Print loss graph
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(losses)
plt.show()

# -- Get finally prediction
y = y.to("cpu").detach().numpy().reshape(-1,1)
y_hat = (y_hat > 0.5).int().to("cpu").detach().numpy().reshape(-1,1)

# -- Confusion Matrix
cm = confusion_matrix(y, y_hat)
print("Confusion Matrix :",cm, sep="\n")

