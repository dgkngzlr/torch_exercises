from operator import le
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix
import time

class Net(nn.Module):

    def __init__(self, input_size: int, output_size: int):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size,8)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(8, 8)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(8, 8)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(8, output_size)
    
    def forward(self, x):

        z1 = self.fc1(x)
        a1 = self.act1(z1)

        z2 = self.fc2(a1)
        a2 = self.act2(z2)

        z3 = self.fc3(a2)
        a3 = self.act3(z3)

        return self.fc4(a3)

class ExampleDataset(Dataset):

    def __init__(self, X, y) -> None:
        
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.n_samples = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

def plt_data(X,y):

    data = np.hstack((X, y.reshape(-1, 1)))

    cls_idxs = []
    cls = []
    for i in range(5):

        idx = np.argwhere(data[:,-1] == i).squeeze()
        cls_idxs.append(idx)

    for i in range(len(cls_idxs)):

        clss = np.take(data, cls_idxs[i], 0)
        cls.append(clss)

    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.title("Classification Problem with 1000 samples")

    for i, category in enumerate(cls):

        plt.scatter(category[:,0], category[:,1], c=np.random.rand(3,), label=f"{i}")

    plt.legend()

# 5 class blob dataset
X, y = datasets.make_blobs(n_samples=1000,
                           centers=5,
                           n_features=2,
                           cluster_std=0.9,
                           random_state=67)

plt_data(X, y)
plt.show()

model = Net(2, 5)
model.train()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dataset = ExampleDataset(X_train, y_train)

# -- params
batch_size = 32
lr = 0.001
n_epoch = 1000
n_iter = len(dataset) // batch_size

data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# --define loss
criterion = nn.CrossEntropyLoss()

# --optim
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

losses = []
for ep in range(n_epoch):

    for i, (inputs, labels) in enumerate(data_loader):

        # -- forward
        y_hat = model.forward(inputs)

        # --loss
        loss = criterion(y_hat, labels)
        losses.append(loss.item())

        # --backward
        optimizer.zero_grad()
        loss.backward()

        # --update 
        optimizer.step()

        if ((ep+1) % 100 == 0) and ((i+1) % n_iter == 0):

            print(f"Epoch => {ep + 1} Step : {i + 1} Loss : {loss.item():.5f}")

model.eval()

# -- Print loss graph
plt.xlabel("Step")
plt.ylabel("Loss")
plt.plot(losses)
plt.show()

# Apply softmax model output for y_test
sm = nn.Softmax(1)
y_hat = model.forward(torch.tensor(X_test,dtype=torch.float32))
y_hat = sm(y_hat)
y_hat = torch.argmax(y_hat, axis=1)

# -- Get finally prediction
y_hat = y_hat.to("cpu").detach().numpy().reshape(-1,1)
y_test = y_test.reshape(-1,1)

# -- Confusion Matrix
cm = confusion_matrix(y_test, y_hat)
print("Confusion Matrix :",cm, sep="\n")

# -- Get score for test set
score = (np.eye(5) * cm).sum() / cm.sum()
print("Score (%):", score*100)
print("y_hat  |  y_true :")
pred = np.hstack((y_hat, y_test))
print(pred)
if score*100 > 90:
    print("Model saved !")
    torch.save(model.state_dict(), "./ex8.pth")