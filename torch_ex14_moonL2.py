from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

class MoonDataset(Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1,1)
        self.n_samples = self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

def plt_data(X,y):

    data = np.hstack((X, y.reshape(-1, 1)))

    cls_idxs = []
    cls = []
    for i in range(len(np.unique(y))):

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

X, y = datasets.make_moons(n_samples=1000, noise=0.1)
plt_data(X,y)
plt.show()

# Split dataset train, dev and test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.33, random_state=42)
X_test, X_dev, y_test, y_dev = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

batch_size = 32

train_dataset = MoonDataset(X_train, y_train)
train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

dev_dataset = MoonDataset(X_dev, y_dev)
dev_loader = DataLoader(dev_dataset,batch_size=len(dev_dataset), shuffle=True)

test_dataset = MoonDataset(X_test, y_test)
test_loader = DataLoader(test_dataset,batch_size=len(test_dataset), shuffle=True)

class Net(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,6)
        self.fc4 = nn.Linear(6,1)
        self.act = nn.Sigmoid()
    
    def forward(self, x):

        x = self.fc1(x);
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.act(x)

        return x

def get_dev_acc(model : nn.Module, dev_loader: DataLoader):

    score = 0
    model.eval()

    with torch.no_grad():

        for i, (inputs, label) in enumerate(dev_loader):

            y_hat = model.forward(inputs)
            pred = (y_hat > 0.5).int().numpy().reshape(-1,1)

            # -- Confusion Matrix
            cm = confusion_matrix(label, pred)
            # -- Get score for dev set
            score = (np.eye(2) * cm).sum() / cm.sum() * 100
    
    model.train()

    return score

def get_test_acc(model, test_loader):

    print("Test Result :")
    model.eval()

    with torch.no_grad():

        for i, (inputs, label) in enumerate(test_loader):

            y_hat = model.forward(inputs)
            pred = (y_hat > 0.5).int().numpy().reshape(-1,1)

            # -- Confusion Matrix
            cm = confusion_matrix(label, pred)
            print("\n CM : ", cm, sep="\n")
            # -- Get score for dev set
            score = (np.eye(2) * cm).sum() / cm.sum() * 100
            print("Score:", score)

# Train
n_epoch = 1000
n_step = len(train_dataset) // batch_size
lr = 0.008
weight_decay = 1e-4

# Binary model
model = Net(2)
model.train()

# Binary classification
criterion = nn.BCELoss()

# -- optim
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

losses = []

for ep in range(n_epoch):
    
    step_losses = []
    for i,(inputs, label) in enumerate(train_loader):
        
        #-- forward
        y_hat = model.forward(inputs)
        
        #--loss
        loss = criterion(y_hat, label)
        step_losses.append(loss.item())

        #--backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    losses.append(np.mean(step_losses))
    step_losses.clear()

    dev_acc = get_dev_acc(model, dev_loader)

    if (ep+1) % 100 == 0:
        print(f"Epoch {ep+1} loss : {losses[ep]} dev_acc : {dev_acc:.2f}")


get_test_acc(model, test_loader)
# -- Print loss graph
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(losses)
plt.show()











