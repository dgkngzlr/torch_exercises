import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix
import time

# Hyper Params
batch_size = 64
learning_rate = 0.01

class Net(nn.Module):

    def __init__(self, input_size, output_size):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, 6)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(6,6)
        self.act2 = nn.ReLU()

        self.out = nn.Linear(6, output_size)

    def forward(self, x):

        z1 = self.fc1(x)
        a1 = self.act1(z1)

        z2 = self.fc2(a1)
        a2 = self.act2(z2)

        return self.out(a2)


class ExampleDataset(Dataset):

    def __init__(self, X, y):

        xy = np.hstack((X, y.reshape(-1, 1)))
        self.x = torch.tensor(xy[:,:2], dtype=torch.float32)#torch.from_numpy(xy[:,:2].astype(np.float32))
        self.y = torch.tensor(xy[:,-1], dtype=torch.long)#torch.from_numpy(xy[:,-1].astype(np.int_))
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

def plt_data(X,y):

    data = np.hstack((X, y.reshape(-1, 1)))

    cls_0_idxs = np.argwhere(data[:,-1] == 0).squeeze()
    cls_1_idxs = np.argwhere(data[:, -1] == 1).squeeze()
    cls_2_idxs = np.argwhere(data[:, -1] == 2).squeeze()

    # Get classes with features
    cls_0 = np.take(data, cls_0_idxs, 0) # Where label is 0
    cls_1 = np.take(data, cls_1_idxs, 0) # Where label is 1
    cls_2 = np.take(data, cls_2_idxs, 0) # Where label is 2

    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.title("Classification Problem with 1000 samples")
    plt.scatter(cls_0[:,0], cls_0[:,1], color="red", label="0")
    plt.scatter(cls_1[:, 0], cls_1[:, 1], color="orange", label="1")
    plt.scatter(cls_2[:, 0], cls_2[:, 1], color="blue", label="2")
    plt.legend()

X, y = datasets.make_blobs(n_samples=1000,
                           centers=3,
                           n_features=2,
                           random_state=13)
plt_data(X,y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# dataset object init
dataset = ExampleDataset(X_train,y_train)

# Print first sample info
print(50 * "=")
print("First feature first label :")
first_feature, first_label = dataset[0]
print(first_feature, first_label)
print(50 * "=")
# Init dataloader
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Get first 32 batch from train data
dataiter = iter(dataloader)
batch_feature, batch_label = next(dataiter)
print(50 * "=")
print("First batch feature size and label size :")
print(batch_feature.shape, batch_label.shape)
print(50 * "=")

# --init model
model = Net(2, 3)
model.train()

# -- Loss
criterion = nn.CrossEntropyLoss()

# -- Optim
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

n_iter = len(dataset) // batch_size
n_epoch = 1000
losses = []
print(50 * "=")
print(f"Total Epoch : {n_epoch} Iter per Epoch : {n_iter}")
print(50 * "=")
time.sleep(3)
for ep in range(n_epoch):

    for i, (inputs, labels) in enumerate(dataloader):

        # --forward
        y_hat = model.forward(inputs)

        # --loss
        loss = criterion(y_hat, labels)
        losses.append(loss.item())

        # --backward
        optimizer.zero_grad()
        loss.backward()

        # --update
        optimizer.step()

        if (ep + 1) % 100 == 0 and (i + 1) % 10 == 0:

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
score = (np.eye(3) * cm).sum() / cm.sum()
print("Score (%):", score*100)
