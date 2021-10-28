import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix
import seaborn as sns

class Net(nn.Module):
    
    def __init__(self, input_size, output_size, n_neuron):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, n_neuron)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(n_neuron, output_size)
    
    def forward(self, x):

        z1 = self.fc1(x)
        a1 = self.act1(z1)

        return self.fc2(a1)

class IrisDataset(Dataset):

    def __init__(self, X, y) -> None:
        
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.n_samples = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

# pandas frame  sepal_length, sepal_width, petal_length, petal_width, species
data = sns.load_dataset("iris")
X = data.iloc[:,:4].to_numpy().astype(np.float32)
y = data.iloc[:,-1]
y = y.replace({"setosa" : 0, "versicolor" : 1, "virginica" : 2}, inplace=False)
y = y.to_numpy().astype(np.int64)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
dataset = IrisDataset(X_train, y_train)
data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

pred_rate = []
# For each experiment 8, 16, 32, 64, 128
for i in range(0, 8):

    model = Net(4, 3, 2**i)
    model.train()
    print(model)
    # --loss
    criterion = nn.CrossEntropyLoss()

    # --optim
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    n_epoch = 1000
    n_iter = len(dataset) // 32

    print(50 * "=")
    print(f"TRAINING IS STARTING FOR {2**i} NEURON")
    print(50 * "=")
    for ep in range(n_epoch):

        for i, (inputs, labels) in enumerate(data_loader):

            # --forward
            y_hat = model.forward(inputs)

            # --loss
            loss = criterion(y_hat, labels)

            # --backward
            optimizer.zero_grad()
            loss.backward()

            # --update
            optimizer.step()

            if ((ep+1) % 100 == 0) and ((i+1) % n_iter == 0):

                print(f"Epoch => {ep + 1} Step : {i + 1} Loss : {loss.item():.5f}")
    
    model.eval()

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

    # -- Get score for test set
    score = (np.eye(3) * cm).sum() / cm.sum()
    print("Score (%):", score*100)
    pred_rate.append(score*100)
    del(model)
    del(criterion)
    del(optimizer)
    time.sleep(2)
    print(50 * "=")
    print(50 * "=")

plt.xlabel("n_neuron")
plt.ylabel("Pred. Rate")
plt.bar([2**i for i in range(0,8)], pred_rate)
plt.show()
