from os import sep
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

class Net(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32,16)
        self.bn4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.fc5(x)
        x = torch.sigmoid(x)

        return x
    
    def train_model(self, train_loader: DataLoader, dev_loader: DataLoader):
        
        n_epoch = 100
        lr = 0.001

        criterion = nn.BCELoss()

        optim = torch.optim.Adam(self.parameters(), lr=lr)

        epoch_losses = []
        dev_accs = []

        best_acc = 0
        for ep in range(n_epoch):
            
            step_losses = []
            for step, (inputs, labels) in enumerate(train_loader):

                y_hat = self.forward(inputs)

                loss = criterion(y_hat, labels)
                
                step_losses.append(loss.item())

                optim.zero_grad()
                loss.backward()
                optim.step()
            
            epoch_losses.append(np.mean(step_losses))
            step_losses.clear()
            dev_acc = self.get_dev_acc(dev_loader)
            dev_accs.append(dev_acc)
            
            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save(self.state_dict(), "water_best.pt")
                print("Best model saved !")

            if (ep+1) % 10 == 0:
                
                print(f"Epoch {ep+1} Step => {step} Loss => {epoch_losses[ep]} Dev_Acc => {dev_acc} Best => {best_acc}")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epoch_losses)
        plt.show()

        plt.xlabel("Epoch")
        plt.ylabel("Dev Acc")
        plt.plot(dev_accs)
        plt.show()


    def get_dev_acc(self, dev_loader: DataLoader):
        
        self.eval()
        score = 0
        with torch.no_grad():

            for step, (inputs, labels) in enumerate(dev_loader):

                y_hat = self.forward(inputs)

                y_pred = (y_hat > 0.5).int().numpy()

                # -- Confusion Matrix
                cm = confusion_matrix(labels.numpy(), y_pred)
                
                # -- Get score for dev set
                score = (np.eye(2) * cm).sum() / cm.sum() * 100
        
        self.train()

        return score

    def get_test_acc(self, test_loader: DataLoader):
        
        self.eval()
        score = 0
        with torch.no_grad():

            for step, (inputs, labels) in enumerate(test_loader):

                y_hat = self.forward(inputs)

                y_pred = (y_hat > 0.5).int().numpy()

                # -- Confusion Matrix
                cm = confusion_matrix(labels.numpy(), y_pred)
                print("CM:", cm, sep="\n")
                # -- Get score for dev set
                score = (np.eye(2) * cm).sum() / cm.sum() * 100
        
        self.train()

        return score

class WaterDataset(Dataset):

    def __init__(self, X, y):
        super().__init__()

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1,1)
        self.n_samples = X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.n_samples



def get_standardized_features(X, mean=[], std=[]):
    
    if len(mean) == 0 and len(std) == 0:
        std_scaler = StandardScaler()
        X_std = std_scaler.fit_transform(X)

        return np.array(X_std)
    
    else :
        X = (X - mean) / std
        return X

data = pd.read_csv("./datasets/water_potability_class.csv", delimiter=",")

X = data.iloc[:, :9]

# Fill NaN values with means
X = data.fillna(X.mean()).to_numpy()
y = data.iloc[:, -1].to_numpy()

X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.8, shuffle=True)
X_test, X_dev, y_test, y_dev = train_test_split(X_temp, y_temp, train_size=0.5, shuffle=True)

train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)

X_train = get_standardized_features(X_train)
X_test = get_standardized_features(X_test, mean=train_mean, std=train_std)
X_dev = get_standardized_features(X_dev, mean=train_mean, std=train_std)

train_dataset = WaterDataset(X_train, y_train)
dev_dataset = WaterDataset(X_dev, y_dev)
test_dataset = WaterDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=len(dev_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

model = Net(10)
model.train()
model.train_model(train_loader, dev_loader)

score = model.get_test_acc(test_loader)
print(score)









