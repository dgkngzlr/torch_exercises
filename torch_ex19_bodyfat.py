from numpy.core.fromnumeric import ptp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def get_standardized_features(X, mean=[], std=[]):
    
    if len(mean) == 0 and len(std) == 0:
        std_scaler = StandardScaler()
        X_std = std_scaler.fit_transform(X)

        return np.array(X_std)
    
    else :
        X = (X - mean) / std
        return X

class ExampleDataset(Dataset):

    def __init__(self,X, y) -> None:
        super().__init__()

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.n_samples = self.X.shape[0]
    
    def __getitem__(self, index) :
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

class Net(nn.Module):

    def __init__(self, input_size=14, output_size=1):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)
    
    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
    
        return self.fc2(x)


data = pd.read_csv('./datasets/bodyfat_reg.csv', delimiter=",")

X = data[["Density","Age","Weight","Height",
          "Neck","Chest","Abdomen","Hip",
          "Thigh","Knee","Ankle","Biceps",
          "Forearm","Wrist"]].to_numpy()

X = get_standardized_features(X)
y = data[["BodyFat"]].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)

train_dataset = ExampleDataset(X_train, y_train)
test_dataset  = ExampleDataset(X_test, y_test)

# Train
model = Net()
model.train()

batch_size = 16
n_epoch = 3000
n_step = len(train_dataset) // batch_size
lr = 0.001

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

criterion = nn.MSELoss()

optim = torch.optim.Adam(model.parameters(), lr=lr)

plt.title("Before Training 1th Layer Weights Histogram")
plt.xlabel("Weight")
plt.ylabel("Amount")
plt.hist(model.fc1.weight.detach().numpy().flatten())
plt.show()
epoch_losses = []
for ep in range(n_epoch):

    step_losses = []
    for step, (inputs, outputs) in enumerate(train_loader):

        y_hat = model.forward(inputs)

        loss = criterion(y_hat, outputs)
        step_losses.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()
    
    epoch_losses.append(np.mean(step_losses))

    if (ep+1) % 100 == 0:
        print(f"Epoch {ep+1} Step {step} MSE loss : {epoch_losses[ep]:.4f}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(epoch_losses)
plt.show()

model.train()

plt.title("After Training 1th Layer Weights Histogram")
plt.xlabel("Weight")
plt.ylabel("Amount")
plt.hist(model.fc1.weight.detach().numpy().flatten())
plt.show()

with torch.no_grad():

    for i, (inputs, outputs) in enumerate(test_loader):

        y_hat = model.forward(inputs)

        mse_loss = F.mse_loss(y_hat, outputs)
        print("Predicted :")
        print(y_hat[:10,:])
        print("Actual :")
        print(outputs[:10,:])

    print("Test MSE loss : ", mse_loss.item())






