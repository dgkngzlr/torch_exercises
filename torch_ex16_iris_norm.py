import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

class Net(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,output_size)
    
    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

class IrisDataset(Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.n_samples = self.X.shape[0]
    
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

def get_iris_data():

    # pandas frame  sepal_length, sepal_width, petal_length, petal_width, species
    data = sns.load_dataset("iris")
    X = data.iloc[:,:4].to_numpy().astype(np.float32)
    y = data.iloc[:,-1]
    y = y.replace({"setosa" : 0, "versicolor" : 1, "virginica" : 2}, inplace=False)
    y = y.to_numpy().astype(np.int32)

    return X, y

X, y = get_iris_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)

X_train = get_standardized_features(X_train)
X_test = get_standardized_features(X_test, mean=train_mean, std=train_std)

train_dataset = IrisDataset(X_train, y_train)
test_dataset = IrisDataset(X_test, y_test)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)


n_epoch = 2000
lr = 0.008
weight_decay = 1e-4

model = Net(4,3)
model.train()

# --loss 
criterion = nn.CrossEntropyLoss()

# --optim
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)


for ep in range(n_epoch):
    step = 0
    
    for step, (inputs, label) in enumerate(train_loader):

        y_hat = model.forward(inputs)
        
        loss = criterion(y_hat, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (ep+1) % 100 == 0:
        print(f"Epoch {ep+1} step : {step} loss : {loss.item():.3f}")

sm = nn.Softmax(dim=1)
model.eval()

with torch.no_grad():

    for inputs,label in test_loader:

        y_hat = model.forward(inputs)
        print(y_hat)
        y_hat = sm(y_hat)
        print(y_hat)
        y_hat = torch.argmax(y_hat, axis=1)
        y_hat = y_hat.numpy().reshape(-1,1)
        label = label.numpy().reshape(-1,1)

        # -- Confusion Matrix
        cm = confusion_matrix(label, y_hat)
        print(cm)
        # -- Get score for test set
        score = (np.eye(3) * cm).sum() / cm.sum()
        print("Test Score (%):", score*100)






