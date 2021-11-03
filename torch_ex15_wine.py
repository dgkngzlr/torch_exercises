import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 0 mean and 1 std
def get_standardized_features(X):
    
    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(X)
    return np.array(X_std)

 # y > 5 than 1, else 0
def get_binarized_label(y):

    y_bin = np.array((y > 5).astype(np.int32))
    return y_bin

def get_test_acc(model, test_loader):

    print("Test Result :")
    model.eval()
    score = 0
    with torch.no_grad():

        for i, (inputs, label) in enumerate(test_loader):

            y_hat = model.forward(inputs)
            y_hat_ = y_hat.cpu().clone().numpy()
            pred = (y_hat_ > 0.5).astype(np.int32).reshape(-1,1)
            label_ = label.cpu().clone().numpy()
            # -- Confusion Matrix
            cm = confusion_matrix(label_, pred)
            print("\n CM : ", cm, sep="\n")
            # -- Get score for dev set
            score = (np.eye(2) * cm).sum() / cm.sum() * 100
            print("Score:", score)
            
    
    model.train()
    return score

class WineDataset(Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()
        
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device).view(-1,1)
        self.n_samples = self.X.shape[0]
    
    def __getitem__(self, index) :
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,32)
        self.fc5 = nn.Linear(32,1)

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = torch.sigmoid(x)

        return x

data = pd.read_csv(URL,sep=';')

X, y = get_standardized_features(data.iloc[:,:11]),\
       get_binarized_label(data.iloc[:,-1]) # Since there is no enough sample per class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle=True)

train_dataset = WineDataset(X_train, y_train)
test_dataset = WineDataset(X_test, y_test)

batches = {16: None,
           32: None,
           64: None,
           128: None}


for batch in batches.keys():
    
    print(f"===== With {batch} batch ! ====== ")
    model = Net(11)
    model.train()
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    lr = 0.01
    n_epoch = 1000
    

    # -- loss
    criterion = nn.BCELoss()

    # -- optim
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    epoch_losses = []
    for ep in range(n_epoch):

        step_losses = []
        for i,(inputs, label) in enumerate(train_loader):

            y_hat = model.forward(inputs)
            
            loss = criterion(y_hat, label)
            step_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        epoch_losses.append(np.mean(step_losses))
        if (ep+1) % 100 == 0:
            print(f"Epoch {ep+1} loss: {np.mean(step_losses)}")
        
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_losses)
    plt.show()

    test_acc = get_test_acc(model, test_loader)
    
    batches[batch] = test_acc

plt.xlabel("Batch size")
plt.ylabel("Acc (%)")
plt.bar(batches.keys(), batches.values())
plt.show()







