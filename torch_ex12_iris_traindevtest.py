import re
import torch
from torch._C import device
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, dataloader
import torch.nn.functional as F
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torchsummary.torchsummary import summary
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 4)
        self.fc4 = nn.Linear(4, output_size)
    
    def forward(self, x):

        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)

        return out

class ExampleDataset(Dataset):

    def __init__(self, X, y):
        super().__init__()

        self.x = torch.tensor(X.astype(float), dtype=torch.float32, device=device)
        self.y = torch.tensor(y.astype(int), dtype=torch.long, device=device)
        self.n_samples = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

def pre_process(data):
    
    X = data.iloc[:,:4].to_numpy()
    y = data.iloc[:,-1].to_numpy()
    categories = np.unique(y)
    y = np.where(y == categories[0], 0, y)
    y = np.where(y == categories[1], 1, y)
    y = np.where(y == categories[2], 2, y)

    return X, y

def get_dev_accur(model: Net, dev_loader: DataLoader):

    sm = nn.Softmax(1)

    with torch.no_grad():

        for inputs, labels in dev_loader:
            
            y_hat = model.forward(inputs)
            y_hat = sm(y_hat)
            predicted_label = torch.argmax(y_hat, axis=1)

            labels_ = labels.to("cpu").numpy().reshape(-1,1)
            predicted_label = predicted_label.to("cpu").numpy().reshape(-1,1)

            # -- Confusion Matrix
            cm = confusion_matrix(labels_, predicted_label)
            # -- Get score for test set
            score = (np.eye(3) * cm).sum() / cm.sum() * 100

    return score    

def train_the_model(model: Net, train_loader: DataLoader, dev_loader: DataLoader):

    # Params
    lr = 0.001
    n_epoch = 2000
    n_feature = next(iter(train_loader))[0].shape[1]

    # loss
    criterion = nn.CrossEntropyLoss()

    # Optim
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    summary(model, (1,n_feature))
    epoch_losses = []
    accuracies = []
    for ep in range(n_epoch):
        
        step_losses = []
        for i, (inputs, labels) in enumerate(train_loader):

            # forward
            y_hat = model.forward(inputs)

            # loss
            loss = criterion(y_hat, labels)
            step_losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # update
            optimizer.step()
        
        epoch_loss = np.mean(step_losses)
        epoch_losses.append(epoch_loss)
        dev_accuracy = get_dev_accur(model, dev_loader)
        accuracies.append(dev_accuracy)

        if (ep+1) % 100 == 0:
            print(f"Epoch {ep+1} loss : {epoch_loss:.3f} Dev-Accur(%):{dev_accuracy:.3f}")
    
    # -- Print loss graph
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_losses)
    plt.show()

    # -- Print dev-acc graph
    plt.xlabel("Dev-Accur")
    plt.ylabel("Prediction rate")
    plt.plot(accuracies)
    plt.show()

    return model

# Get data
iris = sns.load_dataset("iris")
X, y = pre_process(iris)

# Spilit data train, dev, test
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, train_size=0.8, random_state=42)
X_test, X_dev, y_test, y_dev = train_test_split(X_tmp, y_tmp, train_size=0.5, random_state=42)
print("Train Size (%):", X_train.shape[0] / X.shape[0]*100, \
      "Dev Size (%):", X_dev.shape[0] / X.shape[0]*100, \
      "Test Size (%):", X_test.shape[0] / X.shape[0]*100)

# Dataloader
train_batch_size = 32
train_set = ExampleDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True)

dev_set = ExampleDataset(X_dev, y_dev)
dev_loader = DataLoader(dataset=dev_set, batch_size=len(dev_set), shuffle=True)

test_set = ExampleDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_set, batch_size=len(test_set), shuffle=True)

model = Net(4, 3)
model.to(device)

trained_model = train_the_model(model, train_loader, dev_loader)

# Get test dataset score
trained_model.eval()

sm = nn.Softmax(1)

with torch.no_grad():

    for inputs, labels in test_loader:
        
        y_hat = trained_model.forward(inputs)
        y_hat = sm(y_hat)
        predicted_label = torch.argmax(y_hat, axis=1)

        labels_ = labels.to("cpu").numpy().reshape(-1,1)
        predicted_label = predicted_label.to("cpu").numpy().reshape(-1,1)

        # -- Confusion Matrix
        cm = confusion_matrix(labels_, predicted_label)
        print("\nCM:\n", cm)
        # -- Get score for test set
        score = (np.eye(3) * cm).sum() / cm.sum() * 100
        print("\nTest Score (%):", score)



