import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class Net(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def train_model(self, train_loader, dev_loader):

        n_epoch = 500
        lr = 0.001

        criterion = nn.MSELoss()

        optim = torch.optim.Adam(self.parameters(), lr=lr)

        epoch_losses = []
        dev_mses = []
        best_mse = 1e4
        prev_loss = 1e4
        for ep in range(n_epoch):

            step_losses = []
            for step, (inputs, values) in enumerate(train_loader):

                y_hat = self.forward(inputs)

                loss = criterion(y_hat, values)
                step_losses.append(loss.item())

                optim.zero_grad()
                loss.backward()
                optim.step()

            epoch_losses.append(np.mean(step_losses))
            step_losses.clear()
            dev_mse = self.get_dev_acc(dev_loader)
            dev_mses.append(dev_mse)
            
            if dev_mse <= best_mse and epoch_losses[ep] < prev_loss:
                best_mse = dev_mse
                prev_loss = epoch_losses[ep]
                torch.save(self.state_dict(), "adding_best.pt")
                print("Best model saved !")

            if (ep+1) % 10 == 0:
                
                print(f"Epoch {ep+1} Step => {step} Loss => {epoch_losses[ep]} Dev_MSE => {dev_mse} Best-MSE => {best_mse}")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epoch_losses)
        plt.show()

        plt.xlabel("Epoch")
        plt.ylabel("Dev Acc")
        plt.plot(dev_mses)
        plt.show()

    def get_dev_acc(self, dev_loader: DataLoader):
        
        self.eval()
        score = 100
        with torch.no_grad():

            for step, (inputs, values) in enumerate(dev_loader):

                y_hat = self.forward(inputs)
                
                y_pred = torch.round(y_hat).int()
                
                # -- Get score for dev set
                score = F.mse_loss(y_hat, values)
        
        self.train()

        return score
    
    def get_test_acc(self, test_loader: DataLoader):
        
        self.eval()
        score = 0
        with torch.no_grad():

            for step, (inputs, values) in enumerate(test_loader):

                y_hat = self.forward(inputs)
                
                y_pred = torch.round(y_hat).int()
                
                # -- Get score for dev set
                score = F.mse_loss(y_hat, values)

                trues = (y_pred.numpy() == values.numpy())
                print("Score :", trues.sum() / trues.shape[0] * 100)
                print("Input:",inputs[0,:],"Actual:",values[0,0],"Predicted:",y_pred[0,0])
        
        self.train()

        return score

class ExampleDataset(Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.n_samples = X.shape[0]
    
    def __getitem__(self, index) :
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

X = np.random.randint(-10,11, size=(2000,2))
y = (X[:,0] + X[:,1]).reshape(-1,1)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size = 0.8)
X_test, X_dev, y_test, y_dev = train_test_split(X_temp, y_temp, train_size=0.5)

train_dataset = ExampleDataset(X_train, y_train)
dev_dataset = ExampleDataset(X_dev, y_dev)
test_dataset = ExampleDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=len(dev_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

model = Net(2)
model.train()
model.train_model(train_loader, dev_loader)

mse = model.get_test_acc(test_loader)
print("MSE Test:", mse)





