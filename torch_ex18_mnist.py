import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MnistNet(nn.Module):

    def __init__(self, input_size = 28*28, output_size = 10):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128,128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128,32)
        self.fc5 = nn.Linear(32,32)
        self.fc6 = nn.Linear(32, output_size)
    
    def forward(self, x):
        
        # Exclude batch
        x = torch.flatten(x, start_dim=1)
        
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
        x = F.relu(x)

        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        # Log-softmax with NLL loss
        x = torch.log_softmax(x, dim=1)
        return x

    
def plot_digits(train_loader : DataLoader):

    inputs, labels = next(iter(train_loader))
    np_inputs = inputs.numpy()
    np_labels = labels.numpy()
    fig, axs = plt.subplots(4, 4)
    fig.tight_layout()

    i = 0
    for j, img in enumerate(np_inputs):
        axs[i, j%4].set_title(f"Class: #{np_labels[j]}")
        axs[i, j%4].imshow(np.squeeze(img), cmap="gray")

        if j%4 == 3:
            i+=1
def get_test_acc(model: MnistNet, test_loader: DataLoader):

    model.eval()

    with torch.no_grad():

        scores = []
        for i,(inputs, labels) in enumerate(test_loader):

            y_hat = model.forward(inputs.to(device))

            #sm = nn.Softmax(1)
            y_pred = torch.argmax(y_hat,axis=1)

            cm = confusion_matrix(labels.numpy().reshape(-1,1),
                                  y_pred.to("cpu").numpy().reshape(-1,1))
            
            
            score = ((np.eye(cm.shape[0]) * cm).sum() / cm.sum()) * 100
            scores.append(score)
    
    model.train()

    return np.mean(scores)

data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                ])

train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                           transform=data_transform)

test_dataset = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                           transform=data_transform)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

plot_digits(train_loader)
plt.show()

# TRAIN
model = MnistNet()
model.train()
model.to(device)

n_epoch = 10
n_step = len(train_dataset) // batch_size
lr = 0.001
# --loss
criterion = nn.NLLLoss()

# --optim
optim = torch.optim.Adam(model.parameters(), lr=lr)

epoch_losses = np.empty(n_epoch, dtype=np.float32)
test_scores = np.empty(n_epoch, dtype=np.float32)
for ep in range(n_epoch):

    step_losses = np.empty(n_step,dtype=np.float32)
    for step, (inputs, labels) in enumerate(train_loader):

        y_hat = model.forward(inputs.to(device))

        loss = criterion(y_hat, labels.to(device))
        step_losses[step] = loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
    epoch_losses[ep] = step_losses.mean()
    test_acc = get_test_acc(model, test_loader)
    test_scores[ep] = test_acc
    
    if (ep+1) % 1 == 0:

        print(f"Epoch = {ep+1} Step = {step} Loss = {epoch_losses[ep]:.4f} Test-Acc(%) = {test_acc}")
    
# -- Print loss graph
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(epoch_losses)
plt.show()

# -- Print loss graph
plt.xlabel("Epoch")
plt.ylabel("Acc(%)")
plt.plot(test_scores)
plt.show()





