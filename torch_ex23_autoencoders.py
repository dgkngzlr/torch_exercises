import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import datasets, svm, metrics

class AutoencoderMNIST(nn.Module):

    def __init__(self, input_size = 28*28, output_size = 28*28):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)# Bottleneck
        self.fc4 = nn.Linear(64, 128) 
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, output_size)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = F.relu( self.fc3(x) )
        x = F.relu( self.fc4(x) )
        x = F.relu( self.fc5(x) )
        x = self.fc6(x)

        return torch.sigmoid(x)
    
    def train_model(self, train_loader, test_loader):

        n_epoch = 10
        lr = 0.01

        criterion = nn.MSELoss()

        optim = torch.optim.Adam(self.parameters(), lr=lr)

        epoch_losses = []

        for ep in range(n_epoch):

            step_losses = []
            for step, (inputs, _) in enumerate(train_loader):

                y_hat = self.forward(inputs)

                loss = criterion(y_hat, torch.flatten(inputs, start_dim=1))
                
                step_losses.append(loss.item())

                optim.zero_grad()
                loss.backward()
                optim.step()
            
            epoch_losses.append(np.mean(step_losses))
            step_losses.clear()
            
            if (ep+1) % 1 == 0:
        
                print(f"Epoch {ep+1} Step => {step} Loss => {epoch_losses[ep]}")
    
    def get_result_as_image(self, x):
        x = self.forward(x)
        return x.reshape(28,28)




data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                ])

train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                           transform=data_transform)

test_dataset = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                           transform=data_transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=True)

model = AutoencoderMNIST()
model.train()
model.train_model(train_loader, test_loader)

inputs, _ = test_dataset[0]

y_pred = model.get_result_as_image(inputs).detach().numpy()
inputs = inputs.detach().numpy()

fig,axs = plt.subplots(2,5)
fig.tight_layout()

for i in range(5):

    inputs, _ = test_dataset[i]

    y_pred = model.get_result_as_image(inputs).detach().numpy()
    inputs = inputs.detach().numpy()

    axs[0,i].set_title("Model input")
    axs[0,i].imshow(np.squeeze(inputs) ,cmap='gray')
    axs[1,i].set_title("Model output")
    axs[1,i].imshow(np.squeeze(y_pred) ,cmap='gray')

plt.show()