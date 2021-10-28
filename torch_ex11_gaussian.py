import matplotlib.pyplot as plt
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix
import time
from torchsummary import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self, input_size,output_size, n_layer, n_neuron):

        super(Net, self).__init__()

        self.layers = nn.ModuleDict()
        self.n_layer = n_layer
        self.n_neuron = n_neuron

        # input layer
        self.layers['input'] = nn.Linear(input_size,n_neuron)
        
        # hidden layers
        for i in range(n_layer - 2):
            self.layers[f'hidden{i}'] = nn.Linear(n_neuron,n_neuron)

        # output layer
        self.layers['output'] = nn.Linear(n_neuron,output_size)

    def forward(self, x):
        
        out = F.relu(self.layers["input"](x))
       
        for i in range(self.n_layer - 2):
            out = F.relu(self.layers[f"hidden{i}"](out))

        out = self.layers['output'](out)

        return out

class ExampleDataset(Dataset):

    def __init__(self, X, y) -> None:
        
        self.x = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)
        self.n_samples = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

def plt_data(X,y):

    data = np.hstack((X, y.reshape(-1, 1)))

    cls_idxs = []
    cls = []
    for i in range(len(np.unique(y))):

        idx = np.argwhere(data[:,-1] == i).squeeze()
        cls_idxs.append(idx)

    for i in range(len(cls_idxs)):

        clss = np.take(data, cls_idxs[i], 0)
        cls.append(clss)

    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.title("Classification Problem with 1000 samples")

    for i, category in enumerate(cls):

        plt.scatter(category[:,0], category[:,1], c=np.random.rand(3,), label=f"{i}")

    plt.legend()

def train_the_model(model: Net, dataset: ExampleDataset,dataloader: DataLoader):
    model.to(device)
    lr = 0.03
    n_epoch = 1000
    n_iter = len(dataset) // batch_size
    
    # Loss
    criterion = nn.CrossEntropyLoss()

    #Optim
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    
    print(50 * "=")
    print(f"Total Epoch : {n_epoch} Iter per Epoch : {n_iter}")
    print(50 * "=")
    losses = []
    summary(model, (1,2))
    for ep in range(n_epoch):

        for i ,(inputs, outputs) in enumerate(dataloader):

            # --forward
            y_hat = model.forward(inputs)
            
            # --loss
            loss = criterion(y_hat, outputs)
            

            # --backward
            optimizer.zero_grad()
            loss.backward()

            # --update
            optimizer.step()
            if (i + 1) % n_iter == 0:
                losses.append(loss.item())
            if (ep + 1 ) % 100 == 0 and (i + 1) % n_iter == 0:

                print(f"Epoch {ep + 1} Step : {i + 1} loss : {loss.item():.3f}")
    
    # -- Print loss graph
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.show()

    return model

# Prepare regression data
X, y = datasets.make_gaussian_quantiles(n_samples = 1000, n_features = 2, n_classes=3, random_state=1)
plt_data(X,y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
batch_size = 32
dataset = ExampleDataset(X_train, y_train)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

model = Net(input_size=2, output_size=3, n_layer=4, n_neuron=16)
model.train()
model = train_the_model(model, dataset=dataset, dataloader=data_loader)


# Apply softmax model output for y_test
sm = nn.Softmax(1)
y_hat = model.forward(torch.tensor(X_test,dtype=torch.float32, device=device))
y_hat = sm(y_hat)
y_hat = torch.argmax(y_hat, axis=1)

# -- Get finally prediction
y_hat = y_hat.to("cpu").detach().numpy().reshape(-1,1)
y_test = y_test.reshape(-1,1)

# -- Confusion Matrix
cm = confusion_matrix(y_test, y_hat)
print("Confusion Matrix :",cm, sep="\n")

# -- Get score for test set
score = (np.eye(len(np.unique(y))) * cm).sum() / cm.sum()
print("Score (%):", score*100)