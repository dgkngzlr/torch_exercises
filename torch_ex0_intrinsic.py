import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix
import numpy as np
import torch

def plt_data(X,y,n_cls):

    data = np.hstack((X, y.reshape(-1, 1)))

    cls_idxs = []
    cls = []
    for i in range(n_cls):

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

X, y = datasets.make_blobs(n_samples=1000,
                           centers=2,
                           n_features=2,
                           random_state=42)
plt_data(X,y,2)
plt.show()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

# --Lets built a logistic regres. model with a perceptron
def sigmoid(x):
    return 1 / (1+torch.exp(-x))

def binary_cross_entropy(y_hat, y):
    return -torch.mean( y*torch.log(y_hat) + (1-y) * (torch.log(1-y_hat)) )

def init_params(input_size):

    w = torch.randn(input_size,1, requires_grad=True) * 0.01
    w.retain_grad()
    b = torch.randn(1, requires_grad=True) * 0.01
    b.retain_grad()

    return w, b

def forward(x, w, b):
    return sigmoid(torch.mm(x, w)+b)

w, b = init_params(2)
y_hat = forward(X,w,b)
loss = binary_cross_entropy(y_hat, y)

print(50*"=")
print("First Loss:", loss.item())
print("First Weights:", w, sep="\n")
print("First Bias:", b, sep="\n")
print(50*"=")

n_epoch = 1000
lr = 0.01

for ep in range(n_epoch):

    # -- forward
    y_hat = forward(X,w,b)

    # -- loss
    loss = binary_cross_entropy(y_hat, y)

    # -- backward
    loss.backward(retain_graph=True)
    
    # -- update
    with torch.no_grad():
        
        w -= lr * w.grad
        b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()   
    
    
    if (ep+1) % 100 == 0:
        print(f"Loss : {loss.item()}")

print(50*"=")
print("After Train Loss:", loss.item())
print("After Train Weights:", w, sep="\n")
print("After Train Bias:", b, sep="\n")
print(50*"=")

with torch.no_grad():

    y_hat = forward(X,w,b)
    y_hat = (y_hat >= 0.5).int()
    y_hat = y_hat.numpy()

    # -- Confusion Matrix
    cm = confusion_matrix(y.numpy(), y_hat)
    print("\nCM:\n", cm)
    # -- Get score for test set
    score = (np.eye(2) * cm).sum() / cm.sum() * 100
    print("\nTrain Score (%):", score)
    

