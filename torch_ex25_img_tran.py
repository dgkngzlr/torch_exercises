import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T

transform = T.Compose([T.ToTensor(),
                       T.Resize(256),
                       T.CenterCrop(224),
                       #T.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124],std=[0.24703233,0.24348505,0.26158768])
                       ])

train_dataset = torchvision.datasets.CIFAR10("datasets/CIFAR10/train", train=True, \
                                             transform=transform, download=True)

train_dataset = torchvision.datasets.CIFAR10("datasets/CIFAR10/test", train=False, \
                                             transform=transform, download=True)

key_list = list(train_dataset.class_to_idx.keys())


fig,axs = plt.subplots(nrows=5, ncols=5, figsize=(12,9))

k = 0
for i in range(5):
    for j in range(5):
        data, label = train_dataset[k]
        axs[i,j].set_title(key_list[label].title())
        axs[i,j].imshow(data.permute(1,2,0))
        k += 1

plt.tight_layout()
plt.show()
