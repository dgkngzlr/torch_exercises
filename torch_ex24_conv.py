from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img_rgb = plt.imread("https://images.squarespace-cdn.com/content/v1/54ca9d3be4b03719029594ff/1573225122179-OG1SJDF22221CBFJLEES/DSC00135.JPG?format=1000w",format="jpg")#"https://wikiimg.tojsiabtv.com/wikipedia/en/7/7d/Lenna_%28test_image%29.png")
img_gray = rgb2gray(img_rgb)

img_rgb_t = torch.tensor(img_rgb, dtype=torch.float32).view(1, 3, img_rgb.shape[0], img_rgb.shape[1])
img_gray_t = torch.tensor(img_gray, dtype=torch.float32).view(1, 1, img_gray.shape[0], img_gray.shape[1])

# 3 channel convolve
kernel = torch.tensor([[[1,2,1],[0,0,0],[-1,-2,-1]],
                        [[1,2,1],[0,0,0],[-1,-2,-1]],
                        [[0,0,0],[0,0,0],[0,0,0]]],
                         dtype=torch.float32).view(1,3,3,3)

res = F.conv2d(img_rgb_t, kernel, padding=(1,1))

plt.imshow(torch.squeeze(res), cmap="gray")
plt.show()

# 1 channel convolve
kernel = torch.tensor([[3,10,3],[0,0,0],[-3,-10,-3]],dtype=torch.float32).view(1,1,3,3)

res = F.conv2d(img_gray_t, kernel, padding=(1,1))

plt.imshow(torch.squeeze(res), cmap="gray")
plt.show()

in_chan = 3 # RGB
out_chan = 15 # n_of_kernel used for that layer
kernel_size = 5
stride = 1
padding = 0

conv_layer = nn.Conv2d(in_chan, out_chan, kernel_size, stride=stride, padding=padding)

# 15 diff. kernel with form 3x5x5
print("Size of weights :", str(conv_layer.weight.shape))
print("Size of bias :", str(conv_layer.bias.shape))

fig,axs = plt.subplots(3,5,figsize=(10,5))

j = 1
for i,ax in enumerate(axs.flatten()):
  ax.imshow(torch.squeeze(conv_layer.weight[i,0,:,:]).detach(),cmap='Purples')
  ax.set_title('Kernel(%s)'%j)
  j += 1
  ax.axis('off')

plt.tight_layout()
plt.show()

# size of the image (N, RGB, width, height)
imsize = (1,3,64,64)

img = torch.rand(imsize)

# pytorch wants channels first, but matplotlib wants channels last.
# therefore, tensors must be permuted to visualize
img2view = img.permute(2,3,1,0).numpy()
print(img.shape)
print(img2view.shape)

plt.imshow(np.squeeze(img2view))

# convolve the image with the filter bank (set of 'outChans' kernels)
convRes = conv_layer(img)

print(img.shape)
print(convRes.shape)

# What do the convolved images look like? (Hint: think of the bathtub picture.)

fig,axs = plt.subplots(3,5,figsize=(10,5))

for i,ax in enumerate(axs.flatten()):

  # extract this "layer" of the convolution result
  I = torch.squeeze(convRes[0,i,:,:]).detach()

  # and visualize it
  ax.imshow(I,cmap='Purples')
  ax.set_title('Conv. w/ filter %s'%i)
  ax.axis('off')

plt.tight_layout()
plt.show()
