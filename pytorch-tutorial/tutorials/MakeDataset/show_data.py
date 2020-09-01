# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:45:36 2020

@author: opgg
"""

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from dataset import MyDataset
import matplotlib.pyplot as plt
from PIL import Image


train_config_path = 'data_img/mnist_train.csv'

test_data = MyDataset(root=train_config_path, transform=transforms.ToTensor())
data_loader = DataLoader(test_data, batch_size=100,shuffle=True)
print("len(data_loader) = ", len(data_loader))


def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')


for i, (batch_x, batch_y) in enumerate(data_loader):
    if(i<4):
        plt.figure()
        print(i, batch_x.size(),batch_y.size())
        show_batch(batch_x)
        plt.axis('off')
        plt.show()
