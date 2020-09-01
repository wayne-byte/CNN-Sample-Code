# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:16:28 2020

@author: opgg
"""

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import csv

def default_loader(path):
    return Image.open(path).convert('L')

class MyDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        with open(root, newline='') as csvFile:
            rows = csv.reader(csvFile)
            for row in rows:
                imgs.append((row[0],int(row[1])))
            
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)
    
    