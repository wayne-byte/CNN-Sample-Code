# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:12:33 2020

@author: opgg
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import csv
import torchvision.transforms as transforms
import cv2

mnist_train = torchvision.datasets.MNIST(root='../../data', 
                                       train=True, 
                                       download=True)

mnist_test = torchvision.datasets.MNIST(root='../../data', 
                                       train=False, 
                                       download=True)

print('train set:', len(mnist_train))
print('test set:', len(mnist_test))

# 內插改變影像大小
trans = transforms.Resize((64, 64))

# 開啟輸出的 CSV 檔案
train_config_path = 'data_img/mnist_train.csv'
test_config_path = 'data_img/mnist_test.csv'

train_img_path = 'data_img/mnist_train/'
test_img_path = 'data_img/mnist_test/'

with open(train_config_path, 'w', newline='') as csvFile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvFile)
    for i,(img,label) in enumerate(mnist_train):
        img_path = "./" + train_img_path + str(i) + ".png"
        img_trans = trans(img)
        img_trans = np.asanyarray(img_trans)
        cv2.imwrite(img_path,img_trans)
        writer.writerow([img_path,str(label)])
        
with open(test_config_path, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    for i,(img,label) in enumerate(mnist_test):
        img_path = "./" + test_img_path + str(i) + ".png"
        img_trans = trans(img)
        img_trans = np.asanyarray(img_trans)
        cv2.imwrite(img_path,img_trans)
        writer.writerow([img_path,str(label)])


