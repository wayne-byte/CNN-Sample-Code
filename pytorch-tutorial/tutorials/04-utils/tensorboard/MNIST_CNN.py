# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:44:50 2020

@author: opgg
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from logger import Logger

# Hyper parameters
num_epochs = 16
num_classes = 10
batch_size = 100
learning_rate = 1e-5

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset 
dataset = torchvision.datasets.MNIST(root='../../data', 
                                     train=True, 
                                     transform=transforms.ToTensor(),  
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet().to(device)

logger = Logger('./logs2')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)

n_points = len(dataset)
total_step = int(num_epochs * n_points/batch_size)

# Train the model
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Compute accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()
    
        if (i+1) % 100 == 0:
            print ('epoch [{}/{}], Loss: {:.4f}, Acc: {:.2f}' 
                   .format(epoch+1, num_epochs, loss.item(), accuracy.item()))
    
            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #
    
            # 1. Log scalar values (scalar summary)
            info = { 'loss': loss.item(), 'accuracy': accuracy.item() }
    
            for tag, value in info.items():
                logger.scalar_summary(tag, value, i+1)
    
            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), i+1)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), i+1)
    
            # 3. Log training images (image summary)
            info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }
    
            for tag, images in info.items():
                logger.image_summary(tag, images, i+1)
            

### testing part
test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'CNN.ckpt')


