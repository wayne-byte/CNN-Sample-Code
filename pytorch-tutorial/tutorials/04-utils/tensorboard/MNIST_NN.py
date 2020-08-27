# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:59:51 2020

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


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet().to(device)

logger = Logger('./logs')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


# Train the model
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.view(images.size(0), -1).to(device), labels.to(device)
        
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
        images, labels = images.view(images.size(0), -1).to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'NeuralNet.ckpt')