# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:44:50 2020

@author: opgg
"""

import torch
import torch.nn as nn
from torchvision import transforms
from logger import Logger
from dataset import MyDataset
import tqdm


# Hyper parameters
num_epochs = 16
num_classes = 10
batch_size = 100
learning_rate = 1e-5

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_config_path = 'data_img/mnist_train.csv'
train_data = MyDataset(root=train_config_path, transform=transforms.ToTensor())

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                          batch_size=batch_size, 
                                          shuffle=True)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=6, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=1),
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

n_points = len(train_data)
steps_per_epoch = len(data_loader)


for epoch in range(num_epochs):
    # progress bar
    with tqdm.tqdm(total=steps_per_epoch) as pbar:
        pbar.set_description('epoch [{}/{}]'.format(epoch+1, num_epochs))
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
                pbar.update(100)
                pbar.set_postfix(Loss= loss.item(), Acc= accuracy.item())
        
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
                info = { 'images': images.view(-1, 64, 64)[:10].cpu().numpy() }
        
                for tag, images in info.items():
                    logger.image_summary(tag, images, i+1)
            

### testing part

train_config_path = 'data_img/mnist_test.csv'

test_data = MyDataset(root=train_config_path, transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_data,
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


