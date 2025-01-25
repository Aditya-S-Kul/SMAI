import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import hamming_loss

class MultiLabelCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MultiLabelCNN, self).__init__()
        
        # First Convolutional Block
        # Input: 128x128 -> Output: 64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Second Convolutional Block
        # Input: 64x64 -> Output: 32x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Third Convolutional Block
        # Input: 32x32 -> Output: 16x16
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Fourth Convolutional Block
        # Input: 16x16 -> Output: 8x8
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Fully connected layers
        # After conv4: 256 * 8 * 8 = 16384 features
        self.fc = nn.Sequential(
            nn.Linear(16384, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten: 256 * 8 * 8 = 16384
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
