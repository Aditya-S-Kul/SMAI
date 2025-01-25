import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt # type:ignore

# Constants for image dimensions, model hyperparameters, and dataset constraints
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 64
BATCH_SIZE = 32
MAX_WORD_LENGTH = 30  # Maximum length of words to consider
HIDDEN_SIZE = 256
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

# Character mappings for encoding and decoding words
ALL_CHARS = "abcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(ALL_CHARS)}  # Map characters to indices
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(ALL_CHARS)}  # Reverse map indices to characters
VOCAB_SIZE = len(CHAR_TO_IDX) + 1  # Include an additional index for padding

class OCRDataset(Dataset):
    """Custom Dataset class for loading OCR word images and labels."""
    def __init__(self, image_dir, word_list_path, transform=None):
        self.image_dir = image_dir
        with open(word_list_path, 'r') as f:
            self.labels = []
            for line in f:
                word = line.strip()
                if len(word) <= MAX_WORD_LENGTH:  # Filter words exceeding maximum length
                    self.labels.append(word)
        self.labels = self.labels[:100000]  # Limit dataset size for training efficiency
        self.transform = transform

    def __len__(self):
        return len(self.labels)  # Return dataset size

    def __getitem__(self, idx):
        # Load the image corresponding to the word index
        img_path = os.path.join(self.image_dir, f"word_{idx}.png")
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        # Convert word label to indices using CHAR_TO_IDX
        word = self.labels[idx].lower()
        label = torch.zeros(MAX_WORD_LENGTH, dtype=torch.long)  # Initialize label tensor
        for i, char in enumerate(word):
            if char in CHAR_TO_IDX:
                label[i] = CHAR_TO_IDX[char]
        
        return image, label, len(word)

class CNNEncoder(nn.Module):
    """Convolutional Neural Network for feature extraction from images."""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # First convolutional layer
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downsample by 2x

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),  # Further downsample

            nn.Dropout2d(DROPOUT_RATE)  # Regularization
        )
        self.feature_size = 256 * 8  # Calculated output feature size after CNN
        self.sequence_length = 32  # Width dimension of the output feature map

    def forward(self, x):
        x = self.features(x)  # Extract features
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)  # Rearrange dimensions to (batch, width, channels, height)
        x = x.contiguous().view(batch_size, self.sequence_length, -1)  # Flatten height dimension
        x = x[:, :MAX_WORD_LENGTH, :]  # Ensure the sequence length doesn't exceed max word length
        return x

class RNNDecoder(nn.Module):
    """Recurrent Neural Network for decoding sequence features into character probabilities."""
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNDecoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,  # Input from CNNEncoder
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,  # Bidirectional for better context learning
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Map LSTM output to character predictions
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x, _ = self.lstm(x)  # Sequence output from LSTM
        x = self.dropout(x)  # Apply dropout for regularization
        x = self.fc(x)  # Map to vocabulary size
        return x

class OCRModel(nn.Module):
    """OCR Model combining CNN Encoder and RNN Decoder."""
    def __init__(self):
        super(OCRModel, self).__init__()
        self.encoder = CNNEncoder()
        self.decoder = RNNDecoder(
            input_size=256 * 8,  # Matches encoder output size
            hidden_size=HIDDEN_SIZE,
            output_size=VOCAB_SIZE
        )

    def forward(self, x):
        x = self.encoder(x)  # Extract features from CNN
        x = self.decoder(x)  # Decode features to character probabilities
        return x

def train_model(model, train_loader, criterion, optimizer, device, epoch):
    """Training loop for the OCR model."""
    model.train()
    total_loss = 0
    batch_losses = []
    
    for batch_idx, (images, labels, lengths) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()  # Clear gradients
        outputs = model(images)  # Forward pass
        
        loss = 0
        for t in range(MAX_WORD_LENGTH):  # Calculate loss for each time step
            loss += criterion(outputs[:, t, :], labels[:, t])
        
        loss = loss / MAX_WORD_LENGTH  # Average loss over time steps
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        total_loss += loss.item()
        batch_losses.append(loss.item())
        
        if batch_idx % 50 == 0:  # Log progress every 50 batches
            print(f'Epoch: {epoch+1}, Batch: {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader), batch_losses
