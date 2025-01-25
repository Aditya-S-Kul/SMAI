import torch
import torch.nn as nn
import torch.nn.functional as F

# class CNNModel(nn.Module):
#     def __init__(self, task="classification"):

#         super(CNNModel, self).__init__()
        
#         self.task = task
        
#         # Define convolutional layers
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
#         # Define pooling layer
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Define fully connected layers
#         self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Adjust input size based on output of conv layers
#         self.fc2 = nn.Linear(256, 64)
        
#         # Output layer
#         if task == "classification":
#             self.output = nn.Linear(64, 10)  # Assuming 10 classes for classification
#         elif task == "regression":
#             self.output = nn.Linear(64, 1)   # Single output for regression
        
#     def forward(self, x):
#         """
#         Forward pass of the model.
        
#         Parameters:
#         x (Tensor): Input tensor of shape (batch_size, 1, 128, 128).
        
#         Returns:
#         Tensor: Output tensor of the network, adapted to classification or regression task.
#         """
#         # Pass through convolutional layers with activation and pooling
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
        
#         # Flatten the tensor
#         x = x.view(-1, 128 * 16 * 16)
        
#         # Fully connected layers with ReLU activation
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
        
#         # Output layer: adjust output based on task type
#         if self.task == "classification":
#             x = self.output(x)
#         elif self.task == "regression":
#             x = self.output(x)
        
#         return x





# class CNNModel(nn.Module):
#     def __init__(self, task="classification", dropout_rate=0.5, num_conv_layers=3):
#         super(CNNModel, self).__init__()
        
#         self.task = task
        
#         # Define convolutional layers
#         self.conv_layers = nn.ModuleList()
#         in_channels = 1  # Starting with grayscale images
        
#         for i in range(num_conv_layers):
#             out_channels = 32 * (2 ** i)  # Example: 32, 64, 128, etc.
#             self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#             in_channels = out_channels
        
#         # Define pooling layer
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Define fully connected layers
#         self.fc1 = nn.Linear(self._get_flattened_size(128, num_conv_layers), 256)
#         self.fc2 = nn.Linear(256, 64)
        
#         # Dropout layer
#         self.dropout = nn.Dropout(dropout_rate)
        
#         # Output layer
#         if task == "classification":
#             self.output = nn.Linear(64, 4)  # 4 classes for classification (0, 1, 2, 3)
#         elif task == "regression":
#             self.output = nn.Linear(64, 1)   # Single output for regression
        
#     def _get_flattened_size(self, input_size, num_conv_layers):
#         """
#         Calculate the output size after convolutional layers and pooling.
#         """
#         for _ in range(num_conv_layers):
#             # Each Conv2D layer followed by MaxPool2D reduces the spatial size by half
#             input_size = input_size // 2
#         return 32 * (2 ** (num_conv_layers - 1)) * (input_size * input_size)  # 32 is the output channels of the last layer

#     def forward(self, x):
#         # Pass through convolutional layers with activation and pooling
#         for conv in self.conv_layers:
#             x = self.pool(F.relu(conv(x)))
        
#         # Flatten the tensor
#         x = x.view(x.size(0), -1)  # Correct flattening, using batch size
#         # Fully connected layers with ReLU activation and dropout
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)  # Apply dropout
#         x = F.relu(self.fc2(x))
        
#         # Output layer
#         x = self.output(x)
        
#         return x

class CNNModel(nn.Module):
    def __init__(self, task="classification", dropout_rate=0.5, num_conv_layers=3):
        """
        Parameters:
        - task: str, type of task ("classification" or "regression").
        - dropout_rate: float, dropout rate for regularization.
        - num_conv_layers: int, number of convolutional layers.
        """
        super(CNNModel, self).__init__()
        
        self.task = task
        
        # Define convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # Starting with grayscale images
        
        for i in range(num_conv_layers):
            out_channels = 32 * (2 ** i)  # Example: 32, 64, 128, etc.
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels
        
        # Define pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the flattened size after convolutional layers
        flattened_size = self._get_flattened_size(128, num_conv_layers)
        
        # Define fully connected layers with hardcoded sizes
        self.fc_layers = nn.ModuleList([
            nn.Linear(flattened_size, 256),  # Fixed size for first FC layer
            nn.Linear(256, 64)                # Fixed size for second FC layer
        ])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        if task == "classification":
            self.output = nn.Linear(64, 4)  # 4 classes for classification
        elif task == "regression":
            self.output = nn.Linear(64, 1)  # Single output for regression
        
    def _get_flattened_size(self, input_size, num_conv_layers):
        """
        Calculate the output size after convolutional layers and pooling.
        """
        for _ in range(num_conv_layers):
            # Each Conv2D layer followed by MaxPool2D reduces the spatial size by half
            input_size = input_size // 2
        return 32 * (2 ** (num_conv_layers - 1)) * (input_size * input_size)

    def forward(self, x):
        # Pass through convolutional layers with activation and pooling
        for conv in self.conv_layers:
            x = self.pool(F.relu(conv(x)))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers with ReLU and dropout
        for fc in self.fc_layers:
            x = F.relu(fc(x))
            x = self.dropout(x)  # Apply dropout after each FC layer
        
        # Output layer
        x = self.output(x)
        
        return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CNNModel(nn.Module):
#     def __init__(self, task="classification", dropout_rate=0.5, num_conv_layers=3):
#         """
#         Parameters:
#         - task: str, type of task ("classification" or "regression").
#         - dropout_rate: float, dropout rate for regularization.
#         - num_conv_layers: int, number of convolutional layers.
#         """
#         super(CNNModel, self).__init__()
        
#         self.task = task
        
#         # Define convolutional layers
#         self.conv_layers = nn.ModuleList()
#         in_channels = 1  # Starting with grayscale images
        
#         for i in range(num_conv_layers):
#             out_channels = 32 * (2 ** i)  # Example: 32, 64, 128, etc.
#             self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#             in_channels = out_channels
        
#         # Define pooling layer
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Calculate the flattened size after convolutional layers
#         flattened_size = self._get_flattened_size(128, num_conv_layers)
        
#         # Define fully connected layers with hardcoded sizes
#         self.fc_layers = nn.ModuleList([
#             nn.Linear(flattened_size, 256),  # Fixed size for first FC layer
#             nn.Linear(256, 64)                # Fixed size for second FC layer
#         ])
        
#         # Dropout layer
#         self.dropout = nn.Dropout(dropout_rate)
        
#         # Output layer
#         if task == "classification":
#             self.output = nn.Linear(64, 4)  # 4 classes for classification
#         elif task == "regression":
#             self.output = nn.Linear(64, 1)  # Single output for regression
        
#     def _get_flattened_size(self, input_size, num_conv_layers):
#         """
#         Calculate the output size after convolutional layers and pooling.
#         """
#         for _ in range(num_conv_layers):
#             # Each Conv2D layer followed by MaxPool2D reduces the spatial size by half
#             input_size = input_size // 2
#         return 32 * (2 ** (num_conv_layers - 1)) * (input_size * input_size)

#     def forward(self, x):
#         feature_maps = []  # List to hold feature maps
#         # Pass through convolutional layers with activation and pooling
#         for conv in self.conv_layers:
#             x = F.relu(conv(x))
#             feature_maps.append(x)  # Capture feature map after activation
#             x = self.pool(x)

#         # Flatten the tensor
#         x = x.view(x.size(0), -1)
        
#         # Pass through fully connected layers with ReLU and dropout
#         for fc in self.fc_layers:
#             x = F.relu(fc(x))
#             x = self.dropout(x)  # Apply dropout after each FC layer
        
#         # Output layer
#         x = self.output(x)
        
#         return x, feature_maps  # Return output and feature maps

# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt

# class CNNModel(nn.Module):
#     def __init__(self, task="classification"):
#         super(CNNModel, self).__init__()
#         self.task = task
        
#         # Define layers
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc = nn.Linear(64 * 16 * 16, 10 if task == "classification" else 1)  # Example output size
        
#         self.activation = nn.ReLU()

#     def forward(self, x, return_features=False):
#         # Capture feature maps
#         feature_maps = []

#         # First block
#         x = self.conv1(x)
#         x = self.activation(x)
#         x = self.pool(x)
#         if return_features:
#             feature_maps.append(x)
        
#         # Second block
#         x = self.conv2(x)
#         x = self.activation(x)
#         x = self.pool(x)
#         if return_features:
#             feature_maps.append(x)
        
#         # Third block
#         x = self.conv3(x)
#         x = self.activation(x)
#         x = self.pool(x)
#         if return_features:
#             feature_maps.append(x)
        
#         # Flatten for fully connected layer
#         x = x.view(x.size(0), -1)
#         output = self.fc(x)
        
#         # Return output and optionally feature maps
#         return (output, feature_maps) if return_features else output
