import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import sys
import os
import librosa
import librosa.display
from hmmlearn import hmm # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore")



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from performance_measures.metrics import Metrics
from performance_measures import metrics as metrics1
from models.kde.kde import KDE
from models.gmm.gmm import GMM






# Q.2 KDE

# Parameters for the two circles
large_circle_points = 3000
small_circle_points = 500
large_circle_radius = 2
small_circle_radius = 0.22
noise_level_large = 0.2  # Adjust noise level for the large circle
noise_level_small = 0.05  # Adjust noise level for the small circle

# Generate points for the large diffused circle
angles_large = np.random.uniform(0, 2 * np.pi, large_circle_points)
radii_large = large_circle_radius * np.sqrt(np.random.uniform(0, 1, large_circle_points))
x_large = radii_large * np.cos(angles_large)
y_large = radii_large * np.sin(angles_large)

# Add Gaussian noise to blur the boundary
x_large += np.random.normal(0, noise_level_large, large_circle_points)
y_large += np.random.normal(0, noise_level_large, large_circle_points)
large_circle_data = np.vstack((x_large, y_large)).T

# Generate points for the small dense circle, shifted to (0.5, 0.5)
angles_small = np.random.uniform(0, 2 * np.pi, small_circle_points)
radii_small = small_circle_radius * np.sqrt(np.random.uniform(0, 1, small_circle_points))
x_small = radii_small * np.cos(angles_small) + 1  # Shift in x
y_small = radii_small * np.sin(angles_small) + 1  # Shift in y

# Add Gaussian noise to blur the boundary of the small circle
x_small += np.random.normal(0, noise_level_small, small_circle_points)
y_small += np.random.normal(0, noise_level_small, small_circle_points)
small_circle_data = np.vstack((x_small, y_small)).T

# Combine the datasets
data = np.vstack((large_circle_data, small_circle_data))

# Plot the data to verify
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], s=1, color='black')
plt.title("Original Data with Blurred Boundaries")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.grid(True)
plt.show()

# Fit and visualize the KDE
kde = KDE(kernel='gaussian', bandwidth=0.3)
kde.fit(data)
kde.visualize(num_contours=15)




k = 2
gmm = GMM(n_components=k)
gmm.fit(data)

# Get membership and densities
membership = gmm.getMembership()
densities = np.sum(membership, axis=1)

# Create meshgrid for plotting
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
points = np.vstack([X.ravel(), Y.ravel()]).T

# Calculate density for each grid point
Z = np.zeros(X.shape)

# Loop through each component and calculate the contribution to the density
for i in range(k):
    mean = gmm.means_[i]
    cov = gmm.covariances_[i]
    weight = gmm.weights_[i]
    
    # Calculate Gaussian PDF for each point in the grid
    diff = points - mean
    inv_cov = np.linalg.inv(cov)
    norm_const = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
    
    # Calculate the probability density for each point in the grid
    pdf_values = weight * norm_const * np.exp(exponent)
    
    # Sum the densities from all components
    Z += pdf_values.reshape(X.shape)

# Plot the contour plot
plt.figure(figsize=(6, 6))

# Create filled contour plot for densities
contour_plot = plt.contourf(X, Y, Z, levels=50, cmap="coolwarm")
plt.scatter(data[:, 0], data[:, 1], s=1, color="black", alpha=0.5)
plt.colorbar(contour_plot, label="Density")

# Create contour lines to represent constant densities
contour_levels = np.linspace(Z.min(), Z.max(), 10)
plt.contour(X, Y, Z, levels=contour_levels, colors='black', linewidths=1)
# plt.scatter(data[:, 0], data[:, 1], s=1, color="black", alpha=0.5)

# Set title and labels
plt.title(f"GMM Density Estimate {k} components")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()



















# Q.3 HMM

# import joblib

def split_data():
    import os
    import shutil
    import random

    data_dir = '../../data/interim/5/Free-Spoken-Digit-Dataset/recordings/'
    train_dir = '../../data/interim/5/Free-Spoken-Digit-Dataset/train/'
    test_dir = '../../data/interim/5/Free-Spoken-Digit-Dataset/test/'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def split_data(digit, data_dir, train_dir, test_dir, train_ratio=0.8):
        digit_files = [f for f in os.listdir(data_dir) if f.startswith(f"{digit}_") and f.endswith('.wav')]
        random.shuffle(digit_files)
        train_size = int(len(digit_files) * train_ratio)
        train_files = digit_files[:train_size]
        test_files = digit_files[train_size:]
        os.makedirs(os.path.join(train_dir, str(digit)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, str(digit)), exist_ok=True)
        
        for file in train_files:
            shutil.move(os.path.join(data_dir, file), os.path.join(train_dir, str(digit), file))
        
        for file in test_files:
            shutil.move(os.path.join(data_dir, file), os.path.join(test_dir, str(digit), file))
        
        print(f"Digit {digit}: {len(train_files)} files in train and {len(test_files)} files in test.")

    for digit in range(10):
        split_data(digit, data_dir, train_dir, test_dir)

# split_data()

train_dir = '../../data/interim/5/Free-Spoken-Digit-Dataset/train/'
test_dir = '../../data/interim/5/Free-Spoken-Digit-Dataset/test/'

# Verify the correct number of files in each digit folder
for digit in range(10):
    digit_path = os.path.join(train_dir, str(digit))
    files = [f for f in os.listdir(digit_path) if f.endswith('.wav')]
    # print(f"Digit {digit}: {len(files)} files found in {digit_path}")

# Function to extract MFCC features
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T  # Transpose so time is the first dimension

# Function to visualize MFCC features
def visualize_mfcc(mfcc):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc.T, x_axis='time', cmap='coolwarm')
    plt.colorbar()
    plt.title('MFCC Features')
    # plt.savefig(path)
    plt.show()


file_path = os.path.join(train_dir, '0/0_george_0.wav')
mfcc_features = extract_mfcc(file_path)
visualize_mfcc(mfcc_features)

# for i in range(10):
#     file_path = os.path.join(train_dir, str(i)+'/'+str(i)+'_george_'+str(i)+'.wav')
#     mfcc_features = extract_mfcc(file_path)
#     visualize_mfcc(mfcc_features, 'figures/'+str(i)+'_george_'+str(i)+'.png')



# Function to load data for a specific digit
def load_digit_data(digit, data_dir):
    digit_path = os.path.join(data_dir, str(digit)) 
    files = [os.path.join(digit_path, f) for f in os.listdir(digit_path) if f.endswith('.wav')]
    # print(f"Files for digit {digit}: {files}")  # Debugging print
    mfcc_data = [extract_mfcc(file) for file in files]
    return mfcc_data

# Prepare training data
X_train = []  # To store MFCC features
y_train = []  # To store labels (digits 0-9)
for digit in range(10):
    digit_data = load_digit_data(digit, train_dir)
    X_train.extend(digit_data)
    y_train.extend([digit] * len(digit_data))

# Function to train HMM for a specific digit
def train_hmm_for_digit(mfcc_data):
    if not mfcc_data:
        raise ValueError("No data provided for HMM training.")
    model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000)
    X = np.concatenate(mfcc_data, axis=0)
    lengths = [len(mfcc) for mfcc in mfcc_data]  # Lengths of individual sequences
    model.fit(X, lengths)
    return model


hmm

# Train separate HMM models for each digit
hmm_models = {}
for digit in range(10):
    digit_data = [X_train[i] for i in range(len(y_train)) if y_train[i] == digit]
    hmm_models[digit] = train_hmm_for_digit(digit_data)
    # if digit_data:  # Only train if there is data
    #     hmm_models[digit] = train_hmm_for_digit(digit_data)
    # else:
    #     print(f"No training data for digit {digit}")

# # Save the trained models
# joblib.dump(hmm_models, 'hmm_digit_models.pkl')

# # Load the saved HMM models for prediction
# hmm_models = joblib.load('hmm_digit_models.pkl')

# Predict the digit based on MFCC features
def predict_digit(mfcc_features):
    log_likelihoods = []
    for digit, model in hmm_models.items():
        log_likelihood = model.score(mfcc_features)
        log_likelihoods.append(log_likelihood)
    predicted_digit = np.argmax(log_likelihoods)
    return predicted_digit

# Evaluate the model on test data
correct_predictions = 0
total_predictions = 0

for digit in range(10):
    test_files = [os.path.join(test_dir, str(digit), f) for f in os.listdir(os.path.join(test_dir, str(digit))) if f.endswith('.wav')]
    for file in test_files:
        mfcc_features = extract_mfcc(file)
        predicted_digit = predict_digit(mfcc_features)
        if predicted_digit == digit:
            correct_predictions += 1
        total_predictions += 1

# Compute and print accuracy
accuracy = correct_predictions / total_predictions
print(f'Accuracy on test set: {accuracy * 100:.2f}%')




# Evaluate the model on my data
correct_predictions = 0
total_predictions = 0
my_rec_dir = '../../data/interim/5/Free-Spoken-Digit-Dataset/my_recordings/'
for digit in range(10):
    test_files = [os.path.join(my_rec_dir, str(digit), f) for f in os.listdir(os.path.join(my_rec_dir, str(digit))) if f.endswith('.wav')]
    for file in test_files:
        # print(file)
        mfcc_features = extract_mfcc(file)
        predicted_digit = predict_digit(mfcc_features)
        # print(f'digit: {digit}')
        # print(f'predicted_digit: {predicted_digit}')
        if predicted_digit == digit:
            correct_predictions += 1
        total_predictions += 1

# Compute and print accuracy
accuracy = correct_predictions / total_predictions
print(f'Accuracy on my recordings: {accuracy * 100:.2f}%')
















# Q. 4 RNN
# Q. 4.1 Counting bits


# Task 1: Dataset
class BitSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)


# Generate dataset
def generate_dataset(num_samples, min_len=1, max_len=16):
    sequences = []
    labels = []
    for _ in range(num_samples):
        length = np.random.randint(min_len, max_len + 1)
        seq = np.random.choice([0, 1], size=(length,))
        sequences.append(seq)
        labels.append(seq.sum())
    return sequences, labels


# Split dataset
num_samples = 100_000
sequences, labels = generate_dataset(num_samples)

# for i in range(0,10000, 1000):
#     print(f'seq: {sequences[i]}, label: {labels[i]}')  

split_ratios = [0.8, 0.1, 0.1]
split1 = int(split_ratios[0] * num_samples)
split2 = split1 + int(split_ratios[1] * num_samples)

train_sequences, train_labels = sequences[:split1], labels[:split1]
val_sequences, val_labels = sequences[split1:split2], labels[split1:split2]
test_sequences, test_labels = sequences[split2:], labels[split2:]

train_dataset = BitSequenceDataset(train_sequences, train_labels)
val_dataset = BitSequenceDataset(val_sequences, val_labels)
test_dataset = BitSequenceDataset(test_sequences, test_labels)


# Custom collate function to pad sequences and create masks
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    # Create mask to ignore padding during processing
    mask = (padded_sequences != 0).float()
    return padded_sequences, labels, mask


# Update DataLoaders with the collate_fn
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Updated RNN Architecture
class CountingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CountingRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, mask):
        x = x.unsqueeze(-1)  # Shape: (batch_size, seq_len, input_size)
        out, _ = self.rnn(x)
        # Mask out padding effects: weighted sum of hidden states
        out = out * mask.unsqueeze(-1)
        # Avoid division by zero by adding a small epsilon
        mask_sum = mask.sum(dim=1, keepdim=True)
        mask_sum[mask_sum == 0] = 1e-8  # Ensure no division by zero
        out = out.sum(dim=1) / mask_sum
        out = self.fc(out)
        return out


# Gradient Clipping in Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, clip_value=1.0):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0
        for sequences, labels, mask in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences, mask)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels, mask in val_loader:
                outputs = model(sequences, mask)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() * sequences.size(0)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


# Instantiate and initialize model
input_size = 1
hidden_size = 16
output_size = 1
model = CountingRNN(input_size, hidden_size, output_size)

# Training setup
criterion = nn.L1Loss()  # Mean Absolute Error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train model
num_epochs = 10
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Plot training and validation losses
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Task 4: Generalization with Random Baseline Comparison
def evaluate_generalization_with_baseline(model, criterion):
    lengths = range(1, 33)
    mae_per_length_model = []
    mae_per_length_baseline = []

    for length in lengths:
        # Generate dataset for a specific sequence length
        sequences, labels = generate_dataset(1000, min_len=length, max_len=length)
        dataset = BitSequenceDataset(sequences, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # Evaluate model's MAE
        total_loss_model = 0
        total_loss_baseline = 0
        with torch.no_grad():
            for sequences, labels, mask in loader:
                # Model predictions
                outputs = model(sequences, mask)
                loss_model = criterion(outputs.squeeze(), labels)
                total_loss_model += loss_model.item() * sequences.size(0)

                # Random baseline predictions
                random_preds = torch.randint(0, length + 1, labels.shape, dtype=torch.float32)
                loss_baseline = criterion(random_preds, labels)
                total_loss_baseline += loss_baseline.item() * sequences.size(0)

        total_loss_model /= len(loader.dataset)
        total_loss_baseline /= len(loader.dataset)

        mae_per_length_model.append(total_loss_model)
        mae_per_length_baseline.append(total_loss_baseline)

    return lengths, mae_per_length_model, mae_per_length_baseline


# Evaluate generalization performance
lengths, mae_per_length_model, mae_per_length_baseline = evaluate_generalization_with_baseline(model, criterion)

print('Model Loss:')
for digit, loss in enumerate(mae_per_length_model):
    print(f'{digit + 1}: {loss:.4f}')
print('Baseline Loss:')
for digit, loss in enumerate(mae_per_length_baseline):
    print(f'{digit + 1}: {loss:.4f}')

# Plot MAE vs Sequence Length for Model and Random Baseline
plt.plot(lengths, mae_per_length_model, marker='o', label='Model MAE')
plt.plot(lengths, mae_per_length_baseline, marker='x', label='Random Baseline MAE', linestyle='--')
plt.xlabel('Sequence Length')
plt.ylabel('MAE')
plt.title('Generalization Performance: Model vs Random Baseline')
plt.legend()
plt.show()