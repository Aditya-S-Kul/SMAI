import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from performance_measures.metrics import Metrics
from models.MLP.MLP import MLP_Classifier
from performance_measures import metrics as metrics1
from models.MLP.MLP import MLP_Regression

from models.AutoEncoders.AutoEncoders import AutoEncoder
from models.knn.knn import KNN



file_path1 = '../../data/interim/3/WineQT.csv'
df = pd.read_csv(file_path1)




# print(df.head())  
# print(df.info())  

mean_values = df.mean()
std_values = df.std()
min_values = df.min()
max_values = df.max()

summary_stats = pd.DataFrame({
    'Mean': mean_values,
    'Standard Deviation': std_values,
    'Min': min_values,
    'Max': max_values
})

print(summary_stats)
 


# Get the count of each unique value in the 'quality' column
quality_counts = df['quality'].value_counts()

# Plot the distribution of wine quality
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts.values, color='blue')
plt.title('Distribution of Wine Quality Labels')
plt.xlabel('Wine Quality')
plt.ylabel('Count')
plt.xticks(quality_counts.index)  # Set x-axis labels to the unique values in the 'quality' column
# plt.savefig('figures/WineQt_label_distribution.png') 
plt.show()





from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def data_processing():

    file_path1 = '../../data/interim/3/WineQT.csv'
    df = pd.read_csv(file_path1)
    df_features = df.drop(columns=['Id', 'quality'])

    # 1. Handle missing data (if any)
    # Check for missing values
    print(df_features.isnull().sum())

    # Check if there are any missing values in the dataset
    missing_values = df_features.isnull().sum().sum()

    # If there are missing values, apply imputation
    if missing_values > 0:
        print(f"Missing values detected: {missing_values}. Imputing missing values using the mean.")
        # Create the imputer with the 'mean' strategy
        imputer = SimpleImputer(strategy='mean')
        # Apply imputation and return the imputed DataFrame
        df_imputed = pd.DataFrame(imputer.fit_transform(df_features), columns=df_features.columns)
        
    else:
        print("No missing values detected.")
        # No imputation needed, so use the original features DataFrame
        df_imputed = df_features



    # 2. Normalize the feature data (scale between 0 and 1)
    min_max_scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(min_max_scaler.fit_transform(df_imputed), columns=df_imputed.columns)

    # 3. Standardize the feature data (mean = 0, std = 1)
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

    # Print the first few rows of normalized and standardized data
    print("Normalized Data (first 5 rows):\n", df_normalized.head())
    print("Standardized Data (first 5 rows):\n", df_standardized.head())


    # 4. Save both normalized and standardized datasets as CSV files
    df_normalized.to_csv('../../data/interim/3/WineQT_normalised.csv', index=False)
    df_standardized.to_csv('../../data/interim/3/WineQT_standardised.csv', index=False)


# data_processing()




file_path1_1 = '../../data/interim/3/WineQT_normalised.csv'
file_path1_2 = '../../data/interim/3/WineQT_standardised.csv'
df_normalized = pd.read_csv(file_path1_1)
df_standardized = pd.read_csv(file_path1_2)
# print("Normalized Data (first 5 rows):\n", df_normalized.head())
# print("Standardized Data (first 5 rows):\n", df_standardized.head())
# df_normalized.info()
# df_standardized.info()





# def accuracy_score(y_true, y_pred):
#     correct_predictions = np.sum(y_true == y_pred)  # Count correct predictions
#     total_predictions = len(y_true)  # Total number of predictions
#     accuracy = correct_predictions / total_predictions  # Calculate accuracy
#     return accuracy

def accuracy_score(y_true, y_pred):
    y_true_labels = np.argmax(y_true, axis=1)  # True class labels
    y_pred_labels = np.argmax(y_pred, axis=1)  # Predicted class labels
    
    # y_true_labels = y_true.flatten() # True class labels
    # y_pred_labels = y_pred.flatten() # Predicted class labels
    
    correct_predictions = np.sum(y_true_labels == y_pred_labels)
    total_predictions = len(y_true) 
    accuracy = correct_predictions / total_predictions  # Calculate accuracy
    return accuracy


X = df_standardized.values  # Get the standardized feature values as NumPy array
# X = df_normalized.values  
y = df['quality'].values    # Extract 'quality' as target values
# y = y.astype(float).reshape(-1, 1)
y = pd.get_dummies(df['quality']).values
y = y.astype(float)
# print(y[:10])
# Define the split sizes
train_size = 0.7
val_size = 0.2
test_size = 0.1
# 16, 30, 45,9,2,12,3,20,21,33,39,40,47,48
# Shuffle the data
# np.random.seed(41)

np.random.seed(16)

indices = np.random.permutation(len(X))

# Compute the split indices
train_end = int(train_size * len(X))
val_end = train_end + int(val_size * len(X))

# Split the data
X_train, X_val, X_test = X[indices[:train_end]], X[indices[train_end:val_end]], X[indices[val_end:]]
y_train, y_val, y_test = y[indices[:train_end]], y[indices[train_end:val_end]], y[indices[val_end:]]

input_size = X_train.shape[1]
# print(input_size)  # Number of features (columns in X)
# hidden_layers = [30, 30, 1] 
hidden_layers = [20, 19] 
# hidden_layers = [20, 16] 
# hidden_layers = [12, 12, 5, 5]
# hidden_layers = [16, 16] 
# output_size = len(np.unique(y))  # Number of unique classes in the target (e.g., if classification)
output_size = y.shape[1]
# output_size = 1




mlp = MLP_Classifier(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size, learning_rate=0.01, activation='relu', optimizer='sgd', epochs=100, task = 'classification',  early_stopping = True)

mlp.fit(X_train, y_train, X_val, y_val)
# print(y_train[:10])
y_pred_train = np.round(mlp.predict(X_train) ) # Predictions on the training set
# print(y_pred_train[:10])
y_pred_val = np.round(mlp.predict(X_val) )   # Predictions on the val set

y_pred_test = np.round(mlp.predict(X_test) )   # Predictions on the test set
train_accuracy = accuracy_score(y_train, y_pred_train)  # Round predicted values if necessary
val_accuracy = accuracy_score(y_val, y_pred_val)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Step 8: Print accuracy
print(train_accuracy)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


y_test = np.argmax(y_test, axis=1)  
y_pred_test = np.argmax(y_pred_test, axis=1) 

metrics = Metrics(y_test, y_pred_test)
accuracy = metrics.accuracy()

print(f"Test Accuracy: {accuracy:.4f}")

# Calculate accuracy, precision, recall, and F1 scores
macro_precision = metrics.macro_precision()
macro_recall = metrics.macro_recall()
macro_f1 = metrics.macro_f1()

micro_precision = metrics.micro_precision()
micro_recall = metrics.micro_recall()
micro_f1 = metrics.micro_f1()

# Print results
print(f"Test Macro Precision: {macro_precision:.4f}")
print(f"Test Macro Recall: {macro_recall:.4f}")
print(f"Test Macro F1 Score: {macro_f1:.4f}")
print(f"Test Micro Precision: {micro_precision:.4f}")
print(f"Test Micro Recall: {micro_recall:.4f}")
print(f"Test Micro F1 Score: {micro_f1:.4f}")






# 2.7
# Dictionaries to store counts of correct and incorrect predictions for each label
correct_count = {}
incorrect_count = {}

for i in range(len(y_test)):
    true_label = y_test[i]
    pred_label = y_pred_test[i]
    
    # Initialize the count for each label if not already done
    if true_label not in correct_count:
        correct_count[true_label] = 0
        incorrect_count[true_label] = 0

    # Check if prediction is correct or incorrect
    if true_label == pred_label:
        correct_count[true_label] += 1
    else:
        incorrect_count[true_label] += 1

# Print the correct and incorrect predictions for each label
print("Correct and Incorrect Predictions per Label:\n")
for label in sorted(correct_count.keys()):
    print(f"Label {label}:")
    print(f"  Correct Predictions: {correct_count[label]}")
    print(f"  Incorrect Predictions: {incorrect_count[label]}\n")




# wandb.login(key='c354ec9456e466caac732ae8538316fe40e79ec3')

sweep_config = {
    'method': 'grid', 
    'name': 'MLP classifier hyperparameter tuning',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [0.01, 0.1]
        },
        'epochs': {
            'values': [50, 100, 200]
        },
        'activation': {
            'values': ['relu', 'sigmoid', 'tanh', 'linear']
        },
        'optimizer': {
            'values': ['sgd', 'mini_batch', 'batch']
        },
        'hidden_layers': {
            'values': [[5, 5], [10, 10], [20, 20], [30, 30], [50, 50], [20, 15], [30, 20], [25, 20], [60, 30], [20, 19], [7, 7, 7], [20, 20, 16], [12, 12, 8], [25, 20, 15], [16, 14, 12], [12, 12, 5, 5], [20, 16, 12, 10], [30, 30, 20, 10], [25, 20, 15, 10], [28, 23, 20, 15]]
        }
    }
}
# sweep_config = {
#     'method': 'grid', 
#     'name': 'Hyperparameter Tuning 2',
#     'metric': {
#         'name': 'val_accuracy',
#         'goal': 'maximize'   
#     },
#     'parameters': {
#         'learning_rate': {
#             'values': [0.01]
#         },
#         'epochs': {
#             'values': [100]
#         },
#         'activation': {
#             'values': ['relu']
#         },
#         'optimizer': {
#             'values': ['sgd', 'mini_batch', 'batch']
#         },
#         'hidden_layers': {
#             'values': [[20, 19], [30, 30, 1]]
#         }
#     }
# }


def train_model_sweep():
    wandb.init(project="MLP-Classifier-Tuning")
    # Set up your model with hyperparameters from W&B config
    config = wandb.config
    hidden_layers=config["hidden_layers"]
    learning_rate=config["learning_rate"]
    activation=config["activation"]
    optimizer=config["optimizer"]
    epochs=config["epochs"]
     
    run_name = f"Hl_{str(hidden_layers)}_lr_{learning_rate}_act_{activation}_opt_{optimizer}_epochs_{epochs}"
    wandb.run.name = run_name

    mlp = MLP_Classifier(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size, learning_rate=learning_rate, activation=activation, optimizer=optimizer, epochs=epochs, task = 'classification', early_stopping = True)


    mlp.fit(X_train, y_train, X_val, y_val, wb = True)
    y_pred_train = np.round(mlp.predict(X_train) ) 
    y_pred_val = np.round(mlp.predict(X_val) )  
    train_accuracy = accuracy_score(y_train, y_pred_train)  # Round predicted values if necessary
    val_accuracy = accuracy_score(y_val, y_pred_val)
    wandb.log({'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy})

# sweep_id = wandb.sweep(sweep_config, project="MLP-Classifier-Tuning")
# wandb.agent(sweep_id, train_model_sweep)








# # Perform gradient checking
# is_correct = mlp.gradient_check(X_train, y_train)

# if is_correct:
#     print("Gradients are correct. Proceeding with training.")
#     # mlp.fit(X_train, y_train)
# else:
#     print("Gradient check failed. Please review the implementation.")


# mlp = MLP_Classifier(input_size=input_size, hidden_layers=[5, 5], output_size=output_size)
# mlp.fit(X, y)
# mlp.gradient_checking(X[:5], y[:5])  # Use a small batch for gradient checking







2.5

activation_functions = ['relu', 'sigmoid', 'tanh', 'linear']  # List of activation functions
losses = {activation: [] for activation in activation_functions}  # To store losses for each activation function

for activation in activation_functions:
    mlp = MLP_Classifier(input_size=input_size, hidden_layers=[20,19],
                         output_size=output_size, learning_rate=0.01,
                         activation=activation, optimizer='sgd', epochs=100,
                         task='classification', early_stopping=False)
    
    # Fit the model and store losses
    loss = mlp.fit(X_train, y_train, X_val, y_val)
    losses[activation] = loss  # Store the loss values for plotting

# Plotting the losses
plt.figure(figsize=(12, 6))
for activation in activation_functions:
    plt.plot(losses[activation], label=activation)
plt.title('Effect of Activation Functions on Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.savefig('figures/non_linearity_plot.png')
plt.show()


learning_rates = [0.001, 0.01, 0.1, 1.0]  # List of learning rates
losses_lr = {lr: [] for lr in learning_rates}  # To store losses for each learning rate

for lr in learning_rates:
    mlp = MLP_Classifier(input_size=input_size, hidden_layers=[20,19],
                         output_size=output_size, learning_rate=lr,
                         activation='relu', optimizer='sgd', epochs=100,
                         task='classification', early_stopping=False)
    
    # Fit the model and store losses
    loss = mlp.fit(X_train, y_train, X_val, y_val)
    losses_lr[lr] = loss  # Store the loss values for plotting

# Plotting the losses
plt.figure(figsize=(12, 6))
for lr in learning_rates:
    plt.plot(losses_lr[lr], label=f'LR: {lr}')
plt.title('Effect of Learning Rate on Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.savefig('figures/learning_rate_plot.png')
plt.show()

batch_sizes = [8, 16, 32, 64]  # List of batch sizes
losses_bs = {bs: [] for bs in batch_sizes}  # To store losses for each batch size

for batch_size in batch_sizes:
    mlp = MLP_Classifier(input_size=input_size, hidden_layers=[20,19],
                         output_size=output_size, learning_rate=0.01,
                         activation='relu', optimizer='mini-batch', epochs=100,
                         task='classification', early_stopping=False)  
    mlp.set_batch_size(batch_size)
    # print(mlp.batch_size)
    # Fit the model and store losses
    loss = mlp.fit(X_train, y_train, X_val, y_val)
    losses_bs[batch_size] = loss  # Store the loss values for plotting

# Plotting the losses
plt.figure(figsize=(12, 6))
for bs in batch_sizes:
    plt.plot(losses_bs[bs], label=f'Batch Size: {bs}')
plt.title('Effect of Batch Size on Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.savefig('figures/batch_size_plot.png')
plt.show()




















































# 3 Regression

file_path2 = '../../data/interim/3/HousingData.csv'
df = pd.read_csv(file_path2)

# print(df.head())  
# print(df.info())  

mean_values = df.mean()
std_values = df.std()
min_values = df.min()
max_values = df.max()

summary_stats = pd.DataFrame({
    'Mean': mean_values,
    'Standard Deviation': std_values,
    'Min': min_values,
    'Max': max_values
})

print(summary_stats)
 
# Extract the target variable MEDV
medv = df['MEDV']

plt.figure(figsize=(8, 6))

# Plot the histogram of MEDV
plt.hist(medv, bins=30, edgecolor='black', color='skyblue', density=True, alpha=0.6, label='Histogram')

# Manual KDE-like approximation (using Gaussian smoothing)
# Generate an array of x-values for the KDE plot
x_values = np.linspace(medv.min(), medv.max(), 1000)

# Set the bandwidth for smoothing (controls how much to smooth the curve)
bandwidth = 1.0

# Calculate the KDE values manually by summing up Gaussian distributions centered on each data point
kde_values = np.zeros_like(x_values)
for point in medv:
    kde_values += np.exp(-0.5 * ((x_values - point) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))

# Normalize the KDE to make it a proper density
kde_values /= len(medv)

# Plot the KDE curve
plt.plot(x_values, kde_values, color='blue', label='KDE Approximation')

plt.title('Distribution and KDE Approximation of MEDV (Median Value of Homes)', fontsize=16)
plt.xlabel('MEDV ($1000)', fontsize=14)
plt.ylabel('Density', fontsize=14)

plt.legend()
# plt.savefig('figures/HousingData_label_distribution.png')
plt.show()



from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def data_processing():

    # file_path2_1 = '../../data/interim/3/HousingData_normalised.csv'
    # file_path2_2 = '../../data/interim/3/HousingData_standardised.csv'
    # df_normalized = pd.read_csv(file_path2_1)
    # df_standardized = pd.read_csv(file_path2_2)

    file_path2 = '../../data/interim/3/HousingData.csv'
    df = pd.read_csv(file_path2)
    df_features = df.drop('MEDV', axis=1)

    # 1. Handle missing data (if any)
    # Check for missing values
    print(df_features.isnull().sum())

    # Check if there are any missing values in the dataset
    missing_values = df_features.isnull().sum().sum()

    # If there are missing values, apply imputation
    if missing_values > 0:
        print(f"Missing values detected: {missing_values}. Imputing missing values using the mean.")
        # Create the imputer with the 'mean' strategy
        imputer = SimpleImputer(strategy='mean')
        # Apply imputation and return the imputed DataFrame
        df_imputed = pd.DataFrame(imputer.fit_transform(df_features), columns=df_features.columns)
        
    else:
        print("No missing values detected.")
        # No imputation needed, so use the original features DataFrame
        df_imputed = df_features



    # 2. Normalize the feature data (scale between 0 and 1)
    min_max_scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(min_max_scaler.fit_transform(df_imputed), columns=df_imputed.columns)

    # 3. Standardize the feature data (mean = 0, std = 1)
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

    # Print the first few rows of normalized and standardized data
    print("Normalized Data (first 5 rows):\n", df_normalized.head())
    print("Standardized Data (first 5 rows):\n", df_standardized.head())


    # # 4. Save both normalized and standardized datasets as CSV files
    # df_normalized.to_csv('../../data/interim/3/HousingData_normalised.csv', index=False)
    # df_standardized.to_csv('../../data/interim/3/HousingData_standardised.csv', index=False)


# data_processing()





file_path2_1 = '../../data/interim/3/HousingData_normalised.csv'
file_path2_2 = '../../data/interim/3/HousingData_standardised.csv'
df_normalized = pd.read_csv(file_path2_1)
df_standardized = pd.read_csv(file_path2_2)
# print("Normalized Data (first 5 rows):\n", df_normalized.head())
# print("Standardized Data (first 5 rows):\n", df_standardized.head())
# df_normalized.info()
# df_standardized.info()












# def accuracy_score(y_true, y_pred):
#     correct_predictions = np.sum(y_true == y_pred)  # Count correct predictions
#     total_predictions = len(y_true)  # Total number of predictions
#     accuracy = correct_predictions / total_predictions  # Calculate accuracy
#     return accuracy

def accuracy_score(y_true, y_pred):
    y_true_labels = np.argmax(y_true, axis=1)  # True class labels
    y_pred_labels = np.argmax(y_pred, axis=1)  # Predicted class labels
    
    # y_true_labels = y_true.flatten() # True class labels
    # y_pred_labels = y_pred.flatten() # Predicted class labels
    
    correct_predictions = np.sum(y_true_labels == y_pred_labels)

    # print("y_true: ", y_true[:10])
    # print("y_pred: ", y_pred[:10])
    # print("y_true: ", y_true_labels[:10])
    # print("y_pred: ", y_pred_labels[:10])
    # correct_predictions = np.sum(y_true == y_pred)  # Count correct predictions
    
    # print("correct_predictions: ", correct_predictions)
    total_predictions = len(y_true)  # Total number of predictions
    # print("total_predictions: ", total_predictions)
    accuracy = correct_predictions / total_predictions  # Calculate accuracy
    return accuracy


X = df_standardized.values  # Get the standardized feature values as NumPy array
# X = df_normalized.values  
y = df['MEDV'].values    # Extract 'MEDV' as target values
y = y.astype(float).reshape(-1, 1)
# y = pd.get_dummies(df['quality']).values
# y = y.astype(float)
# print(y[:10])
# Define the split sizes
train_size = 0.7
val_size = 0.2
test_size = 0.1
# 2, 3, 7, 11, 
# Shuffle the data
# np.random.seed(41)
np.random.seed(16)
indices = np.random.permutation(len(X))

# Compute the split indices
train_end = int(train_size * len(X))
val_end = train_end + int(val_size * len(X))

# Split the data
X_train, X_val, X_test = X[indices[:train_end]], X[indices[train_end:val_end]], X[indices[val_end:]]
y_train, y_val, y_test = y[indices[:train_end]], y[indices[train_end:val_end]], y[indices[val_end:]]

# Step 4: Initialize the MLP Classifier
input_size = X_train.shape[1]
# print(input_size)  # Number of features (columns in X)
# hidden_layers = [30, 30, 1] 
# hidden_layers = [20, 19] 
# hidden_layers = [2,2] 
# hidden_layers = [20, 16] 
# hidden_layers = [12, 12, 5, 5]
# hidden_layers = [120, 120, 120, 120] 
hidden_layers = [60, 60, 50, 50, 40, 40, 30, 30]
# hidden_layers = [20, 20, 20, 20] 
# output_size = len(np.unique(y))  # Number of unique classes in the target (e.g., if classification)
output_size = y.shape[1]
# print(output_size)
# output_size = 1


# learning_rate=0.001, 0.0008
mlp = MLP_Regression(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size, learning_rate=0.001, activation='relu', optimizer='sgd', epochs=100, task = 'regression',  early_stopping = True)

mlp.fit(X_train, y_train, X_val, y_val)
print(y_train[:10])
y_pred_train = np.round(mlp.predict(X_train) ) # Predictions on the training set
print(y_pred_train[:10])
# print("all same?: ", np.all(y_pred_train == y_pred_train.flat[0]))
y_pred_val = np.round(mlp.predict(X_val) )  
y_pred_test = np.round(mlp.predict(X_test) )  



train_mse = metrics1.calculate_mse(y_train, y_pred_train)  # Round predicted values if necessary
val_mse = metrics1.calculate_mse(y_val, y_pred_val)
test_mse = metrics1.calculate_mse(y_test, y_pred_test)

print(f"Training rmse: {np.sqrt(train_mse)} ")
print(f"Val rmse: {np.sqrt(val_mse)}")
print(f"Test MSE: {test_mse}")
print(f"Test RMSE: {np.sqrt(test_mse)}")
print(f"Test MAE: {metrics1.mean_absolute_error(y_test, y_pred_test)}")

print(f"Training r2: {metrics1.r_squared(y_train, y_pred_train)} ")
print(f"Val r2: {metrics1.r_squared(y_val, y_pred_val)}")
print(f"Test R-suared: {metrics1.r_squared(y_test, y_pred_test)}")




# wandb.login(key='c354ec9456e466caac732ae8538316fe40e79ec3')

sweep_config = {
    'method': 'grid', 
    'name': 'MLP regression hyperparameter tuning',
    'metric': {
        'name': 'val_mse',
        'goal': 'minimize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.0008]
        },
        'epochs': {
            'values': [50, 100, 200]
        },
        'activation': {
            'values': ['relu', 'sigmoid', 'tanh']
        },
        'optimizer': {
            'values': ['sgd', 'mini_batch', 'batch']
        },
        'hidden_layers': {
            'values': [[5, 5], [10, 10], [20, 20], [30, 30], [50, 50], [20, 15], [30, 20], [25, 20], [60, 30], [20, 19], [12, 12, 5, 5], [40, 30, 25, 20], [20, 20, 20, 20] , [60, 60, 50, 50], [28, 23, 20, 15], [20, 20, 20, 20, 20, 20, 20, 20], [60, 60, 50, 50, 40, 40, 30, 30], [40, 40, 30, 30, 25, 25, 20, 20], [40, 35, 30, 25, 20, 15, 10, 10] ]
        }
    }
}
# sweep_config = {
#     'method': 'grid', 
#     'name': 'MLP regression hyperparameter tuning 1',
#     'metric': {
#         'name': 'val_mse',
#         'goal': 'minimize'   
#     },
#     'parameters': {
#         'learning_rate': {
#             'values': [0.001]
#         },
#         'epochs': {
#             'values': [100]
#         },
#         'activation': {
#             'values': ['relu']
#         },
#         'optimizer': {
#             'values': ['sgd', 'mini_batch', 'batch']
#         },
#         'hidden_layers': {
#             'values': [[20, 19], [12, 12, 5, 5]]
#         }
#     }
# }


def train_model_sweep():
    wandb.init(project="MLP-Regression-Tuning")
    # Set up your model with hyperparameters from W&B config
    config = wandb.config
    hidden_layers=config["hidden_layers"]
    learning_rate=config["learning_rate"]
    activation=config["activation"]
    optimizer=config["optimizer"]
    epochs=config["epochs"]
     
    run_name = f"Hl_{str(hidden_layers)}_lr_{learning_rate}_act_{activation}_opt_{optimizer}_epochs_{epochs}"
    wandb.run.name = run_name

    mlp = MLP_Regression(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size, learning_rate=learning_rate, activation=activation, optimizer=optimizer, epochs=epochs, task = 'regression', early_stopping = True)


    mlp.fit(X_train, y_train, X_val, y_val, wb = True)

# sweep_id = wandb.sweep(sweep_config, project="MLP-Regression-Tuning")
# wandb.agent(sweep_id, train_model_sweep)





# # Perform gradient checking
# is_correct = mlp.gradient_check(X_train, y_train)

# if is_correct:
#     print("Gradients are correct. Proceeding with training.")
#     # mlp.fit(X_train, y_train)
# else:
#     print("Gradient check failed. Please review the implementation.")
































from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

file_path3 = '../../data/interim/3/diabetes.csv'
df = pd.read_csv(file_path3)

def data_processing3():

    # file_path2_1 = '../../data/interim/3/HousingData_normalised.csv'
    # file_path2_2 = '../../data/interim/3/HousingData_standardised.csv'
    # df_normalized = pd.read_csv(file_path2_1)
    # df_standardized = pd.read_csv(file_path2_2)

    
    file_path3 = '../../data/interim/3/diabetes.csv'
    df = pd.read_csv(file_path3)
    df_features = df.drop('Outcome', axis=1)

    # 1. Handle missing data (if any)
    # Check for missing values
    print(df_features.isnull().sum())

    # Check if there are any missing values in the dataset
    missing_values = df_features.isnull().sum().sum()

    # If there are missing values, apply imputation
    if missing_values > 0:
        print(f"Missing values detected: {missing_values}. Imputing missing values using the mean.")
        # Create the imputer with the 'mean' strategy
        imputer = SimpleImputer(strategy='mean')
        # Apply imputation and return the imputed DataFrame
        df_imputed = pd.DataFrame(imputer.fit_transform(df_features), columns=df_features.columns)
        
    else:
        print("No missing values detected.")
        # No imputation needed, so use the original features DataFrame
        df_imputed = df_features



    # 2. Normalize the feature data (scale between 0 and 1)
    min_max_scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(min_max_scaler.fit_transform(df_imputed), columns=df_imputed.columns)

    # 3. Standardize the feature data (mean = 0, std = 1)
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

    # Print the first few rows of normalized and standardized data
    print("Normalized Data (first 5 rows):\n", df_normalized.head())
    print("Standardized Data (first 5 rows):\n", df_standardized.head())


    # # 4. Save both normalized and standardized datasets as CSV files
    # df_normalized.to_csv('../../data/interim/3/diabetes_normalized.csv', index=False)
    # df_standardized.to_csv('../../data/interim/3/diabetes_standardized.csv', index=False)


# data_processing3()





file_path2_1 = '../../data/interim/3/diabetes_normalized.csv'
file_path2_2 = '../../data/interim/3/diabetes_standardized.csv'
df_normalized = pd.read_csv(file_path2_1)
df_standardized = pd.read_csv(file_path2_2)

X = df_standardized.values  # Get the standardized feature values as NumPy array
# X = df_normalized.values  
y = df['Outcome'].values    
y = y.astype(float).reshape(-1, 1)
print(y[:10])
# Define the split sizes
train_size = 0.7
val_size = 0.2
test_size = 0.1
# 2, 3, 7, 11, 
# Shuffle the data
# np.random.seed(41)
np.random.seed(16)
indices = np.random.permutation(len(X))

# Compute the split indices
train_end = int(train_size * len(X))
val_end = train_end + int(val_size * len(X))

# Split the data
X_train, X_val, X_test = X[indices[:train_end]], X[indices[train_end:val_end]], X[indices[val_end:]]
y_train, y_val, y_test = y[indices[:train_end]], y[indices[train_end:val_end]], y[indices[val_end:]]

input_size = X_train.shape[1]
hidden_layers = [1]
output_size = y.shape[1]



# learning_rate=0.001, 0.0008
mlp_bce = MLP_Regression(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size, learning_rate=0.001, activation='sigmoid', optimizer='sgd', epochs=100, task = 'regression',  early_stopping = False)

mlp_bce.fit(X_train, y_train, X_val, y_val)

# Fit the model and get the loss values
losses_bce = mlp_bce.fit(X_train, y_train, X_val, y_val)
# losses_mse = mlp_mse.fit(X_train, y_train)

# # Plotting the loss vs epochs
# plt.figure(figsize=(10, 6))
# plt.plot(losses_bce, label='BCE Loss', color='blue')
# # plt.plot(losses_mse, label='MSE Loss', color='orange')
# plt.title('Loss vs Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()
# # plt.savefig('figures/bce_plot.png')
# plt.show()




































# 4 Autoencoder


# Load the CSV files into separate DataFrames
train_df = pd.read_csv('../../data/interim/1/spotify_1/train.csv')
test_df = pd.read_csv('../../data/interim/1/spotify_1/test.csv')
validation_df = pd.read_csv('../../data/interim/1/spotify_1/val.csv')

# Combine all three DataFrames into one
combined_df = pd.concat([train_df, test_df, validation_df], ignore_index=True)
combined_df_1 = combined_df.drop(columns=['track_genre'])

combined_vector = np.array(combined_df_1.apply(lambda row: row.values.tolist(), axis=1).tolist())
print(combined_vector.shape)

X = combined_vector.copy()


# 4.2

def train_autoencoder():
    X_train = X.copy()

    input_size = X_train.shape[1]
    # hidden_layers = [30, 20, 15] 
    hidden_layers = [60,60,50,50,40,40,30,30]
    latent_size = 11 # from assignment 2
    autoencoder = AutoEncoder(input_size=input_size, latent_size=latent_size, hidden_layers=hidden_layers, learning_rate=0.001, activation='relu', optimizer='mini_batch', epochs=3,  early_stopping = False)

    autoencoder.fit(X_train)

# 4.3

    X_reduced = autoencoder.get_latent(X)
    # X_val_reduced = autoencoder.get_latent(X_val)

    print("X_reduced shape: ",X_reduced.shape)
    # print(type(X_reduced))
    np.save('../../data/interim/3/Spotify_reduced.npy', X_reduced) 

# train_autoencoder()

X = np.load('../../data/interim/3/Spotify_reduced.npy')
# X = X_reduced.copy()
print("X extracted shape: ",X.shape)
y = np.array(combined_df['track_genre'])

# Define the split sizes
train_size = 0.8
val_size = 0.1
test_size = 0.1

# Shuffle the data
np.random.seed(42)
indices = np.random.permutation(len(X))

# Compute the split indices
train_end = int(train_size * len(X))
val_end = train_end + int(val_size * len(X))

# Split the data
X_train, X_val, X_test = X[indices[:train_end]], X[indices[train_end:val_end]], X[indices[val_end:]]
y_train, y_val, y_test = y[indices[:train_end]], y[indices[train_end:val_end]], y[indices[val_end:]]






k = 1
distance_metric = 'manhattan'

print(f"k = {k} and distance metric = {distance_metric} ")

# Initialize and train KNN
knn_model = KNN(k=k, distance_metric=distance_metric)
y_val_pred = knn_model.predict(X_train, y_train, X_val)

# Evaluate using the custom metrics
metrics = Metrics(y_val, y_val_pred)
accuracy = metrics.accuracy()

print(f"Validation Accuracy for k={k}, distance_metric={distance_metric}: {accuracy:.4f}")

# Calculate accuracy, precision, recall, and F1 scoresmacro_precision = metrics.macro_precision()
macro_precision = metrics.macro_precision()
macro_recall = metrics.macro_recall()
macro_f1 = metrics.macro_f1()

micro_precision = metrics.micro_precision()
micro_recall = metrics.micro_recall()
micro_f1 = metrics.micro_f1()

# Print results
print(f"Validation Macro Precision: {macro_precision:.4f}")
print(f"Validation Macro Recall: {macro_recall:.4f}")
print(f"Validation Macro F1 Score: {macro_f1:.4f}")
print(f"Validation Micro Precision: {micro_precision:.4f}")
print(f"Validation Micro Recall: {micro_recall:.4f}")
print(f"Validation Micro F1 Score: {micro_f1:.4f}")









# 4.4

combined_vector = np.array(combined_df_1.apply(lambda row: row.values.tolist(), axis=1).tolist())
print(combined_vector.shape)
X = combined_vector.copy()

# Encode the 'track_genre' column using pd.Categorical
combined_df['track_genre_encoded'] = pd.Categorical(combined_df['track_genre']).codes

y = combined_df['track_genre_encoded'].values  # Encoded target variable
y = y.astype(float).reshape(-1, 1)

# Define the split sizes
train_size = 0.8
val_size = 0.1
test_size = 0.1

# Shuffle the data
np.random.seed(42)
indices = np.random.permutation(len(X))

# Compute the split indices
train_end = int(train_size * len(X))
val_end = train_end + int(val_size * len(X))

# Split the data
X_train, X_val, X_test = X[indices[:train_end]], X[indices[train_end:val_end]], X[indices[val_end:]]
y_train, y_val, y_test = y[indices[:train_end]], y[indices[train_end:val_end]], y[indices[val_end:]]


input_size = X_train.shape[1]
hidden_layers = [80, 80, 60, 40] 
output_size = y.shape[1]



# mlp = MLP_Classifier(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size, learning_rate=0.001, activation='relu', optimizer='mini_batch', epochs=200, task = 'classification',  early_stopping = True)

# mlp.fit(X_train, y_train, X_val, y_val)
# y_pred_train = np.round(mlp.predict(X_train) ) # Predictions on the training set
# y_pred_val = np.round(mlp.predict(X_val) )   # Predictions on the test set

# # Evaluate using the custom metrics
# metrics = Metrics(y_val, y_pred_val)
# accuracy = metrics.accuracy()

# print(f"Validation Accuracy: {accuracy:.4f}")

# # Calculate accuracy, precision, recall, and F1 scores
# macro_precision = metrics.macro_precision()
# macro_recall = metrics.macro_recall()
# macro_f1 = metrics.macro_f1()

# micro_precision = metrics.micro_precision()
# micro_recall = metrics.micro_recall()
# micro_f1 = metrics.micro_f1()

# # Print results
# print(f"Validation Macro Precision: {macro_precision:.4f}")
# print(f"Validation Macro Recall: {macro_recall:.4f}")
# print(f"Validation Macro F1 Score: {macro_f1:.4f}")
# print(f"Validation Micro Precision: {micro_precision:.4f}")
# print(f"Validation Micro Recall: {micro_recall:.4f}")
# print(f"Validation Micro F1 Score: {micro_f1:.4f}")