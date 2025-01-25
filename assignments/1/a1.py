import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import json
import ast
from itertools import combinations
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.knn.knn import KNN
from models.linear_regression.linear_regression import LinearRegression
from performance_measures.metrics import Metrics




# # start_time = time.time()

file_path1 = '../../data/interim/1/spotify_unique.csv'
df = pd.read_csv(file_path1)

# Data cleaning and normalization:

def data_cleaning():
    # file_path0 = '../../data/external/spotify.csv'

    # file_path = '../../data/interim/1/spotify_cleaned.csv'
    # file_path = '../../data/interim/1/spotify_final_1.csv'
    # file_path = '../../data/interim/1/spotify_final_2.csv'
    file_path = '../../data/interim/1/spotify_final_3.csv'


    # Data cleaning(removed multiple genres for 1 song and removed missing data rows)

    # df = df.drop_duplicates(subset='track_id', keep='first')
    # df = df.dropna()
    # missing_data = df.isnull().sum()
    # print("Missing data in each column:\n", missing_data)
    # df.to_csv(file_path1, index=False)

    # print(df.info())
    # print(df.describe())

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    print("Numerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)

    def sub_data_cleaning():
        # Print number of unique values for each categorical feature
        for feature in categorical_features:
            unique_values = df[feature].nunique()
            print(f'Feature: {feature}, Number of Unique Values: {unique_values}')


        # Drop specified categorical features
        features_to_drop = ['artists', 'album_name', 'track_name']
        df = df.drop(columns=features_to_drop, errors='ignore')

        # Verify columns after dropping
        print("Columns after dropping specified features:", df.columns.tolist())

        # Update feature lists after dropping columns
        numerical_features = df.select_dtypes(include=['number']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()

        print("Updated Numerical Features:", numerical_features)
        print("Updated Categorical Features:", categorical_features)

        df.to_csv('../../data/interim/1/spotify_final_1.csv', index=False)


        # Convert boolean features to integers (False = 0, True = 1)
        boolean_features = df.select_dtypes(include=['bool']).columns
        print(boolean_features)
        df[boolean_features] = df[boolean_features].astype(int)

        df.to_csv('../../data/interim/1/spotify_final_2.csv', index=False)

        features_to_normalize = ['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

        # Standardization
        def standardize(df, features):
            for feature in features:
                mean = df[feature].mean()
                std_dev = df[feature].std()
                df[feature] = (df[feature] - mean) / std_dev
            return df

        df = standardize(df, features_to_normalize)
        df.to_csv('../../data/interim/1/spotify_final_3.csv', index=False)

    # sub_data_cleaning()

    # Drop the 'track_id' column as it's not useful for KNN
    df = df.drop(columns=['track_id'])
    df.to_csv('../../data/interim/1/spotify_final_4.csv', index=False)

    # Separate the features and target variable
    X = df.drop(columns=['track_genre'])  # Input features
    y = df['track_genre']  # Target feature

    # Convert dataframes to numpy arrays for easy manipulation
    X = X.values
    y = y.values

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

    # save the datasets to CSV
    train_df = pd.DataFrame(X_train, columns=df.columns[:-1])
    train_df['track_genre'] = y_train
    train_df.to_csv('../../data/interim/1/spotify_1/train.csv', index=False)

    val_df = pd.DataFrame(X_val, columns=df.columns[:-1])
    val_df['track_genre'] = y_val
    val_df.to_csv('../../data/interim/1/spotify_1/val.csv', index=False)

    test_df = pd.DataFrame(X_test, columns=df.columns[:-1])
    test_df['track_genre'] = y_test
    test_df.to_csv('../../data/interim/1/spotify_1/test.csv', index=False)

    # Print out the shapes to confirm the splits
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Validation set shape:", X_val.shape, y_val.shape)
    print("Test set shape:", X_test.shape, y_test.shape)


# data_cleaning()




















# KNN Code starts here:

# Exploratory Data Analysis

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
numerical_features = df.select_dtypes(include=['number']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Histograms

df[numerical_features].hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Features')
# plt.savefig('figures/Histograms.png', format='png')
plt.show()

# Boxplots to check for outliers

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_features):
    plt.subplot(len(numerical_features)//3 + 1, 3, i + 1)
    plt.boxplot(df[col], vert=False)
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
# plt.savefig('figures/Boxplots.png', format='png')
plt.show()

# Bar plots for categorical features

# Define the number of top categories to display

categorical_features.remove('track_id')
categorical_features.remove('track_name')

top_n = 10
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_features):
    plt.subplot(len(categorical_features)//3 + 1, 3, i + 1)
    
    # If a feature has too many unique values, display only the top N
    if df[col].nunique() > top_n:
        df[col].value_counts().nlargest(top_n).plot(kind='bar')
        plt.title(f'Bar plot of {col} (Top {top_n})')
    else:
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Bar plot of {col}')
        
    plt.tight_layout()


# plt.savefig('figures/Barplots.png', format='png')

plt.show()


# Calculate the correlation matrix
corr_matrix = df[numerical_features].corr()

# Create a figure and axes with specific size using plt.subplots
fig, ax = plt.subplots(figsize=(16, 12))  # Set the desired size

# Plotting the correlation matrix
cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)

# Add labels
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90)
ax.set_yticklabels(corr_matrix.columns)

# Add the correlation values on the plot
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')

ax.set_title('Correlation Matrix of Numerical Features')
# plt.savefig('figures/Correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()









# Hyper parameter Tuning and rest

# Define the file paths
train_path = '../../data/interim/1/spotify_1/train.csv'
val_path = '../../data/interim/1/spotify_1/val.csv'
test_path = '../../data/interim/1/spotify_1/test.csv'
# # Define the file paths
# train_path = '../../data/external/spotify-2/train.csv'
# val_path = '../../data/external/spotify-2/validate.csv'
# test_path = '../../data/external/spotify-2/test.csv'

# Load the datasets
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

# Reduce data size by sampling
# train_df = train_df.sample(frac=0.2, random_state=42)  
# val_df = val_df.sample(frac=1, random_state=42)  
# test_df = val_df.sample(frac=1, random_state=42)  

# Separate the features (X) and target (y)
X_train = train_df.drop(columns=['track_genre']).values
y_train = train_df['track_genre'].values

X_val = val_df.drop(columns=['track_genre']).values
y_val = val_df['track_genre'].values

X_test = test_df.drop(columns=['track_genre']).values
y_test = test_df['track_genre'].values





# Initial testing

def initial_test():
    # Set k and distance metric
    k = 1
    distance_metric = 'manhattan'

    print(f"k = {k} and distance metric = {distance_metric} Initial model")
    # Initialize and train KNN
    knn_model = KNN(k=k, distance_metric=distance_metric)
    y_val_pred = knn_model.predict(X_train, y_train, X_val)

    # Evaluate using the custom metrics
    metrics = Metrics(y_val, y_val_pred)
    accuracy = metrics.accuracy()

    print(f"Validation Accuracy for k={k}, distance_metric={distance_metric}: {accuracy:.4f}")

# initial_test()


# Testing with sklearn

def sklearn_test():
    k = 2
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')

    # Train the model
    knn.fit(X_train, y_train)

    # Predict on validation data
    y_val_pred = knn.predict(X_val)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_val_pred)

    print(f"Validation Accuracy for k={k}, distance_metric='euclidean': {accuracy:.4f}")


# sklearn_test()





# Hyperparameter tuning

def hyper_parameter_tuning():
    # Define the range of k and distance metrics to try
    k_values = [1,3,5,9,13,17,21] # Example range, you can adjust this
    distance_metrics = ['euclidean', 'manhattan', 'cosine']

    # Dictionary to store validation accuracy for each (k, distance_metric) pair
    validation_results = {}

    for k in k_values:
        for metric in distance_metrics:
            knn_model = KNN(k=k, distance_metric=metric)
            y_val_pred = knn_model.predict(X_train, y_train, X_val)
            metrics = Metrics(y_val, y_val_pred)
            accuracy = metrics.accuracy()
            
            print(f"For k = {k} and distance_metric = {metric}   Accuracy = {accuracy:.4f}")
            validation_results[(k, metric)] = accuracy


    # Convert the dictionary with tuple keys to a dictionary with string keys
    validation_results_str_keys = {str(k): v for k, v in validation_results.items()}

    # Save the dictionary to a JSON file
    with open('validation_results.json', 'w') as file:
        json.dump(validation_results_str_keys, file, indent=4)


# hyper_parameter_tuning()





# Load the JSON file
with open('validation_results.json', 'r') as file:
    loaded_data = json.load(file)

# Convert string keys back to tuples
validation_results = {ast.literal_eval(k): v for k, v in loaded_data.items()}


# Find the best (k, distance_metric) pair

best_k_metric_pair = max(validation_results, key=validation_results.get)
best_accuracy = validation_results[best_k_metric_pair]

print(f"Best (k, distance_metric) pair: {best_k_metric_pair} with validation accuracy: {best_accuracy}\n\n")

# Sort the validation results by accuracy in descending order
sorted_results = sorted(validation_results.items(), key=lambda x: x[1], reverse=True)


# Print the top 10 pairs
print("Top 10 (k, distance_metric) pairs by validation accuracy:\n")
for i, ((k, metric), accuracy) in enumerate(sorted_results[:10]):
    print(f"{i + 1}: k={k}, distance_metric={metric}, accuracy={accuracy:.4f}")
print()
# Choose a distance metric to plot
selected_metric = 'manhattan'

# Extract accuracies for the selected distance metric
k_values = [k for (k, metric) in validation_results.keys() if metric == selected_metric]
accuracies = [accuracy for (k, metric), accuracy in validation_results.items() if metric == selected_metric]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title(f'k vs Accuracy for {selected_metric} Distance Metric')
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()





# Finding best column subset to drop


def find_best_column_subset():
    # Function to compute accuracy using KNN
    def compute_accuracy(X_train, y_train, X_val, y_val, k, metric):
        knn = KNN(k = k, distance_metric = metric)
        y_val_pred = knn.predict(X_train, y_train, X_val)
        metrics = Metrics(y_val, y_val_pred)
        return metrics.accuracy()

    # Define possible feature subsets to try
    # Columns to try dropping
    columns_to_try_dropping = ['popularity', 'duration_ms', 'explicit', 'key', 'mode', 'valence', 'tempo', 'time_signature']

    best_results = {}
    for r in range(1, len(columns_to_try_dropping) + 1):
        for subset in combinations(columns_to_try_dropping, r):
            # Drop the columns in the current subset
            columns_to_drop = list(subset)
            column_indices = [np.where(train_df.columns == col)[0][0] for col in columns_to_drop]
            
            X_train_subset = np.delete(X_train, column_indices, axis=1)
            X_val_subset = np.delete(X_val, column_indices, axis=1)
            
            # Compute accuracy
            accuracy = compute_accuracy(X_train_subset, y_train, X_val_subset, y_val, best_k_metric_pair[0], best_k_metric_pair[1])
            
            # Store the result
            best_results[tuple(columns_to_drop)] = accuracy
            print(f"Dropped columns: {columns_to_drop}, Validation Accuracy: {accuracy:.4f}")

    # Find and print the best feature subset result
    best_subset_columns = max(best_results, key=best_results.get)
    print(f"Best feature subset by dropping {best_subset_columns}, Accuracy: {best_results[best_subset_columns]:.4f}")

    return 

# find_best_column_subset()





# Define the models and their inference times for train size 1
models = ['Initial KNN Model', 'Best KNN Model', 'Optimized KNN Model', 'SKLearn KNN Model']
times = [2141.7567, 2037.0937, 20.2083, 3.7695]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot inference times for each model
plt.bar(models, times, color=['blue', 'green', 'red', 'purple'])

# Adding labels and title
plt.xlabel('Model')
plt.ylabel('Inference Time (s)')
plt.title('Inference Time for Different KNN Models at Train Size 1')
plt.yscale('log')  # Use logarithmic scale for better visualization due to large range
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# plt.savefig('figures/Inf_times.png', format='png')
plt.show()




# Define the train sizes and inference times
train_sizes = [0.2, 0.4, 0.6, 0.8, 1]
times_initial = [251.0111, 698.3058, 1034.7844, 1549.2609, 2141.7567]
times_best = [249.2745, 693.0538, 1009.8664, 1498.9563, 2037.0937]
times_optimized = [3.9232, 8.4679, 12.1807, 15.6675, 20.2083]
times_sklearn = [1.2999, 1.9799, 2.7296, 3.0184, 3.7695]

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)

# Plot for Initial KNN Model
axs[0, 0].plot(train_sizes, times_initial, 'o-', color='blue')
axs[0, 0].set_title('Initial KNN Model')
axs[0, 0].set_ylabel('Inference Time (s)')
axs[0, 0].grid(True)

# Plot for Best KNN Model
axs[0, 1].plot(train_sizes, times_best, 'o-', color='green')
axs[0, 1].set_title('Best KNN Model')
axs[0, 1].grid(True)

# Plot for Optimized KNN Model
axs[1, 0].plot(train_sizes, times_optimized, 'o-', color='red')
axs[1, 0].set_title('Optimized KNN Model')
axs[1, 0].set_xlabel('Train Size')
axs[1, 0].set_ylabel('Inference Time (s)')
axs[1, 0].grid(True)

# Plot for SKLearn KNN Model
axs[1, 1].plot(train_sizes, times_sklearn, 'o-', color='purple')
axs[1, 1].set_title('SKLearn KNN Model')
axs[1, 1].set_xlabel('Train Size')
axs[1, 1].grid(True)

# Adjust layout
plt.tight_layout()
# plt.savefig('figures/Inf_train.png', format='png')
plt.show()




# end_time = time.time()

# execution_time = end_time - start_time
# print(f"Execution time: {execution_time:.4f} seconds")



































# Linear Regression


# Use the class
model = LinearRegression()


# Load the dataset
data = pd.read_csv("../../data/interim/1/linreg.csv")

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Split the data into train, validation, and test sets (80:10:10)
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))

# train_data = data[:train_size]
# val_data = data[train_size:train_size + val_size]
# test_data = data[train_size + val_size:]


# Save the datasets to separate CSV files
# train_data.to_csv('../../data/interim/1/linreg/train.csv', index=False)
# val_data.to_csv('../../data/interim/1/linreg/validation.csv', index=False)
# test_data.to_csv('../../data/interim/1/linreg/test.csv', index=False)

train_data = pd.read_csv("../../data/interim/1/linreg/train.csv")
val_data = pd.read_csv("../../data/interim/1/linreg/validation.csv")
test_data = pd.read_csv("../../data/interim/1/linreg/test.csv")


X_train = train_data['x'].values
y_train = train_data['y'].values

X_val = val_data['x'].values
y_val = val_data['y'].values

X_test = test_data['x'].values
y_test = test_data['y'].values



# Visualize the splits
plt.figure(figsize=(10, 6))
plt.scatter(train_data['x'], train_data['y'], color='blue', label='Train Data', alpha=0.6)
plt.scatter(val_data['x'], val_data['y'], color='green', label='Validation Data', alpha=0.6)
plt.scatter(test_data['x'], test_data['y'], color='red', label='Test Data', alpha=0.6)

plt.title('Train, Validation, and Test Splits')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# plt.savefig('figures/Train_Val_Test_Data.png', format='png')
plt.show()



# Degree 1

# Initialize and train the model
model = LinearRegression(learning_rate=0.01, tolerance=1e-4)
model.fit(X_train, y_train)

# Predictions on train, validation, and test data
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calculate and print metrics for train, validation, and test data
train_mse, train_variance, train_std_dev = model.report_metrics(y_train, y_train_pred)
val_mse, val_variance, val_std_dev = model.report_metrics(y_val, y_val_pred)
test_mse, test_variance, test_std_dev = model.report_metrics(y_test, y_test_pred)

print(f"\nTrain MSE: {train_mse:.4f}, Variance: {train_variance:.4f}, Standard Deviation: {train_std_dev:.4f}")
print(f"Validation MSE: {val_mse:.4f}, Variance: {val_variance:.4f}, Standard Deviation: {val_std_dev:.4f}")
print(f"Test MSE: {test_mse:.4f}, Variance: {test_variance:.4f}, Standard Deviation: {test_std_dev:.4f}\n")

# Plot the fitted line on training data
model.plot_fit(X_train, y_train)




# Degree > 1

best_degree = None
best_mse = float('inf')
best_model = None

for degree in range(1, 16):  # Testing degrees from 1 to 15
    model = LinearRegression(degree=degree, learning_rate=0.01, tolerance=1e-4, regularization=0.0)
    model.fit(X_train, y_train)
    
    # Report metrics on train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mse, train_variance, train_std_dev = model.report_metrics(y_train, y_train_pred)
    test_mse, test_variance, test_std_dev = model.report_metrics(y_test, y_test_pred)
    
    print(f"Degree {degree}:")
    print(f"Train MSE: {train_mse}, Variance: {train_variance}, Std Dev: {train_std_dev}")
    print(f"Test MSE: {test_mse}, Variance: {test_variance}, Std Dev: {test_std_dev}")
    
    if test_mse < best_mse:
        best_mse = test_mse
        best_degree = degree
        best_model = model

# Save the best model
if best_model is not None:
    best_model.save_model('best_polynomial_model.json')
    print(f"\nBest model saved with degree {best_degree}\n")

# Plot the best fit
if best_model is not None:
    best_model.plot_fit(X_train, y_train)



# Animation

# Define the degrees to test
# # degrees = [1,2,3,4,5,6,7,8,9,10]

# for degree in degrees:
#     model = LinearRegression(degree=degree)
#     model.fit(X_train, y_train)
#     model.save_gif(X_train, y_train, f'figures/animation_degree_{degree}.gif')






# Regularization

# Load and prepare data
data = pd.read_csv("../../data/interim/1/regularisation.csv")
data = data.sample(frac=1).reset_index(drop=True)

train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))

# train_data = data[:train_size]
# val_data = data[train_size:train_size + val_size]
# test_data = data[train_size + val_size:]

train_data = pd.read_csv("../../data/interim/1/regularization/train.csv")
val_data = pd.read_csv("../../data/interim/1/regularization/validation.csv")
test_data = pd.read_csv("../../data/interim/1/regularization/test.csv")

X_train = train_data['x'].values
y_train = train_data['y'].values

X_val = val_data['x'].values
y_val = val_data['y'].values

X_test = test_data['x'].values
y_test = test_data['y'].values

# Save the datasets to separate CSV files
# train_data.to_csv('../../data/interim/1/regularization/train.csv', index=False)
# val_data.to_csv('../../data/interim/1/regularization/validation.csv', index=False)
# test_data.to_csv('../../data/interim/1/regularization/test.csv', index=False)

# Visualize the splits
plt.scatter(train_data['x'], train_data['y'], color='blue', label='Train Data', alpha=0.6)

plt.title('Training Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# plt.savefig('figures/Training_Data.png', format='png')
plt.show()

# Test polynomial regression with and without regularization for different degrees
for reg_type in ['l1', 'l2']:
    best_degree = None
    best_mse = float('inf')
    best_model = None

    for degree in range(1, 21):  # Testing degrees from 1 to 20
        model = LinearRegression(degree=degree, learning_rate=0.01, tolerance=1e-4, regularization=0.1, reg_type=reg_type)
        model.fit(X_train, y_train)

        # Report metrics on train and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_mse, train_variance, train_std_dev = model.report_metrics(y_train, y_train_pred)
        test_mse, test_variance, test_std_dev = model.report_metrics(y_test, y_test_pred)
        
        print(f"Regularization: {reg_type}, Degree {degree}:")
        print(f"Train MSE: {train_mse}, Variance: {train_variance}, Std Dev: {train_std_dev}")
        print(f"Test MSE: {test_mse}, Variance: {test_variance}, Std Dev: {test_std_dev}")
        
        if test_mse < best_mse:
            best_mse = test_mse
            best_degree = degree
            best_model = model

    # Save the best model
    if best_model is not None:
        best_model.save_model(f'best_polynomial_model_{reg_type}.json')
        print(f"\nBest model with {reg_type} regularization saved with degree {best_degree}\n")

    # Plot the best fit for each type of regularization
    if best_model is not None:
        best_model.plot_fit(X_train, y_train)
