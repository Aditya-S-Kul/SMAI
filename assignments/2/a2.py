import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.k_means.k_means import K_means
from models.gmm.gmm import GMM
from models.pca.pca import PCA
from models.knn.knn import KNN
from performance_measures.metrics import Metrics
from sklearn.mixture import GaussianMixture

df = pd.read_feather('../../data/interim/2/word-embeddings.feather')
X = np.stack(df['vit'].to_numpy())
y = df['words'].to_numpy()

# K-means 

# # Initially tested k-means on given small 2D dataset().

# df = pd.read_csv('../../data/interim/2/data.csv')

# print(df.head())

# # Extract the 'x' and 'y' columns and convert them to numpy arrays
# x_values = df['x'].to_numpy()  # Vector for column 'x'
# y_values = df['y'].to_numpy()  # Vector for column 'y'
# color = df['color'].to_numpy() 
# # Combine 'x' and 'y' into an array of n data samples (2D array with shape [n, 2])
# X = np.column_stack((x_values, y_values))


# k_means_model = K_means(k = 3)

# k_means_model.fit(X)
# C = k_means_model.predict(X)
# print(color)
# print(C)
# print(k_means_model.getCost(X))

# count = 0
# for i in range(len(X)):
#     if str(C[i]) == str(color[i]):
#         count += 1
# print("accuracy = "+str(count/len(X)*100)+" %")
# print("count = "+str(count))



# Main code for k-means starts here

k_values = range(1,21)
costs = []
for i  in k_values:
    
    k_means_model = K_means(k = i)
    k_means_model.fit(X)
    cost = k_means_model.getCost(X)
    costs.append(cost)
    # print("k = "+str(i)+" WCSS = "+str(cost))

plt.figure(figsize=(6, 6))
plt.plot(k_values , costs, marker='o', linestyle='-', color='b')
plt.xlabel('k')
plt.ylabel('WCSS')
plt.title('WCSS vs Number of Clusters (k)')
plt.grid(True)
plt.show()

kkmeans1 = 12

k_means_model = K_means(k = kkmeans1)

k_means_model.fit(X)
clusters = k_means_model.predict()
means = k_means_model.getMeans()
wcss = k_means_model.getCost(X)
print("\nUsing elbow method I got k = "+str(kkmeans1))
print("\nClusters assigned are: \n" ,clusters)
print("\nk means after convergence are: \n" , means)
print("\nWithin-Cluster Sum of Squares (WCSS) = ", wcss)













# GMM


# X = X[:, :190]

# # Generate data from 3 different Gaussian clusters
# n_samples = 500
# mean1 = [2, 5]
# cov1 = [[0.3, 0.4], [0.4, 0.7]]

# mean2 = [3, 5]
# cov2 = [[0.3, -0.4], [-0.4, 0.7]]

# mean3 = [4, 5]
# cov3 = [[0.3, 0.4], [0.4, 0.7]]

# X1 = np.random.multivariate_normal(mean1, cov1, n_samples // 3)
# X2 = np.random.multivariate_normal(mean2, cov2, n_samples // 3)
# X3 = np.random.multivariate_normal(mean3, cov3, n_samples // 3)

# X = np.vstack([X1, X2, X3])  # Combine the clusters into a single dataset
# # print(X)
# print(X)
# # Plot the original data
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], s=10, color='blue')
# plt.title("Original Data from 3 Gaussian Clusters")
# plt.show()



# # Apply GMM clustering

# gmm = GMM(n_components=3)
# gmm.fit(X)

# # Get parameters and membership values
# weights, means, covariances = gmm.getParams()
# membership = gmm.getMembership()

# # Assign each point to the cluster with the highest membership value
# labels = np.argmax(membership, axis=1)

# # # Plot the data after clustering with different colors for each cluster
# # plt.figure(figsize=(8, 6))
# # plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
# # plt.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100, label='Cluster Centers')
# # plt.title("Data After GMM Clustering")
# # plt.legend()
# # plt.show()

# # Print log-likelihood
# likelihood = gmm.getLikelihood(X)
# print("Log-likelihood of the dataset:", likelihood)

# # Print GMM parameters
# print("GMM Weights:", weights)
# print("GMM Means:\n", means)
# print("GMM Covariances:\n", covariances)




# Performing GMM clustering using inbuilt library class

from sklearn.mixture import GaussianMixture

# Define the range for the number of components
n_components_range = range(1, 21)  
bic_scores = []
aic_scores = []

# Compute BIC, and AIC for each number of components
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state = 2)
    gmm.covariance_type = "spherical"
    gmm.fit(X)

    n_features = X.shape[1]
    cov_params = n_components
    mean_params = n_features * n_components
    n_params = int(cov_params + mean_params + n_components - 1)

    bic = -2 * gmm.score(X) * X.shape[0] + n_params * np.log(X.shape[0])
    aic = -2 * gmm.score(X) * X.shape[0] + 2 * n_params

    aic_scores.append(aic)
    bic_scores.append(bic)
    # bic_scores.append(gmm.bic(X))
    # aic_scores.append(gmm.aic(X))

# Determine optimal number of clusters
optimal_n_components_bic = n_components_range[np.argmin(bic_scores)]
optimal_n_components_aic = n_components_range[np.argmin(aic_scores)]

print(f'Optimal number of clusters based on BIC: {optimal_n_components_bic}')
print(f'Optimal number of clusters based on AIC: {optimal_n_components_aic}')

# for i in range(len(bic_scores)):
#     print("\nBIC score for no. of clusters = "+str(i+1)+" is "+ str(bic_scores[i]))
#     print("AIC score for no. of clusters = "+str(i+1)+" is "+ str(aic_scores[i]))

# Plot BIC and AIC scores
plt.figure(figsize=(8, 6))

plt.plot(n_components_range, bic_scores, marker='o', label='BIC')
plt.plot(n_components_range, aic_scores, marker='o', label='AIC')

plt.xlabel('Number of clusters')
plt.ylabel('Scores')
plt.title('BIC and AIC vs Number of Clusters')
plt.legend()

plt.tight_layout()
# plt.savefig('figures/aic_bic_plot.png') 
plt.show()

kgmm1 = 6

# Fit the Gaussian Mixture Model using the in-built sklearn class

gmm = GaussianMixture(n_components=kgmm1)
gmm.fit(X)

# Get parameters
weights = gmm.weights_
means = gmm.means_
covariances = gmm.covariances_

# Get membership (responsibilities)
membership = gmm.predict_proba(X)

# Assign each point to the cluster with the highest membership value
labels = np.argmax(membership, axis=1)
print(labels)


# Print log-likelihood
likelihood = gmm.score(X) * X.shape[0]  # score() returns per-sample log likelihood
print("Log-likelihood of the dataset:", likelihood)

# Print GMM parameters
print("GMM Weights:", weights)
print("GMM Means:\n", means)
print("GMM Covariances:\n", covariances)




# # Checking same generated gaussian dataset on k-means clustering

# k = 3
# k_means_model = K_means(k)
# k_means_model.fit(X)
# labels = k_means_model.predict()
# means = k_means_model.getMeans()
# wcss = k_means_model.getCost(X)
# print("\nUsing elbow method I got k = "+str(k))
# print("\nClusters assigned are: \n" ,labels)
# print("\nk means after convergence are: \n" , means)
# print("\nWithin-Cluster Sum of Squares (WCSS) = ", wcss)

# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
# plt.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100, label='Cluster Centers')
# plt.title("Data After K-means Clustering")
# plt.legend()
# plt.show()
















# PCA

# from mpl_toolkits.mplot3d import Axes3D


# Perform PCA for 2 dimensions
pca_2d = PCA(n_components=2)
pca_2d.fit(X)
X_2d = pca_2d.transform(X)

print("Principal Components (2D):")
print(pca_2d.components)

# Verify functionality
print("PCA 2D check:", pca_2d.checkPCA(X))

# Perform PCA for 3 dimensions
pca_3d = PCA(n_components=3)
pca_3d.fit(X)
X_3d = pca_3d.transform(X)
X_3d = 3 * X_3d


print("Principal Components (3D):")
print(pca_3d.components)


# Verify functionality
print("PCA 3D check:", pca_3d.checkPCA(X))

# Visualize 2D projection
plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.5)
plt.title("2D PCA Projection")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()

# Visualize 3D projection
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], alpha=0.5)
ax.set_title("3D PCA Projection")
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")
ax.set_zlabel("Third Principal Component")
plt.show()





# 6.1
k2 = 4

k_means_model = K_means(k = k2)

k_means_model.fit(X)
clusters = k_means_model.predict()
means = k_means_model.getMeans()
wcss = k_means_model.getCost(X)
print("\nUsing k2 = "+str(k2))
print("\nClusters assigned are: \n" ,clusters)
print("\nk means after convergence are: \n" , means)
print("\nWithin-Cluster Sum of Squares (WCSS) = ", wcss)







# 6.2

# Initialize PCA to fit all dimensions and compute eigenvalues
pca_full = PCA(n_components=X.shape[1])
pca_full.fit(X)

# Get eigenvalues
eigenvalues = pca_full.explained_variance()

# Generate the Eigenvalue Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--')
plt.title('Eigenvalue Scree Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Eigenvalue')
plt.grid(True)
# plt.savefig('figures/Eigen_Scree_plot.png')
plt.show()

# Compute explained variance ratio
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance

# Compute cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Identify number of components capturing >90% of the variance
threshold = 0.90
num_components = np.argmax(cumulative_explained_variance >= threshold) + 1

print(f"Number of principal components needed to capture at least {threshold * 100}% of the variance: {num_components}")

# Plot the cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.axhline(y=threshold, color='r', linestyle='--', label='90% variance threshold')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.legend()
plt.grid(True)
# plt.savefig('figures/Cum_Scree_plot.png')
plt.show()



reduced_data_comps = num_components

pca_reduced = PCA(n_components=reduced_data_comps)
pca_reduced.fit(X)
X_reduced_data = pca_reduced.transform(X)


k_values = range(1,20)
costs = []
for i  in k_values:
    
    k_means_model = K_means(k = i)
    k_means_model.fit(X_reduced_data)
    cost = k_means_model.getCost(X_reduced_data)
    costs.append(cost)
    # print("k = "+str(i)+" WCSS = "+str(cost))

plt.figure(figsize=(6, 6))
plt.plot(k_values , costs, marker='o', linestyle='-', color='b')
plt.xlabel('k')
plt.ylabel('WCSS')
plt.title('WCSS vs Number of Clusters (k)')
plt.grid(True)
plt.show()

kkmeans3 = 10

k_means_model = K_means(k = kkmeans3)

k_means_model.fit(X_reduced_data)
clusters = k_means_model.predict()
means = k_means_model.getMeans()
wcss = k_means_model.getCost(X_reduced_data)
print("\nUsing elbow method I got k = "+str(kkmeans3))
print("\nClusters assigned are: \n" ,clusters)
print("\nk means after convergence are: \n" , means)
print("\nWithin-Cluster Sum of Squares (WCSS) = ", wcss)



# 6.3


gmm = GMM(n_components=k2)
gmm.fit(X)

# Get parameters and membership values
weights, means, covariances = gmm.getParams()
membership = gmm.getMembership()

# Assign each point to the cluster with the highest membership value
labels = np.argmax(membership, axis=1)
print(labels)

# Print log-likelihood
likelihood = gmm.getLikelihood(X)

# Print GMM parameters
print("GMM Weights:", weights)
print("GMM Means:\n", means)
print("GMM Covariances:\n", covariances)



# 6.4


from sklearn.mixture import GaussianMixture

# Define the range for the number of components
n_components_range = range(1, 21)  
bic_scores = []
aic_scores = []

# Compute log-likelihood, BIC, and AIC for each number of components
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state = 2)
    gmm.covariance_type = "spherical"
    gmm.fit(X_reduced_data)

    n_features = X_reduced_data.shape[1]
    cov_params = n_components
    mean_params = n_features * n_components
    n_params = int(cov_params + mean_params + n_components - 1)

    bic = -2 * gmm.score(X_reduced_data) * X_reduced_data.shape[0] + n_params * np.log(X_reduced_data.shape[0])
    aic = -2 * gmm.score(X_reduced_data) * X_reduced_data.shape[0] + 2 * n_params

    aic_scores.append(aic)
    bic_scores.append(bic)
    # bic_scores.append(gmm.bic(X))
    # aic_scores.append(gmm.aic(X))

# Determine optimal number of clusters
optimal_n_components_bic = n_components_range[np.argmin(bic_scores)]
optimal_n_components_aic = n_components_range[np.argmin(aic_scores)]

print(f'Optimal number of clusters based on BIC: {optimal_n_components_bic}')
print(f'Optimal number of clusters based on AIC: {optimal_n_components_aic}')

# for i in range(len(bic_scores)):
#     print("\nBIC score for no. of clusters = "+str(i+1)+" is "+ str(bic_scores[i]))
#     print("AIC score for no. of clusters = "+str(i+1)+" is "+ str(aic_scores[i]))

# Plot BIC and AIC scores
plt.figure(figsize=(8, 6))

plt.plot(n_components_range, bic_scores, marker='o', label='BIC')
plt.plot(n_components_range, aic_scores, marker='o', label='AIC')

plt.xlabel('Number of clusters')
plt.ylabel('Scores')
plt.title('BIC and AIC vs Number of Clusters')
plt.legend()

plt.tight_layout()
# plt.savefig('figures/aic_bic_reduced_data.png') 
plt.show()



kgmm3 = 5


gmm = GMM(n_components=kgmm3)
gmm.fit(X_reduced_data)

# Get parameters and membership values
weights, means, covariances = gmm.getParams()
membership = gmm.getMembership()

# Assign each point to the cluster with the highest membership value
labels = np.argmax(membership, axis=1)
print(labels)

# Print log-likelihood
likelihood = gmm.getLikelihood(X_reduced_data)
print("Log-likelihood of the dataset:", likelihood)

# Print GMM parameters
print("GMM Weights:", weights)
print("GMM Means:\n", means)
print("GMM Covariances:\n", covariances)







7.1

kkmeans1 = 12
k2 = 4
kkmeans3 = 10
print("k = ", kkmeans3)
k_means_model = K_means(k = kkmeans3)

k_means_model.fit(X)
clusters = k_means_model.predict()
print("\nClusters assigned are: \n" ,clusters)

# Create a dictionary to hold words for each cluster
cluster_dict = {}
for cluster_id in np.unique(clusters):
    cluster_dict[cluster_id] = y[clusters == cluster_id]

# Print words in each cluster
for cluster_id, cluster_words in cluster_dict.items():
    print(f"Cluster {cluster_id}:")
    print(", ".join(cluster_words))
    print()  # Print a blank line for better readability


# 7.2

kgmm1 = 6
k2 = 4
kgmm3 = 5


print("k = ", kgmm3)
# Fit the Gaussian Mixture Model using the in-built sklearn class

gmm = GaussianMixture(n_components=kgmm3)
gmm.fit(X)

# Get membership (responsibilities)
membership = gmm.predict_proba(X)

# Assign each point to the cluster with the highest membership value
clusters = np.argmax(membership, axis=1)

cluster_dict = {}
for cluster_id in np.unique(clusters):
    cluster_dict[cluster_id] = y[clusters == cluster_id]

# Print words in each cluster
for cluster_id, cluster_words in cluster_dict.items():
    print(f"Cluster {cluster_id}:")
    print(", ".join(cluster_words))
    print()  # Print a blank line for better readability
















# 8

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

linkage_methods = ['single', 'complete', 'average', 'ward']
distance_metric = 'euclidean'

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, method in enumerate(linkage_methods):
    Z = hierarchy.linkage(X, method=method, metric=distance_metric)
    hierarchy.dendrogram(Z, ax=axes[i], truncate_mode='lastp', p=15, leaf_rotation=90)
    axes[i].set_title(f'{method.capitalize()} Linkage')
    axes[i].set_xlabel('Sample Index')
    axes[i].set_ylabel('Distance')

plt.tight_layout()
# plt.savefig("figures/dendograms.png")
plt.show()


best_linkage = 'ward'
Z = hierarchy.linkage(X, method=best_linkage, metric=distance_metric)

kbest1, kbest2 = 10, 5

print("\nk = 5\n")

# Hierarchical Clustering
hc_labels_10 = hierarchy.fcluster(Z, t=kbest1, criterion='maxclust')
hc_labels_5 = hierarchy.fcluster(Z, t=kbest2, criterion='maxclust')

cluster_dict = {}
for cluster_id in np.unique(hc_labels_5):
    cluster_dict[cluster_id] = y[hc_labels_5 == cluster_id]

# Print words in each cluster
for cluster_id, cluster_words in cluster_dict.items():
    print(f"Cluster {cluster_id-1}:")
    print(", ".join(cluster_words))
    print()  # Print a blank line for better readability


print("\nk = 10\n")
cluster_dict = {}
for cluster_id in np.unique(hc_labels_10):
    cluster_dict[cluster_id] = y[hc_labels_10 == cluster_id]

# Print words in each cluster
for cluster_id, cluster_words in cluster_dict.items():
    print(f"Cluster {cluster_id-1}:")
    print(", ".join(cluster_words))
    print()  # Print a blank line for better readability













# 9.1

# Load the CSV files into separate DataFrames
train_df = pd.read_csv('../../data/interim/1/spotify_1/train.csv')
test_df = pd.read_csv('../../data/interim/1/spotify_1/test.csv')
validation_df = pd.read_csv('../../data/interim/1/spotify_1/val.csv')

# Combine all three DataFrames into one
combined_df = pd.concat([train_df, test_df, validation_df], ignore_index=True)
combined_df_1 = combined_df.drop(columns=['track_genre'])

# print(combined_df.head())
# all_numeric = combined_df.map(lambda x: isinstance(x, (int, float))).all().all()
# print(all_numeric)

combined_vector = np.array(combined_df_1.apply(lambda row: row.values.tolist(), axis=1).tolist())
print(combined_vector.shape)

X = combined_vector.copy()

# Initialize PCA to fit all dimensions and compute eigenvalues
pca_full = PCA(n_components=X.shape[1])
pca_full.fit(X)

# Get eigenvalues
eigenvalues = pca_full.explained_variance()

# Generate the Eigenvalue Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--')
plt.title('Eigenvalue Scree Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()

# Compute explained variance ratio
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance

# Compute cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Identify number of components capturing >90% of the variance
threshold = 0.90
num_components = np.argmax(cumulative_explained_variance >= threshold) + 1

print(f"Number of principal components needed to capture at least {threshold * 100}% of the variance: {num_components}")

# Plot the cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.axhline(y=threshold, color='r', linestyle='--', label='90% variance threshold')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.legend()
plt.grid(True)
plt.show()



reduced_data_comps = 11

pca_reduced = PCA(n_components=reduced_data_comps)
pca_reduced.fit(X)
X_reduced_data = pca_reduced.transform(X)





X = X_reduced_data.copy()
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

# print("Training set shape:", X_train.shape, y_train.shape)
# print("Validation set shape:", X_val.shape, y_val.shape)
# print("Test set shape:", X_test.shape, y_test.shape)

k = 1
distance_metric = 'manhattan'

print(f"k = {k} and distance metric = {distance_metric} ")

# start_time = time.time()

# Initialize and train KNN
knn_model = KNN(k=k, distance_metric=distance_metric)
y_val_pred = knn_model.predict(X_train, y_train, X_val)

# end_time = time.time()
# inference_time = end_time - start_time
# print("inference_time = ",inference_time)

# Evaluate using the custom metrics
metrics = Metrics(y_val, y_val_pred)
accuracy = metrics.accuracy()

print(f"Validation Accuracy for k={k}, distance_metric={distance_metric}: {accuracy:.4f}")




# 9.2

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








# Testing on assignment 1 dataset
print("\nResults of assignment 1 dataset: \n")

train_path = '../../data/interim/1/spotify_1/train.csv'
val_path = '../../data/interim/1/spotify_1/val.csv'
test_path = '../../data/interim/1/spotify_1/test.csv'

# Load the datasets
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

# Separate the features (X) and target (y)
X_train = train_df.drop(columns=['track_genre']).values
y_train = train_df['track_genre'].values

X_val = val_df.drop(columns=['track_genre']).values
y_val = val_df['track_genre'].values

X_test = test_df.drop(columns=['track_genre']).values
y_test = test_df['track_genre'].values


k = 1
distance_metric = 'manhattan'

print(f"k = {k} and distance metric = {distance_metric} ")

# start_time = time.time()

# Initialize and train KNN
knn_model = KNN(k=k, distance_metric=distance_metric)
y_val_pred = knn_model.predict(X_train, y_train, X_val)

# end_time = time.time()
# inference_time = end_time - start_time
# print("inference_time = ", inference_time)

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



inference_times = [20.696399688720703, 19.81725788116455]


# Labels for the two bars
labels = ['reduced_data', 'full_data']

# Plotting the inference times as a bar chart
plt.figure(figsize=(8, 6))
plt.bar(labels, inference_times, color='blue') 

# Adding labels and title
plt.xlabel('Data Type')
plt.ylabel('Inference Time (seconds)')  
plt.title('Inference Times for Reduced and Full Data')

# Save the figure as a PNG file
# plt.savefig('figures/inference_times_plot.png')  

# Optionally, show the plot
plt.show()