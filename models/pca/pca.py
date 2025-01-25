import numpy as np

class PCA:
    def __init__(self, n_components):
        """
        Initialize the PCA class with the desired number of components.
        :param n_components: The number of principal components to reduce to.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.singular_values = None
        self.eigenvalues_ = None

    def fit(self, X):
        """
        Fit the PCA model to the data X by computing the principal components.
        :param X: Data to perform PCA on. Shape: (n_samples, n_features)
        """
        # Step 1: Center the data (subtract the mean)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2: Compute SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Step 3: Select the top 'n_components' from Vt (right singular vectors)
        self.components = Vt[:self.n_components]
        self.singular_values = S[:self.n_components]
        
        # Compute eigenvalues
        n_samples = X.shape[0]
        self.eigenvalues_ = (self.singular_values ** 2) / (n_samples - 1)

    def transform(self, X):
        """
        Transform the data X into the principal component space.
        :param X: Data to transform. Shape: (n_samples, n_features)
        :return: Transformed data of shape (n_samples, n_components)
        """
        # Step 1: Center the data
        X_centered = X - self.mean
        
        # Step 2: Project the data onto the principal components
        return np.dot(X_centered, self.components.T)

    # def checkPCA(self, X):
    #     """
    #     Check if PCA reduces the data to the desired number of dimensions.
    #     :param X: Original data of shape (n_samples, n_features)
    #     :return: True if PCA reduces the dimensions correctly, False otherwise.
    #     """
    #     # Step 1: Transform the data
    #     transformed_X = self.transform(X)
        
    #     # Step 2: Check the shape of the transformed data
    #     return transformed_X.shape[1] == self.n_components

    def checkPCA(self, X, threshold=0.05):
        """
        Verify PCA by reconstructing the original data from the reduced data and calculating reconstruction error.
        :param X: Original data (n_samples, n_features)
        :param threshold: Maximum acceptable reconstruction error
        :return: True if reconstruction error is below the threshold, False otherwise.
        """
        # Step 1: Transform the data to the reduced space
        X_reduced = self.transform(X)
        
        # Step 2: Reconstruct the original data from the reduced space
        X_reconstructed = self.inverse_transform(X_reduced)
        
        # Step 3: Calculate the reconstruction error (mean squared error)
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        # Step 4: Check if the reconstruction error is below the threshold
        return reconstruction_error < threshold

    
    def inverse_transform(self, X_reduced):
        """
        Reconstruct the original data from the reduced principal component space.
        :param X_reduced: Reduced data in the principal component space (n_samples, n_components)
        :return: Reconstructed data in the original feature space (n_samples, n_features)
        """
        return np.dot(X_reduced, self.components) + self.mean


    def explained_variance(self):
        """
        Return the eigenvalues (explained variance) for each principal component.
        :return: Eigenvalues of shape (n_components,)
        """
        return self.eigenvalues_
