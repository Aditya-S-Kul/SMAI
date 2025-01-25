import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-6, reg_covar=1e-6):
        """
        Initialize the GMM model.
        
        :param n_components: Number of Gaussian components (clusters)
        :param max_iter: Maximum number of iterations for the EM algorithm
        :param tol: Tolerance for convergence
        :param reg_covar: Regularization term to add to covariance matrix
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.resp_ = None

    def fit(self, X):
        """
        Fit the GMM to the data using the EM algorithm.
        
        :param X: Input data, shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        # Randomly initialize the means, covariances, and weights
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.cov(X.T) + self.reg_covar * np.eye(n_features) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components

        log_likelihoods = []
        for i in range(self.max_iter):
            # E-step: Calculate responsibilities
            self.resp_ = self._e_step(X)
            # M-step: Update parameters
            self._m_step(X)
            # Compute log likelihood and check for convergence
            log_likelihood = self.getLikelihood(X)
            log_likelihoods.append(log_likelihood)
            if i > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                break

    def _e_step(self, X):
        """
        E-step: Compute the membership probabilities (responsibilities).
        
        :param X: Input data, shape (n_samples, n_features)
        :return: Responsibilities matrix, shape (n_samples, n_components)
        """
        n_samples = X.shape[0]
        log_resp = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            cov_reg = self.covariances_[k] + self.reg_covar * np.eye(X.shape[1])
            try:
                # Compute the log of the probabilities for numerical stability
                log_resp[:, k] = np.log(self.weights_[k]) + multivariate_normal.logpdf(X, mean=self.means_[k], cov=cov_reg)
            except np.linalg.LinAlgError:
                log_resp[:, k] = -np.inf

        # Stabilize computation by subtracting the max log response
        log_resp -= np.max(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp)
        total_resp = resp.sum(axis=1, keepdims=True)
        total_resp[total_resp == 0] = np.finfo(float).eps  # Avoid division by zero
        resp /= total_resp

        return resp

    def _m_step(self, X):
        """
        M-step: Update the model parameters (weights, means, covariances).
        
        :param X: Input data, shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        resp_sum = self.resp_.sum(axis=0)

        # Update the weights
        self.weights_ = resp_sum / n_samples
        # Update the means
        self.means_ = np.dot(self.resp_.T, X) / resp_sum[:, np.newaxis]

        # Update the covariances
        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = diff.T @ (self.resp_[:, k][:, np.newaxis] * diff)
            self.covariances_[k] = weighted_diff / resp_sum[k]
            # Ensure covariance matrix is positive definite
            self.covariances_[k] += self.reg_covar * np.eye(n_features)

    def getParams(self):
        """
        Get the parameters of the GMM.
        
        :return: Tuple of (weights, means, covariances)
        """
        return self.weights_, self.means_, self.covariances_

    def getMembership(self):
        """
        Get the membership values (responsibilities) for each sample.
        
        :return: Responsibilities matrix, shape (n_samples, n_components)
        """
        return self.resp_

    def getLikelihood(self, X):
        """
        Calculate the log likelihood of the dataset under the current model.
        
        :param X: Input data, shape (n_samples, n_features)
        :return: Log likelihood value
        """
        n_samples = X.shape[0]
        log_likelihood = np.zeros(n_samples)

        for k in range(self.n_components):
            cov_reg = self.covariances_[k] + self.reg_covar * np.eye(X.shape[1])
            try:
                # Compute log likelihood for each sample
                log_likelihood += self.weights_[k] * np.exp(multivariate_normal.logpdf(X, mean=self.means_[k], cov=cov_reg))
            except np.linalg.LinAlgError:
                log_likelihood += 0

        # Avoid log of zero by setting small epsilon values
        log_likelihood[log_likelihood == 0] = np.finfo(float).eps
        return np.sum(np.log(log_likelihood))
