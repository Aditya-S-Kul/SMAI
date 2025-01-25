import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from performance_measures import metrics

# class KNN:
#     def __init__(self, k=3, distance_metric='euclidean'):
#         self.k = k
#         self.distance_metric = distance_metric
    
#     def set_params(self, k=None, distance_metric=None):
#         if k is not None:
#             self.k = k
#         if distance_metric is not None:
#             self.distance_metric = distance_metric
    
#     def get_params(self):
#         return {'k': self.k, 'distance_metric': self.distance_metric}
    
#     def predict(self, x_train, y_train, x_test):
#         return np.array([self._get_neighbors(x_train, y_train, x) for x in x_test])
    
#     def predict_batch(self, x_train, y_train, x_test):
#         return self.predict(x_train, y_train, x_test)
    
#     # Private methods
#     def _compute_distance(self, x1, x2):
#         if self.distance_metric == 'euclidean':
#             return np.sqrt(np.sum((x1 - x2) ** 2))
#         elif self.distance_metric == 'manhattan':
#             return np.sum(np.abs(x1 - x2))
#         elif self.distance_metric == 'cosine':
#             return 1 - (np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))
#         else:
#             raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
#     def _get_neighbors(self, x_train, y_train, x_test):
#         distances = [self._compute_distance(x_test, x_train[i]) for i in range(len(x_train))]
#         sorted_indices = np.argsort(distances)
#         nearest_indices = sorted_indices[:self.k]
#         nearest_labels = y_train[nearest_indices]
        
#         # Manually compute the most common label
#         unique_labels, counts = np.unique(nearest_labels, return_counts=True)
#         max_count_index = np.argmax(counts)
#         return unique_labels[max_count_index]






class KNN:
    def __init__(self, k=3, distance_metric='euclidean', batch_size=100):
        self.k = k
        self.distance_metric = distance_metric
        self.batch_size = batch_size  # New: Batch size for processing
        
    def set_params(self, k=None, distance_metric=None, batch_size=None):
        if k is not None:
            self.k = k
        if distance_metric is not None:
            self.distance_metric = distance_metric
        if batch_size is not None:
            self.batch_size = batch_size
    
    def get_params(self):
        return {'k': self.k, 'distance_metric': self.distance_metric, 'batch_size': self.batch_size}
    
    def predict(self, x_train, y_train, x_test):
        predictions = []
        for start in range(0, len(x_test), self.batch_size):
            end = min(start + self.batch_size, len(x_test))
            x_test_batch = x_test[start:end]
            
            if self.distance_metric == 'euclidean':
                distances = self._euclidean_distances(x_train, x_test_batch)
            elif self.distance_metric == 'manhattan':
                distances = self._manhattan_distances(x_train, x_test_batch)
            elif self.distance_metric == 'cosine':
                distances = self._cosine_distances(x_train, x_test_batch)
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
            
            predictions.extend([self._get_neighbors(y_train, distances[i]) for i in range(len(x_test_batch))])
        
        return np.array(predictions)
    
    # Private methods
    def _euclidean_distances(self, x_train, x_test):
        return np.sqrt(np.sum((x_test[:, np.newaxis] - x_train) ** 2, axis=2))
    
    def _manhattan_distances(self, x_train, x_test):
        return np.sum(np.abs(x_test[:, np.newaxis] - x_train), axis=2)
    
    def _cosine_distances(self, x_train, x_test):
        x_train_norm = np.linalg.norm(x_train, axis=1)
        x_test_norm = np.linalg.norm(x_test, axis=1)
        dot_product = np.dot(x_test, x_train.T)
        return 1 - (dot_product / (x_test_norm[:, np.newaxis] * x_train_norm))
    
    def _get_neighbors(self, y_train, distances):
        nearest_indices = np.argpartition(distances, self.k)[:self.k]
        nearest_labels = y_train[nearest_indices]
        
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]
