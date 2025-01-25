import numpy as np
import wandb
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from performance_measures import metrics

class MLP_Classifier:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01,activation='relu', optimizer='sgd', epochs=100, task = 'classification', early_stopping=False, patience=15):
        # Model parameters
        self.input_size = input_size
        self.hidden_layers = hidden_layers # List containing number of neurons per hidden layer
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.task = task
        self.early_stopping = early_stopping
        self.patience = patience

        # Weights and biases initialization
        self.weights = []
        self.biases = []
        
        # Initialize the weights and biases
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights and biases based on the layer sizes
        np.random.seed(42) 
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        # print("layer_sizes: ",layer_sizes )
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1  
            b = np.zeros((1, layer_sizes[i + 1]))  # Biases initialized to zero
            self.weights.append(w)
            self.biases.append(b)

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def set_activation(self, activation):
        self.activation = activation

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    # Activation functions
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _tanh(self, z):
        return np.tanh(z)

    def _tanh_derivative(self, z):
        return 1 - np.tanh(z)**2

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def _linear(self, z):
        return z

    def _linear_derivative(self, z):
        return 1

    # Activation function selector
    def _activate(self, z):
        if self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'tanh':
            return self._tanh(z)
        elif self.activation == 'relu':
            return self._relu(z)
        elif self.activation == 'linear':
            return self._linear(z)
        else:
            raise ValueError("Unknown activation function")

    def _activation_derivative(self, z):
        if self.activation == 'sigmoid':
            return self._sigmoid_derivative(z)
        elif self.activation == 'tanh':
            return self._tanh_derivative(z)
        elif self.activation == 'relu':
            return self._relu_derivative(z)
        elif self.activation == 'linear':
            return self._linear_derivative(z)

    # Optimization methods
    def _sgd_update(self, dw, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dw[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def _batch_gradient_descent(self, dw, db):
        self._sgd_update(dw, db)

    def _mini_batch_gradient_descent(self, dw, db):
        self._sgd_update(dw, db)


    def _softmax(self, Z):
        exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # stability correction
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
    
    # Forward propagation
    def _forward_propagation(self, X):
        activations = [X]
        A = X
        zs = []
        for i in range(len(self.weights)-1):
            Z = A.dot(self.weights[i]) + self.biases[i]
            zs.append(Z)
            A = self._activate(Z)
            activations.append(A)

        # Output layer
        Z = A.dot(self.weights[-1]) + self.biases[-1]
        zs.append(Z)
        if self.task == 'classification':
            # Assuming onehot encoded   
            A = self._softmax(Z)
            # A = self._linear(Z)
        elif self.task == 'regression':
            A = self._linear(Z)
        else:
            raise ValueError("Task type not recognized!")

        activations.append(A)
        
        return activations, zs
    
    def _back_propagation(self, X, y, activations, zs):

        m = X.shape[0]
        # List to store gradients of weights and biases
        d_weights = []
        d_biases = []

        # Ensure y is the correct shape
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Start with the derivative of the loss function w.r.t. predictions
        delta = activations[-1] - y

        # Loop over layers in reverse order
        for i in reversed(range(len(self.weights))):
            # Current layer's input
            A_prev = activations[i]

            # Derivative of the weight matrix
            dW = A_prev.T.dot(delta) / m
            # Derivative of the bias vector (sum over samples)
            db = np.sum(delta, axis=0, keepdims=True) / m

            # Store gradients
            d_weights.append(dW)
            d_biases.append(db)

            # If not the first layer, backpropagate the gradient through the activation function
            if i > 0:
                W_current = self.weights[i]
                dZ = delta.dot(W_current.T)
                back_activation_function = self._activation_derivative
                delta = dZ * back_activation_function(A_prev)  # element-wise multiplication

        # Reverse the gradients' list to match the order of self.weights and self.biases
        d_weights = d_weights[::-1]
        d_biases = d_biases[::-1]
        
        
        return d_weights, d_biases

    def fit(self, X, y, X_val=None, y_val=None, wb=False):
        losses = []
        early_stop = False
        n_samples = X.shape[0]
        # Set batch size based on optimizer
        if self.optimizer == 'sgd':
            self.batch_size = 1
        elif self.optimizer == 'mini_batch':
            self.batch_size = min(32, n_samples)
        else:  # batch gradient descent
            self.batch_size = n_samples

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                
                activations, zs = self._forward_propagation(X_batch)
                dw, db = self._back_propagation(X_batch, y_batch, activations, zs)
                
                if self.optimizer in ['sgd', 'mini_batch', 'batch']:
                    self._sgd_update(dw, db)
                
                batch_loss = self._compute_loss(activations[-1], y_batch)
                epoch_loss += batch_loss * len(X_batch)
            epoch_loss /= n_samples
            losses.append(epoch_loss)
            
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_loss = self._compute_loss(val_predictions, y_val)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience and self.early_stopping:
                    # print(f"Early stopping at epoch {epoch + 1}")
                    if wb:
                        y_pred_train = self.predict(X)
                        y_pred_val = self.predict(X_val)
                        train_loss = self._compute_loss(y_pred_train, y)
                        val_loss = self._compute_loss(y_pred_val, y_val)
                        wandb.log({'train_loss': train_loss, 'val_loss': val_loss })
                        wandb.log({'early_stop_epoch': epoch + 1})
                        early_stop = True
                    self.weights = best_weights
                    self.biases = best_biases
                    break
            
            # if (epoch + 1) % 10 == 0:
                # print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

        if wb and not early_stop:
            y_pred_train = self.predict(X)
            y_pred_val = self.predict(X_val)
            train_loss = self._compute_loss(y_pred_train, y)
            val_loss = self._compute_loss(y_pred_val, y_val)
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss })

        return losses
    def _compute_loss(self, predictions, targets):
        if self.task == 'classification':
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        else:  # regression
            return np.mean((predictions - targets) ** 2)


    # Prediction method
    def predict(self, X):
        # Forward propagation only to get predictions
        activations, _ = self._forward_propagation(X)
        return activations[-1]  # Final layer activations (output layer)





    def gradient_check(self, X, y, epsilon=1e-1, tolerance=1e-2):
        """
        Perform comprehensive gradient checking on the MLP model for multiple batches.
        
        Args:
        X (numpy.ndarray): Input data.
        y (numpy.ndarray): True labels.
        epsilon (float): Small perturbation to use in approximation.
        tolerance (float): Tolerance for maximum relative error.
        num_checks (int): Number of batches to check.
        
        Returns:
        bool: True if maximum relative error is within tolerance, False otherwise.
        """
        n_samples = X.shape[0]
        
        # # Determine batch size based on optimizer
        # if self.optimizer == 'sgd':
        #     batch_size = 1
        # elif self.optimizer == 'mini_batch':
        #     batch_size = min(32, n_samples)  # Assuming a default mini-batch size of 32
        # else:  # batch gradient descent
        #     batch_size = n_samples
        #     num_checks = 1  # For batch gradient descent, we only need to check once

        max_rel_error = 0
        min_rel_error = float('inf')
        count1 = 0
        count2 = 0

        for i in range(0, n_samples, self.batch_size):
            X_batch = X[i:i + self.batch_size]
            y_batch = y[i:i + self.batch_size]
            # print(f"Performing gradient check {check + 1}/{num_checks}")
            
            # Forward and backward pass
            activations, zs = self._forward_propagation(X_batch)
            analytical_gradients, _ = self._back_propagation(X_batch, y_batch, activations, zs)
            
            for layer in range(len(self.weights)):
                for i in range(self.weights[layer].shape[0]):
                    for j in range(self.weights[layer].shape[1]):
                        # Compute numerical gradient
                        old_value = self.weights[layer][i, j]
                        
                        # Compute loss with w[i, j] + epsilon
                        self.weights[layer][i, j] = old_value + epsilon
                        activations_plus, _ = self._forward_propagation(X_batch)
                        loss_plus = np.mean((activations_plus[-1] - y_batch) ** 2)
                        
                        # Compute loss with w[i, j] - epsilon
                        self.weights[layer][i, j] = old_value - epsilon
                        activations_minus, _ = self._forward_propagation(X_batch)
                        loss_minus = np.mean((activations_minus[-1] - y_batch) ** 2)
                        
                        # Restore original value
                        self.weights[layer][i, j] = old_value
                        
                        # Compute numerical gradient
                        numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
                        
                        # Compare numerical and analytical gradient
                        analytical_gradient = analytical_gradients[layer][i, j]
                        rel_error = abs(numerical_gradient - analytical_gradient) / (abs(numerical_gradient) + abs(analytical_gradient) + 1e-15)
                        
                        if rel_error > tolerance:
                            count1 += 1
                        else:
                            count2 += 1
                        # Update max and min relative errors
                        max_rel_error = max(max_rel_error, rel_error)
                        min_rel_error = min(min_rel_error, rel_error)

        print(f"Gradient check completed.")
        print(f"Maximum relative error: {max_rel_error}")
        print(f"Minimum relative error: {min_rel_error}")
        print(f"percentage of acceptable gradients: : {(count1/(count1+count2)*100)}" )
        print(f"percentage of unacceptable gradients: : {(count2/(count1+count2)*100)}" )

        if max_rel_error > tolerance:
            print(f"Gradient check failed. Maximum relative error ({max_rel_error}) exceeds tolerance ({tolerance}).")
            return False
        else:
            print(f"Gradient check passed. Maximum relative error ({max_rel_error}) is within tolerance ({tolerance}).")
            return True

class MLP_Regression:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01,activation='relu', optimizer='sgd', epochs=100, task = 'regression', early_stopping=False, patience=15):
        # Model parameters
        self.input_size = input_size
        self.hidden_layers = hidden_layers # List containing number of neurons per hidden layer
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.task = task
        self.early_stopping = early_stopping
        self.patience = patience

        # Weights and biases initialization
        self.weights = []
        self.biases = []
        
        # Initialize the weights and biases
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights and biases based on the layer sizes
        np.random.seed(42) 
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        print("layer_sizes: ",layer_sizes )
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1  
            b = np.zeros((1, layer_sizes[i + 1]))  # Biases initialized to zero
            self.weights.append(w)
            self.biases.append(b)

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def set_activation(self, activation):
        self.activation = activation

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    # Activation functions
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _tanh(self, z):
        return np.tanh(z)

    def _tanh_derivative(self, z):
        return 1 - np.tanh(z)**2

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def _linear(self, z):
        return z

    def _linear_derivative(self, z):
        return 1

    # Activation function selector
    def _activate(self, z):
        if self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'tanh':
            return self._tanh(z)
        elif self.activation == 'relu':
            return self._relu(z)
        elif self.activation == 'linear':
            return self._linear(z)
        else:
            raise ValueError("Unknown activation function")

    def _activation_derivative(self, z):
        if self.activation == 'sigmoid':
            return self._sigmoid_derivative(z)
        elif self.activation == 'tanh':
            return self._tanh_derivative(z)
        elif self.activation == 'relu':
            return self._relu_derivative(z)
        elif self.activation == 'linear':
            return self._linear_derivative(z)

    # Optimization methods
    def _sgd_update(self, dw, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dw[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def _batch_gradient_descent(self, dw, db):
        self._sgd_update(dw, db)

    def _mini_batch_gradient_descent(self, dw, db):
        self._sgd_update(dw, db)


    def _softmax(self, Z):
        exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # stability correction
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
    
    # Forward propagation
    def _forward_propagation(self, X):
        activations = [X]  # Store activations for each layer
        zs = []  # Store z values for backpropagation
        
        # Forward pass through each layer
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            a = self._activate(z)
            activations.append(a)
        
        # activations[-1] = self._softmax(activations[-1])
        return activations, zs
    
    def _back_propagation(self, X, y, activations, zs):
        m = X.shape[0]  # number of samples
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        
        # Output layer error
        delta = np.array(activations[-1] - y)  # Error at the output layer
        
        # Backpropagate through layers
        for l in range(len(self.weights)):
            delt_temp = delta
            
            layer = -(l + 1)
            dw[layer] = np.dot(activations[layer-1].T, delta) / m
            db[layer] = np.sum(delta, axis=0, keepdims=True) / m
            
            if l < len(self.weights) - 1:  # If not at the first layer
                z = zs[layer-1]
                delta = np.dot(delta, self.weights[layer].T) * self._activation_derivative(z)
        return dw, db

    def fit(self, X, y, X_val=None, y_val=None, wb=False):
        losses = []
        early_stop = False
        n_samples = X.shape[0]
        # Set batch size based on optimizer
        if self.optimizer == 'sgd':
            self.batch_size = 1
        elif self.optimizer == 'mini_batch':
            self.batch_size = min(32, n_samples)
        else:  # batch gradient descent
            self.batch_size = n_samples

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                
                activations, zs = self._forward_propagation(X_batch)
                dw, db = self._back_propagation(X_batch, y_batch, activations, zs)
                
                if self.optimizer in ['sgd', 'mini_batch', 'batch']:
                    self._sgd_update(dw, db)
                
                batch_loss = self.binary_cross_entropy(activations[-1], y_batch)
                epoch_loss += batch_loss * len(X_batch)
            
            epoch_loss /= n_samples
            losses.append(epoch_loss)

            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_loss = self._compute_loss(val_predictions, y_val)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience and self.early_stopping:
                    print(f"Early stopping at epoch {epoch + 1}")
                    if wb:
                        y_pred_train = self.predict(X)
                        y_pred_val = self.predict(X_val)
                        train_mse = self._compute_loss(y_pred_train, y)
                        val_mse = self._compute_loss(y_pred_val, y_val)
                        train_rmse = np.sqrt(train_mse)
                        val_rmse = np.sqrt(val_mse)
                        train_r2 = metrics.r_squared(y, y_pred_train)
                        val_r2 = metrics.r_squared(y_val, y_pred_val)
                        
                        wandb.log({'train_mse': train_mse, 'val_mse': val_mse, 
                                'train_rmse': train_rmse, 'val_rmse': val_rmse, 
                                'train_r2': train_r2, 'val_r2': val_r2})
                        wandb.log({'early_stop_epoch': epoch + 1})
                        early_stop = True
                    self.weights = best_weights
                    self.biases = best_biases
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

        if wb and not early_stop:
            y_pred_train = self.predict(X)
            y_pred_val = self.predict(X_val)
            train_mse = self._compute_loss(y_pred_train, y)
            val_mse = self._compute_loss(y_pred_val, y_val)
            train_rmse = np.sqrt(train_mse)
            val_rmse = np.sqrt(val_mse)
            train_r2 = metrics.r_squared(y, y_pred_train)
            val_r2 = metrics.r_squared(y_val, y_pred_val)
            
            wandb.log({'train_mse': train_mse, 'val_mse': val_mse, 
                       'train_rmse': train_rmse, 'val_rmse': val_rmse, 
                       'train_r2': train_r2, 'val_r2': val_r2})
            
        # return losses

    def _compute_loss(self, predictions, targets):
        if self.task == 'classification':
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        else:  # regression
            return np.mean((predictions - targets) ** 2)

    def binary_cross_entropy(self, predictions, targets, epsilon=1e-15):
        # Clip predictions to avoid log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        # BCE loss calculation
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    # Prediction method
    def predict(self, X):
        # Forward propagation only to get predictions
        activations, _ = self._forward_propagation(X)
        return activations[-1]  # Final layer activations (output layer)





    def gradient_check(self, X, y, epsilon=1e-1, tolerance=1e-2):
        """
        Perform comprehensive gradient checking on the MLP model for multiple batches.
        
        Args:
        X (numpy.ndarray): Input data.
        y (numpy.ndarray): True labels.
        epsilon (float): Small perturbation to use in approximation.
        tolerance (float): Tolerance for maximum relative error.
        num_checks (int): Number of batches to check.
        
        Returns:
        bool: True if maximum relative error is within tolerance, False otherwise.
        """
        n_samples = X.shape[0]
        
        # # Determine batch size based on optimizer
        # if self.optimizer == 'sgd':
        #     batch_size = 1
        # elif self.optimizer == 'mini_batch':
        #     batch_size = min(32, n_samples)  # Assuming a default mini-batch size of 32
        # else:  # batch gradient descent
        #     batch_size = n_samples
        #     num_checks = 1  # For batch gradient descent, we only need to check once

        max_rel_error = 0
        min_rel_error = float('inf')
        count1 = 0
        count2 = 0

        for i in range(0, n_samples, self.batch_size):
            X_batch = X[i:i + self.batch_size]
            y_batch = y[i:i + self.batch_size]
            # print(f"Performing gradient check {check + 1}/{num_checks}")
            
            # Forward and backward pass
            activations, zs = self._forward_propagation(X_batch)
            analytical_gradients, _ = self._back_propagation(X_batch, y_batch, activations, zs)
            
            for layer in range(len(self.weights)):
                for i in range(self.weights[layer].shape[0]):
                    for j in range(self.weights[layer].shape[1]):
                        # Compute numerical gradient
                        old_value = self.weights[layer][i, j]
                        
                        # Compute loss with w[i, j] + epsilon
                        self.weights[layer][i, j] = old_value + epsilon
                        activations_plus, _ = self._forward_propagation(X_batch)
                        loss_plus = np.mean((activations_plus[-1] - y_batch) ** 2)
                        
                        # Compute loss with w[i, j] - epsilon
                        self.weights[layer][i, j] = old_value - epsilon
                        activations_minus, _ = self._forward_propagation(X_batch)
                        loss_minus = np.mean((activations_minus[-1] - y_batch) ** 2)
                        
                        # Restore original value
                        self.weights[layer][i, j] = old_value
                        
                        # Compute numerical gradient
                        numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
                        
                        # Compare numerical and analytical gradient
                        analytical_gradient = analytical_gradients[layer][i, j]
                        rel_error = abs(numerical_gradient - analytical_gradient) / (abs(numerical_gradient) + abs(analytical_gradient) + 1e-15)
                        
                        if rel_error > tolerance:
                            count1 += 1
                        else:
                            count2 += 1
                        # Update max and min relative errors
                        max_rel_error = max(max_rel_error, rel_error)
                        min_rel_error = min(min_rel_error, rel_error)

        print(f"Gradient check completed.")
        print(f"Maximum relative error: {max_rel_error}")
        print(f"Minimum relative error: {min_rel_error}")
        print(f"percentage of acceptable gradients: : {(count1/(count1+count2)*100)}" )
        print(f"percentage of unacceptable gradients: : {(count2/(count1+count2)*100)}" )

        if max_rel_error > tolerance:
            print(f"Gradient check failed. Maximum relative error ({max_rel_error}) exceeds tolerance ({tolerance}).")
            return False
        else:
            print(f"Gradient check passed. Maximum relative error ({max_rel_error}) is within tolerance ({tolerance}).")
            return True





class MLP_multi_Classifier:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, optimizer='sgd', epochs=100, early_stopping=False, patience=15):
        # Model parameters
        self.input_size = input_size
        self.hidden_layers = hidden_layers # List containing number of neurons per hidden layer
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = 'sigmoid' # fixed for multi class classification
        self.optimizer = optimizer
        self.epochs = epochs
        self.task = 'classification'
        self.early_stopping = early_stopping
        self.patience = patience

        # Weights and biases initialization
        self.weights = []
        self.biases = []
        
        # Initialize the weights and biases
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights and biases based on the layer sizes
        np.random.seed(42) 
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        # print("layer_sizes: ",layer_sizes )
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1  
            b = np.zeros((1, layer_sizes[i + 1]))  # Biases initialized to zero
            self.weights.append(w)
            self.biases.append(b)

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def set_activation(self, activation):
        self.activation = activation

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    # Activation functions
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _tanh(self, z):
        return np.tanh(z)

    def _tanh_derivative(self, z):
        return 1 - np.tanh(z)**2

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def _linear(self, z):
        return z

    def _linear_derivative(self, z):
        return 1

    # Activation function selector
    def _activate(self, z):
        if self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'tanh':
            return self._tanh(z)
        elif self.activation == 'relu':
            return self._relu(z)
        elif self.activation == 'linear':
            return self._linear(z)
        else:
            raise ValueError("Unknown activation function")

    def _activation_derivative(self, z):
        if self.activation == 'sigmoid':
            return self._sigmoid_derivative(z)
        elif self.activation == 'tanh':
            return self._tanh_derivative(z)
        elif self.activation == 'relu':
            return self._relu_derivative(z)
        elif self.activation == 'linear':
            return self._linear_derivative(z)

    # Optimization methods
    def _sgd_update(self, dw, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dw[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def _batch_gradient_descent(self, dw, db):
        self._sgd_update(dw, db)

    def _mini_batch_gradient_descent(self, dw, db):
        self._sgd_update(dw, db)


    def _softmax(self, Z):
        exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # stability correction
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
    
    # Forward propagation
    def _forward_propagation(self, X):
        activations = [X]
        A = X
        zs = []
        for i in range(len(self.weights)-1):
            Z = A.dot(self.weights[i]) + self.biases[i]
            zs.append(Z)
            A = self._activate(Z)
            activations.append(A)

        # Output layer
        Z = A.dot(self.weights[-1]) + self.biases[-1]
        zs.append(Z)
        if self.task == 'classification':
            # Assuming onehot encoded   
            A = self._softmax(Z)
            # A = self._linear(Z)
        elif self.task == 'regression':
            A = self._linear(Z)
        else:
            raise ValueError("Task type not recognized!")

        activations.append(A)
        
        return activations, zs
    
    def _back_propagation(self, X, y, activations, zs):

        m = X.shape[0]
        # List to store gradients of weights and biases
        d_weights = []
        d_biases = []

        # Ensure y is the correct shape
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Start with the derivative of the loss function w.r.t. predictions
        delta = activations[-1] - y

        # Loop over layers in reverse order
        for i in reversed(range(len(self.weights))):
            # Current layer's input
            A_prev = activations[i]

            # Derivative of the weight matrix
            dW = A_prev.T.dot(delta) / m
            # Derivative of the bias vector (sum over samples)
            db = np.sum(delta, axis=0, keepdims=True) / m

            # Store gradients
            d_weights.append(dW)
            d_biases.append(db)

            # If not the first layer, backpropagate the gradient through the activation function
            if i > 0:
                W_current = self.weights[i]
                dZ = delta.dot(W_current.T)
                back_activation_function = self._activation_derivative
                delta = dZ * back_activation_function(A_prev)  # element-wise multiplication

        # Reverse the gradients' list to match the order of self.weights and self.biases
        d_weights = d_weights[::-1]
        d_biases = d_biases[::-1]
        
        
        return d_weights, d_biases

    def fit(self, X, y, X_val=None, y_val=None, wb=False):
        losses = []
        early_stop = False
        n_samples = X.shape[0]
        # Set batch size based on optimizer
        if self.optimizer == 'sgd':
            self.batch_size = 1
        elif self.optimizer == 'mini_batch':
            self.batch_size = min(32, n_samples)
        else:  # batch gradient descent
            self.batch_size = n_samples

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                
                activations, zs = self._forward_propagation(X_batch)
                dw, db = self._back_propagation(X_batch, y_batch, activations, zs)
                
                if self.optimizer in ['sgd', 'mini_batch', 'batch']:
                    self._sgd_update(dw, db)
                
                batch_loss = self._compute_loss(activations[-1], y_batch)
                epoch_loss += batch_loss * len(X_batch)
            epoch_loss /= n_samples
            losses.append(epoch_loss)
            
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_loss = self._compute_loss(val_predictions, y_val)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience and self.early_stopping:
                    # print(f"Early stopping at epoch {epoch + 1}")
                    if wb:
                        y_pred_train = self.predict(X)
                        y_pred_val = self.predict(X_val)
                        train_loss = self._compute_loss(y_pred_train, y)
                        val_loss = self._compute_loss(y_pred_val, y_val)
                        wandb.log({'train_loss': train_loss, 'val_loss': val_loss })
                        wandb.log({'early_stop_epoch': epoch + 1})
                        early_stop = True
                    self.weights = best_weights
                    self.biases = best_biases
                    break
            
            # if (epoch + 1) % 10 == 0:
                # print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

        if wb and not early_stop:
            y_pred_train = self.predict(X)
            y_pred_val = self.predict(X_val)
            train_loss = self._compute_loss(y_pred_train, y)
            val_loss = self._compute_loss(y_pred_val, y_val)
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss })

        # return losses
    def _compute_loss(self, predictions, targets):
        if self.task == 'classification':
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        else:  # regression
            return np.mean((predictions - targets) ** 2)




    # Prediction method
    def predict(self, X, threshold=0.5):
        # Forward propagation to get predictions
        activations, _ = self._forward_propagation(X)

        # Get predicted probabilities from the last layer
        predicted_probs = activations[-1]  # Shape: (num_samples, num_classes)

        # Initialize a list to hold the labels for each sample
        labels = []

        # Assign labels based on the threshold
        for i in range(predicted_probs.shape[0]):
            class_indices = np.where(predicted_probs[i] >= threshold)[0]
            # Store the 1-based indices of classes that exceed the threshold
            labels.append(class_indices + 1)  # Add 1 for 1-based indexing

        return labels 






    def gradient_check(self, X, y, epsilon=1e-1, tolerance=1e-2):
        """
        Perform comprehensive gradient checking on the MLP model for multiple batches.
        
        Args:
        X (numpy.ndarray): Input data.
        y (numpy.ndarray): True labels.
        epsilon (float): Small perturbation to use in approximation.
        tolerance (float): Tolerance for maximum relative error.
        num_checks (int): Number of batches to check.
        
        Returns:
        bool: True if maximum relative error is within tolerance, False otherwise.
        """
        n_samples = X.shape[0]
        max_rel_error = 0
        min_rel_error = float('inf')
        count1 = 0
        count2 = 0

        for i in range(0, n_samples, self.batch_size):
            X_batch = X[i:i + self.batch_size]
            y_batch = y[i:i + self.batch_size]
            # print(f"Performing gradient check {check + 1}/{num_checks}")
            
            # Forward and backward pass
            activations, zs = self._forward_propagation(X_batch)
            analytical_gradients, _ = self._back_propagation(X_batch, y_batch, activations, zs)
            
            for layer in range(len(self.weights)):
                for i in range(self.weights[layer].shape[0]):
                    for j in range(self.weights[layer].shape[1]):
                        # Compute numerical gradient
                        old_value = self.weights[layer][i, j]
                        
                        # Compute loss with w[i, j] + epsilon
                        self.weights[layer][i, j] = old_value + epsilon
                        activations_plus, _ = self._forward_propagation(X_batch)
                        loss_plus = np.mean((activations_plus[-1] - y_batch) ** 2)
                        
                        # Compute loss with w[i, j] - epsilon
                        self.weights[layer][i, j] = old_value - epsilon
                        activations_minus, _ = self._forward_propagation(X_batch)
                        loss_minus = np.mean((activations_minus[-1] - y_batch) ** 2)
                        
                        # Restore original value
                        self.weights[layer][i, j] = old_value
                        
                        # Compute numerical gradient
                        numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
                        
                        # Compare numerical and analytical gradient
                        analytical_gradient = analytical_gradients[layer][i, j]
                        rel_error = abs(numerical_gradient - analytical_gradient) / (abs(numerical_gradient) + abs(analytical_gradient) + 1e-15)
                        
                        if rel_error > tolerance:
                            count1 += 1
                        else:
                            count2 += 1
                        # Update max and min relative errors
                        max_rel_error = max(max_rel_error, rel_error)
                        min_rel_error = min(min_rel_error, rel_error)

        print(f"Gradient check completed.")
        print(f"Maximum relative error: {max_rel_error}")
        print(f"Minimum relative error: {min_rel_error}")
        print(f"percentage of acceptable gradients: : {(count1/(count1+count2)*100)}" )
        print(f"percentage of unacceptable gradients: : {(count2/(count1+count2)*100)}" )

        if max_rel_error > tolerance:
            print(f"Gradient check failed. Maximum relative error ({max_rel_error}) exceeds tolerance ({tolerance}).")
            return False
        else:
            print(f"Gradient check passed. Maximum relative error ({max_rel_error}) is within tolerance ({tolerance}).")
            return True





    
class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01,activation='relu', optimizer='sgd', epochs=100, task = 'regression', early_stopping=False, patience=15):
        # Model parameters
        self.input_size = input_size
        self.hidden_layers = hidden_layers # List containing number of neurons per hidden layer
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.task = task
        self.early_stopping = early_stopping
        self.patience = patience

        # Weights and biases initialization
        self.weights = []
        self.biases = []
        
        # Initialize the weights and biases
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights and biases based on the layer sizes
        np.random.seed(42) 
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        print("layer_sizes: ",layer_sizes )
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1  
            b = np.zeros((1, layer_sizes[i + 1]))  # Biases initialized to zero
            self.weights.append(w)
            self.biases.append(b)

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def set_activation(self, activation):
        self.activation = activation

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    # Activation functions
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _tanh(self, z):
        return np.tanh(z)

    def _tanh_derivative(self, z):
        return 1 - np.tanh(z)**2

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def _linear(self, z):
        return z

    def _linear_derivative(self, z):
        return 1

    # Activation function selector
    def _activate(self, z):
        if self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'tanh':
            return self._tanh(z)
        elif self.activation == 'relu':
            return self._relu(z)
        elif self.activation == 'linear':
            return self._linear(z)
        else:
            raise ValueError("Unknown activation function")

    def _activation_derivative(self, z):
        if self.activation == 'sigmoid':
            return self._sigmoid_derivative(z)
        elif self.activation == 'tanh':
            return self._tanh_derivative(z)
        elif self.activation == 'relu':
            return self._relu_derivative(z)
        elif self.activation == 'linear':
            return self._linear_derivative(z)

    # Optimization methods
    def _sgd_update(self, dw, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dw[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def _batch_gradient_descent(self, dw, db):
        self._sgd_update(dw, db)

    def _mini_batch_gradient_descent(self, dw, db):
        self._sgd_update(dw, db)


    def _softmax(self, Z):
        exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # stability correction
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    


    # Forward propagation
    def _forward_propagation(self, X):

        if self.task == "classification":
            activations = [X]
            A = X
            zs = []
            for i in range(len(self.weights)-1):
                Z = A.dot(self.weights[i]) + self.biases[i]
                zs.append(Z)
                A = self._activate(Z)
                activations.append(A)

            # Output layer
            Z = A.dot(self.weights[-1]) + self.biases[-1]
            zs.append(Z)
            if self.task == 'classification':
                # Assuming onehot encoded   
                A = self._softmax(Z)
                # A = self._linear(Z)
            elif self.task == 'regression':
                A = self._linear(Z)
            else:
                raise ValueError("Task type not recognized!")

            activations.append(A)
            
            return activations, zs
        
        else:
            activations = [X]  # Store activations for each layer
            zs = []  # Store z values for backpropagation
            
            # Forward pass through each layer
            for i in range(len(self.weights)):
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                zs.append(z)
                a = self._activate(z)
                activations.append(a)
            
            # activations[-1] = self._softmax(activations[-1])
            return activations, zs
    
    def _back_propagation(self, X, y, activations, zs):


        if(self.task == "classification"):

            m = X.shape[0]
            # List to store gradients of weights and biases
            d_weights = []
            d_biases = []

            # Ensure y is the correct shape
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            # Start with the derivative of the loss function w.r.t. predictions
            delta = activations[-1] - y

            # Loop over layers in reverse order
            for i in reversed(range(len(self.weights))):
                # Current layer's input
                A_prev = activations[i]

                # Derivative of the weight matrix
                dW = A_prev.T.dot(delta) / m
                # Derivative of the bias vector (sum over samples)
                db = np.sum(delta, axis=0, keepdims=True) / m

                # Store gradients
                d_weights.append(dW)
                d_biases.append(db)

                # If not the first layer, backpropagate the gradient through the activation function
                if i > 0:
                    W_current = self.weights[i]
                    dZ = delta.dot(W_current.T)
                    back_activation_function = self._activation_derivative
                    delta = dZ * back_activation_function(A_prev)  # element-wise multiplication

            # Reverse the gradients' list to match the order of self.weights and self.biases
            d_weights = d_weights[::-1]
            d_biases = d_biases[::-1]
            
            
            return d_weights, d_biases

        else:

            m = X.shape[0]  # number of samples
            dw = [np.zeros(w.shape) for w in self.weights]
            db = [np.zeros(b.shape) for b in self.biases]
            
            # Output layer error
            delta = np.array(activations[-1] - y)  # Error at the output layer
            
            # Backpropagate through layers
            for l in range(len(self.weights)):
                delt_temp = delta
                
                layer = -(l + 1)
                dw[layer] = np.dot(activations[layer-1].T, delta) / m
                db[layer] = np.sum(delta, axis=0, keepdims=True) / m
                
                if l < len(self.weights) - 1:  # If not at the first layer
                    z = zs[layer-1]
                    delta = np.dot(delta, self.weights[layer].T) * self._activation_derivative(z)
            return dw, db

    def fit(self, X, y, X_val=None, y_val=None, wb=False):
        losses = []
        early_stop = False
        n_samples = X.shape[0]
        # Set batch size based on optimizer
        if self.optimizer == 'sgd':
            self.batch_size = 1
        elif self.optimizer == 'mini_batch':
            self.batch_size = min(32, n_samples)
        else:  # batch gradient descent
            self.batch_size = n_samples

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                
                activations, zs = self._forward_propagation(X_batch)
                dw, db = self._back_propagation(X_batch, y_batch, activations, zs)
                
                if self.optimizer in ['sgd', 'mini_batch', 'batch']:
                    self._sgd_update(dw, db)
                
                batch_loss = self.binary_cross_entropy(activations[-1], y_batch)
                epoch_loss += batch_loss * len(X_batch)
            
            epoch_loss /= n_samples
            losses.append(epoch_loss)

            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_loss = self._compute_loss(val_predictions, y_val)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience and self.early_stopping:
                    print(f"Early stopping at epoch {epoch + 1}")
                    if wb:
                        y_pred_train = self.predict(X)
                        y_pred_val = self.predict(X_val)
                        train_mse = self._compute_loss(y_pred_train, y)
                        val_mse = self._compute_loss(y_pred_val, y_val)
                        train_rmse = np.sqrt(train_mse)
                        val_rmse = np.sqrt(val_mse)
                        train_r2 = metrics.r_squared(y, y_pred_train)
                        val_r2 = metrics.r_squared(y_val, y_pred_val)
                        
                        wandb.log({'train_mse': train_mse, 'val_mse': val_mse, 
                                'train_rmse': train_rmse, 'val_rmse': val_rmse, 
                                'train_r2': train_r2, 'val_r2': val_r2})
                        wandb.log({'early_stop_epoch': epoch + 1})
                        early_stop = True
                    self.weights = best_weights
                    self.biases = best_biases
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

        if wb and not early_stop:
            y_pred_train = self.predict(X)
            y_pred_val = self.predict(X_val)
            train_mse = self._compute_loss(y_pred_train, y)
            val_mse = self._compute_loss(y_pred_val, y_val)
            train_rmse = np.sqrt(train_mse)
            val_rmse = np.sqrt(val_mse)
            train_r2 = metrics.r_squared(y, y_pred_train)
            val_r2 = metrics.r_squared(y_val, y_pred_val)
            
            wandb.log({'train_mse': train_mse, 'val_mse': val_mse, 
                       'train_rmse': train_rmse, 'val_rmse': val_rmse, 
                       'train_r2': train_r2, 'val_r2': val_r2})
            
        # return losses

    def _compute_loss(self, predictions, targets):
        if self.task == 'classification':
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        else:  # regression
            return np.mean((predictions - targets) ** 2)

    def binary_cross_entropy(self, predictions, targets, epsilon=1e-15):
        # Clip predictions to avoid log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        # BCE loss calculation
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    # Prediction method
    def predict(self, X):
        # Forward propagation only to get predictions
        activations, _ = self._forward_propagation(X)
        return activations[-1]  # Final layer activations (output layer)


