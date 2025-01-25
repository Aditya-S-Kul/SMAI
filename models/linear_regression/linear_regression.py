import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from performance_measures import metrics

class LinearRegression:
    def __init__(self, degree=1, learning_rate=0.01, tolerance=1e-4, regularization=0.0, reg_type='l2'):
        self.degree = degree
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.regularization = regularization
        self.reg_type = reg_type
        self.coefficients = np.random.uniform(0, 10, degree + 1)
        self.history = []  # Store the loss and coefficients at each iteration

    def polynomial_features(self, X):
        """Generate polynomial features for input data X up to the given degree."""
        return np.vstack([X**i for i in range(self.degree + 1)]).T

    def fit(self, X, y):
        """Fit the Linear regression model using gradient descent."""
        X_poly = self.polynomial_features(X)
        n = len(y)
        previous_loss = float('inf')
        iteration = 0
        
        while True:
            y_pred = X_poly.dot(self.coefficients)
            loss = metrics.calculate_mse(y, y_pred)
            
            # Store progress
            self.history.append({
                'iteration': iteration,
                'coefficients': self.coefficients.copy(),
                'loss': loss,
                'mse': loss,
                'variance': metrics.calculate_variance(y, y_pred),
                'std_dev': metrics.calculate_std(y, y_pred)
            })
            
            # Check for convergence
            if abs(previous_loss - loss) < self.tolerance:
                break
            
            previous_loss = loss
            
            # Gradient descent step
            gradients = (-2/n) * X_poly.T.dot(y - y_pred)
            if self.reg_type == 'l2':
                gradients += 2 * self.regularization * self.coefficients
            elif self.reg_type == 'l1':
                gradients += self.regularization * np.sign(self.coefficients)
                
            self.coefficients -= self.learning_rate * gradients
            iteration += 1

    def predict(self, X):
        """Predict using the fitted Linear regression model."""
        X_poly = self.polynomial_features(X)
        return X_poly.dot(self.coefficients)

    # def calculate_mse(self, y_true, y_pred):
    #     """Calculate Mean Squared Error."""
    #     return np.mean((y_true - y_pred) ** 2)

    # def calculate_std(self, y_true, y_pred):
    #     """Calculate Standard Deviation of the residuals."""
    #     return np.std(y_true - y_pred)

    # def calculate_variance(self, y_true, y_pred):
    #     """Calculate Variance of the residuals."""
    #     return np.var(y_true - y_pred)

    def report_metrics(self, y_true, y_pred):
        """Report MSE, variance, and standard deviation."""
        mse = metrics.calculate_mse(y_true, y_pred)
        variance = metrics.calculate_variance(y_true, y_pred)
        std_dev = metrics.calculate_std(y_true, y_pred)
        return mse, variance, std_dev

    def plot_fit(self, X, y):
        """Plot the training data and the fitted polynomial."""
        # Sort the values for smooth plotting
        sort_indices = np.argsort(X)
        X_sorted = X[sort_indices]
        y_pred = self.predict(X_sorted)
        
        plt.scatter(X, y, color='blue', label='Training Data', alpha=0.6)
        plt.plot(X_sorted, y_pred, color='red', label=f'Fitted Polynomial (Degree {self.degree}), {self.reg_type.upper()} Regularization')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'Linear Regression Fit (Degree {self.degree})')
        plt.show()

    def save_model(self, file_path):
        """Save the model's coefficients to a file."""
        model_data = {
            'degree': self.degree,
            'coefficients': self.coefficients.tolist(),
            'reg_type': self.reg_type,
            'regularization': self.regularization
        }
        with open(file_path, 'w') as f:
            json.dump(model_data, f)

    @classmethod
    def load_model(cls, file_path):
        """Load the model's coefficients from a file."""
        with open(file_path, 'r') as f:
            model_data = json.load(f)
        model = cls(degree=model_data['degree'], reg_type=model_data['reg_type'], regularization=model_data['regularization'])
        model.coefficients = np.array(model_data['coefficients'])
        return model

    def plot_fit_and_metrics(self, X, y, index, save_path):
        """Plot the training data and the fitted polynomial, including metrics."""
        plt.figure(figsize=(12, 10))
        
        # Plot the original data and the fitted line
        plt.subplot(2, 2, 1)
        sort_indices = np.argsort(X)
        X_sorted = X[sort_indices]
        y_pred = self.predict(X_sorted)
        plt.scatter(X, y, color='blue', label='Training Data', alpha=0.6)
        plt.plot(X_sorted, y_pred, color='red', label=f'Fitted Polynomial (Degree {self.degree}), {self.reg_type.upper()} Regularization')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression Fit')
        plt.legend()
        
        # Plot the metrics
        mse = self.history[index]['mse']
        variance = self.history[index]['variance']
        std_dev = self.history[index]['std_dev']
        
        plt.subplot(2, 2, 2)
        plt.bar(['MSE'], [mse], color='blue')
        plt.title('Mean Squared Error')
        
        plt.subplot(2, 2, 3)
        plt.bar(['Variance'], [variance], color='green')
        plt.title('Variance')
        
        plt.subplot(2, 2, 4)
        plt.bar(['Standard Deviation'], [std_dev], color='red')
        plt.title('Standard Deviation')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def save_gif(self, X, y, output_path):
        """Save the animation of the fitting process as a GIF."""
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        
        def update(i):

            if i % 5 != 0:
                return
        
            axs[0, 0].clear()
            axs[0, 1].clear()
            axs[1, 0].clear()
            axs[1, 1].clear()
            
            # Plot the data and the line being fitted
            axs[0, 0].scatter(X, y, color='blue', label='Data')
            x_line = np.linspace(min(X), max(X), 100)  # Increase the number of points for smooth plotting
            X_line_poly = self.polynomial_features(x_line)
            y_line = X_line_poly.dot(self.history[i]['coefficients'])
            axs[0, 0].plot(x_line, y_line, color='black', label=f'Fit Degree {self.degree}')
            axs[0, 0].set_title(f"Iteration {i + 1}")
            axs[0, 0].legend()
            
            # Plot MSE
            axs[0, 1].plot([h['iteration'] for h in self.history[:i+1]], [h['mse'] for h in self.history[:i+1]], color='red')
            axs[0, 1].set_title("MSE")
            
            # Plot Variance
            axs[1, 0].plot([h['iteration'] for h in self.history[:i+1]], [h['variance'] for h in self.history[:i+1]], color='green')
            axs[1, 0].set_title("Variance")
            
            # Plot Standard Deviation
            axs[1, 1].plot([h['iteration'] for h in self.history[:i+1]], [h['std_dev'] for h in self.history[:i+1]], color='purple')
            axs[1, 1].set_title("Standard Deviation")
            
            plt.tight_layout()

        ani = FuncAnimation(fig, update, frames=len(self.history), repeat=False, interval = 50)
        ani.save(output_path, writer='pillow', fps=60) 

    plt.close()
