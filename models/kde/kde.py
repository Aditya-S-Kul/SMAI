import numpy as np
import matplotlib.pyplot as plt # type: ignore

class KDE:
    def __init__(self, kernel='gaussian', bandwidth=0.2):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit the KDE model to the data."""
        self.data = data

    def kernel_function(self, distance):
        """Select and compute the kernel function."""
        if self.kernel == 'box':
            # Box kernel: 1 if within bandwidth, 0 otherwise
            return np.where(np.abs(distance) <= 1, 0.5, 0)
        elif self.kernel == 'gaussian':
            # Gaussian kernel: exp(-0.5 * distance^2) / sqrt(2 * pi)
            return np.exp(-0.5 * distance**2) / np.sqrt(2 * np.pi)
        elif self.kernel == 'triangular':
            # Triangular kernel: 1 - |distance| if within bandwidth, 0 otherwise
            return np.where(np.abs(distance) <= 1, 1 - np.abs(distance), 0)
        else:
            raise ValueError("Unsupported kernel type")

    def predict(self, points):
        """Calculate the density at given points."""
        densities = np.zeros(points.shape[0])
        n = self.data.shape[0]
        
        for i, point in enumerate(points):
            # Compute distances and apply the kernel function
            distances = np.linalg.norm((self.data - point) / self.bandwidth, axis=1)
            kernel_values = self.kernel_function(distances)
            densities[i] = np.sum(kernel_values) / (n * self.bandwidth**self.data.shape[1])
        
        return densities

    def visualize(self, x_min=-4, x_max=4, y_min=-4, y_max=4, grid_size=100, num_contours=10):
        """Visualize the density over a 2D grid with contour lines."""
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x, y)
        points = np.vstack([X.ravel(), Y.ravel()]).T
        Z = self.predict(points).reshape(X.shape)

        plt.figure(figsize=(6, 6))

        # Create filled contour plot
        contour_plot = plt.contourf(X, Y, Z, levels=20, cmap="coolwarm")
        plt.scatter(self.data[:, 0], self.data[:, 1], s=1, color="black", alpha=0.5)
        plt.colorbar(contour_plot, label="Density")

        # Create contour lines to represent constant densities
        contour_levels = np.linspace(Z.min(), Z.max(), num_contours)
        plt.contour(X, Y, Z, levels=contour_levels, colors='black', linewidths=1)
        # plt.scatter(self.data[:, 0], self.data[:, 1], s=1, color="black", alpha=0.5)
        plt.title(f"{self.kernel.capitalize()} Kernel Density Estimate")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

