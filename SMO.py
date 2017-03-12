import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline
# This line is only needed if you have a HiDPI display
# %config InlineBackend.figure_format = 'retina'

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler


class SMOModel:
    """Container object for the model used for sequential minimal optimization."""

    def __init__(self, X, y, C, kernel, alphas, b, errors):
        self.X = X  # training data vector
        self.y = y  # class label vector
        self.C = C  # regularization parameter
        self.kernel = kernel  # kernel function
        self.alphas = alphas  # lagrange multiplier vector
        self.b = b  # scalar bias term
        self.errors = errors  # error cache
        self._obj = []  # record of objective function value
        self.m = len(self.X)  # store size of training set

    # def linear_kernel(x, y, b=1):
    #     """Returns the linear combination of arrays `x` and `y` with
    #     the optional bias term `b` (set to 1 by default)."""
    #
    #     return x @ y.T + b  # Note the @ operator for matrix multiplication
    #
    def gaussian_kernel(x, y, sigma=1):
        """Returns the gaussian similarity of arrays `x` and `y` with
        kernel width parameter `sigma` (set to 1 by default)."""

        if np.ndim(x) == 1 and np.ndim(y) == 1:
            result = np.exp(- np.linalg.norm(x - y) / (2 * sigma ** 2))
        elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
            result = np.exp(- np.linalg.norm(x - y, axis=1) / (2 * sigma ** 2))
        elif np.ndim(x) > 1 and np.ndim(y) > 1:
            result = np.exp(- np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], axis=2) / (2 * sigma ** 2))
        return result


    def objective_function(alphas, target, kernel, X_train):
        """Returns the SVM objective function based in the input model defined by:
        `alphas`: vector of Lagrange multipliers
        `target`: vector of class labels (-1 or 1) for training data
        `kernel`: kernel function
        `X_train`: training data for model."""

        return np.sum(alphas) - 0.5 * np.sum(target * target * kernel(X_train, X_train) * alphas * alphas)


# Decision function

def decision_function(alphas, target, kernel, X_train, x_test, b):
    """Applies the SVM decision function to the input feature vectors in `x_test`."""

    result = (alphas * target) @ kernel(X_train, x_test) - b
    return result

if __name__ == '__main__':
    # smo = SMOModel()
    x_len, y_len = 5, 10
    # smo.linear_kernel(np.random.rand(x_len, 1), np.random.rand(y_len, 1)).shape == (x_len, y_len)
    # smo.gaussian_kernel(np.random.rand(x_len, 1), np.random.rand(y_len, 1)).shape == (5, 10)
