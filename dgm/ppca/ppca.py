import numpy as np

class PPCA:
    """
    Probabilistic Principal Component Analysis (PPCA) class.

    Parameters:
        n_components (int): Number of principal components.

    Attributes:
        W (numpy.ndarray): Principal component loading matrix.
        sigma_squared (float): Estimated noise variance.
        mean (numpy.ndarray): Mean of the data.

    Methods:
        fit(X, max_iter=100, tol=1e-4): Fit the PPCA model to the input data.
        transform(X): Project data onto the principal components.
        generate_samples(n_samples): Generate new samples from the fitted PPCA model.
    """

    def __init__(self,
                 n_components: int):
        self.n_components = n_components
        self.W = None
        self.sigma_squared = None
        self.mean = None

    def fit(self, X, max_iter=100, tol=1e-4):
        """
        Fit the PPCA model to the input data using the EM algorithm.

        Parameters:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).
            max_iter (int, optional): Maximum number of iterations for the EM algorithm. Default is 100.
            tol (float, optional): Tolerance for convergence of the EM algorithm. Default is 1e-4.

        Returns:
            self (PPCA): The fitted PPCA model.
        """
        N, D = X.shape

        # Initialize parameters
        self.W = np.random.randn(D, self.n_components)
        self.sigma_squared = 1.0
        self.mean = np.mean(X, axis=0)

        for i in range(max_iter):
            # E-step: compute posterior mean and covariance
            M = self.W.T @ self.W + self.sigma_squared * np.eye(self.n_components)
            M_inv = np.linalg.inv(M)
            Z = (X - self.mean) @ self.W @ M_inv

            # M-step: update parameters
            self.W = ((X - self.mean).T @ Z) @ np.linalg.inv(Z.T @ Z + N * self.sigma_squared * M_inv)
            self.sigma_squared = np.mean(np.sum((X - self.mean - Z @ self.W.T) ** 2, axis=1)) / D
            if np.abs(np.sum(Z) - N * self.n_components) < tol:
                break

        return self

    def transform(self, X):
        """
        Project data onto the principal components.

        Parameters:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: The projected data of shape (n_samples, n_components).
        """
        M = self.W.T @ self.W + self.sigma_squared * np.eye(self.n_components)
        M_inv = np.linalg.inv(M)
        return (X - self.mean) @ self.W @ M_inv

    def generate_samples(self, n_samples):
        """
        Generate new samples from the fitted PPCA model.

        Parameters:
            n_samples (int): Number of samples to generate.

        Returns:
            numpy.ndarray: The generated samples of shape (n_samples, n_features).
        """
        z_samples = np.random.randn(n_samples, self.n_components)
        X_samples = self.mean + z_samples @ self.W.T
        return X_samples
