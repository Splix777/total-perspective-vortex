import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import linalg
from scipy.linalg import eigh

from preprocess_data import preprocess_data


class CustomCSP(BaseEstimator, TransformerMixin):
    """
    Custom implementation of the Common Spatial Patterns (CSP).
    CSP is a method to extract features from EEG data that are
    discriminative for a given classification task.

    Parameters:
    - n_components: int, default=4
        The number of CSP components to extract.

    Attributes:
    - filters: ndarray, shape (n_components, n_channels)
        The CSP filters.
    - n_classes: array, shape (n_classes,)
        The unique class labels.
    - mean: ndarray, shape (n_components,)
        The mean of the transformed data.
    - std: ndarray, shape (n_components,)
        The standard deviation of the transformed data.
    """
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters = None
        self.n_classes = None
        self.mean = None
        self.std = None

    def calculate_cov_(self, X, y):
        n_epochs, n_channels, n_times = X.shape
        self.n_classes = np.unique(y)
        covs = []

        for label in self.n_classes:
            # Select epochs corresponding to the current class label
            epochs_for_label = X[np.where(y == label)]
            print(
                f"Label shape (label={label}): {epochs_for_label.shape}")

            # Transpose epochs shape (n_channels, n_epochs_for_label, n_times)
            epochs_for_label = epochs_for_label.transpose([1, 0, 2])
            print(
                f"Label shape after transpose: {epochs_for_label.shape}")

            # Reshape epochs shape (n_channels, n_epochs_for_label * n_times)
            epochs_for_label = epochs_for_label.reshape(n_channels, -1)
            print(
                f"Label shape after reshape: {epochs_for_label.shape}")

            # Calculate covariance matrix
            cov_matrix = np.cov(epochs_for_label)
            print(f"Cov Matrix shape: {cov_matrix.shape}")

            covs.append(cov_matrix)

        # Convert a list of covariance matrices to a 3D numpy array
        covs = np.asarray(covs)
        print(f"covs shape: {covs.shape}")
        return covs

    def calculate_eig_(self, covs):
        eigenvalues, eigenvectors = [], []
        print(f"Eigenvalues: {eigenvalues}")
        print(f"Eigenvectors: {eigenvectors}")

        epsilon = 1e-6

        print(f"Covs length: {len(covs)}")
        for idx, cov in enumerate(covs):
            for iidx, compCov in enumerate(covs):
                if idx < iidx:
                    cov_reg = cov + epsilon * np.eye(cov.shape[0])
                    compCov_reg = compCov + epsilon * np.eye(compCov.shape[0])

                    eigVals, eigVects = linalg.eig(cov_reg, compCov_reg)

                    sorted_indices = np.argsort(np.abs(eigVals))[::-1]

                    eigenvalues.append(eigVals[sorted_indices])
                    eigenvectors.append(eigVects[:, sorted_indices])

        print(f"Eigenvalues Shape: {len(eigenvalues)}")
        print(f"Eigenvectors Shape: {len(eigenvectors)}")
        return eigenvalues, eigenvectors

    def pick_filters(self, eigenvectors):
        filters = None

        for EigVects in eigenvectors:
            print(f"EigVects Shape: {EigVects.shape}")
            if filters is None:
                filters = EigVects[:, :self.n_components]
                print(f"Filters Shape: {filters.shape}")
            else:
                print(f"Filters Shape: {filters.shape}")
                filters = np.concatenate(
                    [filters, EigVects[:, :self.n_components]], axis=1)

        # Transpose the filters matrix and store it in the `filters` attribute
        print(f"Filters Shape: {filters.shape}")
        print(f"Filters Shape after Transpose: {filters.T.shape}")
        self.filters = filters.T

    def transform_epochs(self, X):
        n_epochs, n_channels, n_times = X.shape
        print(f"n_epochs: {n_epochs}")
        print(f"n_channels: {n_channels}")
        print(f"n_times: {n_times}")
        n_filters = self.filters.shape[1]
        print(f"n_filters: {n_filters}")

        transformed_data = np.zeros((n_epochs, n_filters, n_times))
        print(f"Transformed Data Shape: {transformed_data.shape}")

        X = np.asarray([np.dot(self.filters, epoch) for epoch in X])
        X = (X ** 2).mean(axis=2)
        print(f"X Shape: {X.shape}")

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def fit(self, X, y):
        # Covariance matrices for each class
        print(f"X Shape: {X.shape}")
        covs = self.calculate_cov_(X, y)
        eigenvalues, eigenvectors = self.calculate_eig_(covs)

        self.pick_filters(eigenvectors)
        self.transform_epochs(X)

    def transform(self, X):
        # Transform the input data using the selected CSP filters
        X = np.asarray([np.dot(self.filters, epoch) for epoch in X])

        # Square and average along the time axis
        X = (X ** 2).mean(axis=2)

        # Standardize features
        X -= self.mean
        X /= self.std

        return X

    def fit_transform(self, X, y):
        """
        Fit CSP on input data and transform it.

        Parameters:
        - X: ndarray, shape (n_epochs, n_channels, n_times)
            The EEG data.
        - y: array, shape (n_epochs,)
            The labels for each epoch.

        Returns:
        - X_transformed: ndarray, shape (n_epochs, n_components)
            Transformed EEG data using CSP filters.
        """
        self.fit(X, y)
        return self.transform(X)


if __name__ == '__main__':
    for X, y in preprocess_data(subjects=[1]):
        csp = CustomCSP()

    csp.fit(X, y)
