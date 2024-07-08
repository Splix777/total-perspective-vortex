import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import linalg
from itertools import combinations

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

    def _calculate_cov(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the covariance matrices for each class.
        Covariance matrices are used to calculate the eigenvalues
        and eigenvectors for each class. In simple terms, the
        covariance matrix represents the relationship between
        different channels in the EEG data.

        Args:
            - X: ndarray, shape (n_epochs, n_channels, n_times)
                The EEG data.
            - y: array, shape (n_epochs, )
                The labels for each epoch.

        Returns:
            - covs: ndarray, shape (n_classes, n_channels, n_channels)
                The covariance matrices for each class.
        """
        n_epochs, n_channels, n_times = X.shape
        self.n_classes = np.unique(y)
        covs = []

        for label in self.n_classes:
            epochs_for_label = X[np.where(y == label)]
            epochs_for_label = epochs_for_label.transpose([1, 0, 2])
            epochs_for_label = epochs_for_label.reshape(n_channels, -1)
            cov_matrix = np.cov(epochs_for_label)
            covs.append(cov_matrix)

        return np.asarray(covs)

    @staticmethod
    def _calculate_eig(covs: np.ndarray, regularization_epsilon: float = 1e-6):
        """
        Calculate the eigenvalues and eigenvectors for each class
        to derive Common Spatial Patterns (CSP) filters.

        In the context of EEG data analysis, CSP filters are used
        to maximize the variance between two or more classes
        of brain states or activities, such as
        'rest', 'left_hand_open', and 'right_hand_open'.
        Each class is represented by its covariance matrix,
        which captures the statistical relationship between
        EEG channels.

        Eigenvalues quantify the variance along the directions
        represented by eigenvectors. In a typical scenario with
        64 EEG channels and 4 labels
        (e.g., left hand open, right hand open), there are
        6 unique combinations of covariance matrices (4 choose 2 = 6).
        For each combination, this function computes 6 eigenvalues
        and their corresponding eigenvectors.

        The use of generalized eigenvalue calculation is unnecessary
        here, as we are dealing with covariance matrices representing
        distinct EEG conditions without a need for a second matrix
        (B) in the form Ax = λBx. Instead, each covariance matrix is
        regularized with a small epsilon value to ensure numerical
        stability and avoid singularities.

        Args:
            covs (np.ndarray): Covariance matrices for each class.
                Shape (n_classes, n_channels, n_channels).
            regularization_epsilon (float, optional): Small value
                for regularization to avoid singular matrices.
                Default is 1e-6.

        Returns:
            tuple: Tuple containing:
                - eigenvalues_list (list of np.ndarray):
                    List of eigenvalues for each class, sorted
                    in descending order of importance.
                - eigenvectors_list (list of np.ndarray):
                    List of eigenvectors for each class corresponding
                    to the sorted eigenvalues.
        """
        eigenvalues_list = []
        eigenvectors_list = []

        n_classes, n_channels, _ = covs.shape

        for idx, iidx in combinations(range(n_classes), 2):
            cov_regularized = (covs[idx]
                               + regularization_epsilon
                               * np.eye(n_channels))

            values, vectors = np.linalg.eig(cov_regularized)

            sorted_indices = np.argsort(np.abs(values))[::-1]

            eigenvalues_list.append(values[sorted_indices])
            eigenvectors_list.append(vectors[:, sorted_indices])

        return eigenvalues_list, eigenvectors_list

    def pick_filters(self, eigenvectors: list[np.ndarray]):
        """
        Pick the CSP filters from the eigenvectors.
        In simple terms, the CSP filters are the eigenvectors
        that maximize the variance between two classes.
        We select the first n_components eigenvectors from
        each class to extract the CSP filters.

        If we have 64 channels and 4 labels, we will have 6 eigenvalues
        and eigenvectors for each class. This is because we have 6
        unique combinations of covariance matrices for 4 labels
        (4 choose 2 = 6).

        We will end up with 24 eigenvectors (6 * 4) for 4 labels.
        Shape (n_channels, n_channels) for each class.

        We then concatenate the eigenvectors for each class to get
        the final CSP filters. Shape (n_channels, n_components).

        Finally, we transpose the filter matrix to get the final
        shape (n_components, n_channels).

       Args:
            eigenvectors (list of np.ndarray): A list of
                eigenvectors for each class.

        Returns:
            None
        """
        self.filters = np.concatenate(
            [EigVects[:, :self.n_components] for EigVects in eigenvectors],
            axis=1
        ).T

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the CSP algorithm on the input data.
        The fit step involves calculating the covariance matrices
        for each class and then deriving the eigenvalues and
        eigenvectors from the covariance matrices.

        Finally, we select the CSP filters from the eigenvectors.

        Args:
            X (np.ndarray): The EEG data.
                Shape (n_epochs, n_channels, n_times).
            y (np.ndarray): The labels for each epoch.
                Shape (n_epochs, ).

        Returns:
            None
        """
        covs = self._calculate_cov(X=X, y=y)
        eigenvalues, eigenvectors = self._calculate_eig(covs=covs)
        self.pick_filters(eigenvectors)

    def transform_epochs(self, X: np.ndarray):
        X = np.asarray([np.dot(self.filters, epoch) for epoch in X])
        X = (X ** 2).mean(axis=2)

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, X: np.ndarray):
        self.transform_epochs(X)
        # Transform the input data using the selected CSP filters
        X = np.asarray([np.dot(self.filters, epoch) for epoch in X])

        # Square and average along the time axis
        X = (X ** 2).mean(axis=2)

        # Standardize features
        X -= self.mean
        X /= self.std

        return X

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, **fit_params):
        """
        Fit CSP on input data and transform it.

        Parameters:
        - X: ndarray, shape (n_epochs, n_channels, n_times)
            The EEG data.
        - y: array, shape (n_epochs, )
            The labels for each epoch.

        Returns:
        - X_transformed: ndarray, shape (n_epochs, n_components)
            Transformed EEG data using CSP filters.
        """
        self.fit(X=X, y=y)
        return self.transform(X=X)
