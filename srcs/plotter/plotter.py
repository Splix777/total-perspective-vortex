import mne
import os
import matplotlib.pyplot as plt
import numpy as np

from mne.io.edf.edf import RawEDF
from mne import Epochs

from srcs.utils.utils import get_directory
from srcs.reduction_algorithms.csp import CustomCSP


class Plotter:
    def __init__(self, save_directory: str = None):
        if not save_directory:
            save_directory = get_directory('plots')

        os.makedirs(save_directory, exist_ok=True)
        self.save_directory = save_directory
        self.report = mne.Report(title='Total Perspective Vortex')

    def plot_raw_data(self, raw: RawEDF):
        self.report.add_raw(
            raw=raw,
            title='Raw Data',
            scalings=dict(eeg=20e-5),
            replace=True,
            psd=True,
        )

    def plot_standard_montage(self, raw: RawEDF):
        fig_3d = mne.viz.plot_sensors(
            info=raw.info,
            kind='3d',
            ch_type='eeg',
            show_names=True,
            title='3D Standard Montage 10-10',
            ch_groups='position',
            to_sphere=True,
            sphere='auto',
            cmap='viridis',
            show=False
        )
        self.report.add_figure(fig=fig_3d, title='Montage 3D')
        fig_2d = mne.viz.plot_sensors(
            info=raw.info,
            kind='topomap',
            ch_type='eeg',
            show_names=True,
            title='Standard Montage 10-10',
            show=False
        )
        self.report.add_figure(fig=fig_2d, title='Montage 2D')

    def plot_annotations(self, raw):
        events, event_ids = mne.events_from_annotations(raw=raw)
        sfreq = raw.info['sfreq']
        self.report.add_events(
            events=events,
            event_id=event_ids,
            title='Run Events',
            sfreq=sfreq
        )

    def plot_ica(self, epochs: Epochs, ica, eog_scores):
        self.report.add_ica(
            ica=ica,
            title='Ica Filtering',
            picks=ica.exclude,
            inst=epochs,
            eog_scores=eog_scores,
            n_jobs=-1
        )

    def plot_epochs(self, epochs: Epochs, event_id: dict):
        self.report.add_epochs(epochs=epochs, title='Epochs')
        for event in event_id:
            evoked = epochs[event].average()
            self.report.add_evokeds(
                evokeds=evoked,
                titles=f'Evoked {event}',
                n_time_points=10
            )

    def plot_csp_results(self, X: np.ndarray, y: np.ndarray):
        """
        Fit CSP, transform the data, and plot the results
        along with the direction of the eigenvectors.

        Args:
            X (np.ndarray): The EEG data.
                Shape (n_epochs, n_channels, n_times).
            y (np.ndarray): The labels for each epoch.
                Shape (n_epochs, ).

        Returns
            None
        """
        csp = CustomCSP(n_components=2)
        X_transformed = csp.fit_transform(X, y)
        filters = csp.get_filters()

        fig, ax = plt.subplots(figsize=(10, 7))

        classes = np.unique(y)
        for cls in classes:
            ax.scatter(
                X_transformed[y == cls, 0],
                X_transformed[y == cls, 1],
                label=f'Class {cls}'
            )

        origin = np.zeros((filters.shape[0], 2))
        for vec in filters.T[:2]:
            ax.quiver(
                origin[:, 0], origin[:, 1],
                vec[0], vec[1],
                angles='xy', scale_units='xy', scale=0.1,
                color='r', alpha=0.5
            )

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()
        ax.grid(True)

        self.report.add_figure(
            fig=fig,
            title='CSP Transformed'
        )

    def plot_covariance_matrices(self, epochs: Epochs):
        """
        Plots the covariance matrices of the input epochs.

        Args:
            epochs (Epochs): The input epochs for which covariance
                matrices will be plotted.

        Returns:
            None
        """
        data_cov = mne.compute_covariance(
            epochs=epochs,
            method='shrunk',
            tmin=0,
            tmax=4.0
        )
        self.report.add_covariance(
            cov=data_cov,
            info=epochs.info,
            title='Data Covariance'
        )
