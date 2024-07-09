import mne
import os
import matplotlib.pyplot as plt

from matplotlib import use
from mne.io.edf.edf import RawEDF
from mne import Epochs

from srcs.utils.utils import get_directory


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

    def plot_filtered_data(self, raw):
        pass

    def plot_ica(self, raw: RawEDF, ica, eog_scores):
        self.report.add_ica(
            ica=ica,
            title='Ica Filtering',
            picks=ica.exclude,
            inst=raw,
            eog_scores=eog_scores,
            n_jobs=1
        )

    def plot_re_referenced_data(self, raw):
        pass

    def plot_downsampled_data(self, raw):
        pass

    def plot_epochs(self, epochs: Epochs, event_id: dict):
        self.report.add_epochs(epochs=epochs, title='Epochs')
        for event in event_id.keys():
            evoked = epochs[event].average()
            self.report.add_evokeds(
                evokeds=evoked,
                titles=f'Evoked {event}',
                n_time_points=5
            )
