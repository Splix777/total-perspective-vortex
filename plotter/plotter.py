import mne

import matplotlib.pyplot as plt

from functools import wraps
from matplotlib import use


plt.close('all')
use('TkAgg')


class Plotter:
    def __init__(self, plot_enabled=True):
        self.plot_enabled = plot_enabled

    def enable_plot(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.plot_enabled:
                func(*args, **kwargs)
        return wrapper

    @staticmethod
    def plot_raw_data(raw):
        # fig = mne.viz.plot_raw(
        #     raw=raw,
        #     duration=10,
        #     n_channels=64,
        #     scalings=dict(eeg=20e-5),
        #     title='Raw EEG Data',
        #     show=False
        # )
        # plt.show()
        pass

    @staticmethod
    def plot_standard_montage(raw):
        # fig_3d = mne.viz.plot_sensors(
        #     info=raw.info,
        #     kind='3d',
        #     ch_type='eeg',
        #     show_names=True,
        #     title='3D Standard Montage 10-10',
        #     ch_groups='position',
        #     to_sphere=True,
        #     sphere='auto',
        #     cmap='viridis',
        #     show=False
        # )
        # fig_2d = mne.viz.plot_sensors(
        #     info=raw.info,
        #     kind='topomap',
        #     ch_type='eeg',
        #     show_names=True,
        #     title='Standard Montage 10-10',
        #     show=False
        # )
        # plt.show()
        pass

    def plot_annotations(self, raw):
        pass

    def plot_filtered_data(self, raw):
        pass

    def plot_ica(self, raw):
        pass

    def plot_re_referenced_data(self, raw):
        pass

    def plot_downsampled_data(self, raw):
        pass