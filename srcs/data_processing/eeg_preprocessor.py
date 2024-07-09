import numpy as np

from dataclasses import dataclass, field

from mne.preprocessing import ICA, create_eog_epochs
from mne.io import read_raw_edf
from mne.datasets import eegbci
from mne.io.edf.edf import RawEDF
from mne import events_from_annotations
from mne.channels import make_standard_montage
from mne import concatenate_raws, Epochs, annotations_from_events, pick_types
from mne.epochs import make_metadata

from srcs.utils.utils import get_experiment
from srcs.utils.decorators import time_limit
from srcs.plotter.plotter import Plotter
from srcs.reduction_algorithms.csp import CustomCSP


@dataclass
class EEGProcessor:
    subject: int
    runs: list[int]
    ica: bool
    plot: bool
    plotter: Plotter = field(init=False)
    raw: RawEDF = field(init=False, default=None)
    epochs: Epochs = field(init=False, default=None)
    features: np.ndarray = field(init=False, default=None)
    labels: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        if self.plot:
            self.plotter = Plotter()
        self.preprocess_data()

    def _event_description(self) -> dict:
        """
        Returns a dictionary mapping integers to descriptions
        based on the experiment type.

        Args:
            self (EEGProcessor): The EEGProcessor instance.

        Returns:
            dict: A dictionary mapping integers to descriptions
                based on the experiment type.

        Raises:
            ValueError: If the run is invalid.
        """
        experiment = get_experiment(run=self.runs)
        return {int(k): v for k, v in experiment['mapping'].items()}

    @time_limit(limit=10)
    def _ica_filter(self):
        """
        Independent Component Analysis (ICA) is used to remove
        eye movement artifacts from the raw data. The function
        fits the ICA model to the raw data and identifies
        components that are likely to be eye movements. These
        components are then removed from the raw data.

        Susceptible channels are identified based on the
        following criteria:
            - Fp1, Fp2, Fz, F4, F8, Fpz

        These channels are located near the eyes and are
        likely to be affected by eye movements. Channels
        not close to the eyes are also susceptible to
        eye movement artifacts, but the above channels
        are the most likely to be affected.

        Modifies the raw data in-place by applying the ICA
        model to the raw data and removing the identified
        eye movement components.

        Args:
            self (EEGProcessor): The EEGProcessor instance.

        Returns:
            None
        """
        ica = ICA(
            n_components=20,
            max_iter=10_000,
            method='fastica',
            random_state=42,
            verbose=False
        )
        ica.fit(inst=self.raw, verbose=False)

        eog_susceptible_channels = ['Fp1', 'Fp2', 'Fz', 'F4', 'F8', 'Fpz']
        eog_scores = None
        for channel in eog_susceptible_channels:
            eog_indices, eog_scores = ica.find_bads_eog(
                inst=self.raw,
                ch_name=channel,
                threshold=1.5,
                l_freq=8,
                h_freq=30,
                verbose=False
            )
            ica.exclude.extend(eog_indices)

        ica.apply(inst=self.raw, exclude=ica.exclude, verbose=False)

        if self.plot:
            self.plotter.plot_ica(
                raw=self.raw,
                ica=ica,
                eog_scores=eog_scores,
            )

    def _filter_raw(self):
        """
        Applies notch filter of 60 Hz. Notch filters are used
        to eliminate power line noise from the raw data. Common
        power line frequencies include 50 Hz and 60 Hz. Depends
        on the country of origin. Method 'iir' (Infinite Impulse
        Response) is used to apply the notch filter. Its digital
        filter that can achieve a sharper cutoff than FIR filters.
        Our recordings were done by PhysioNet Located in Cambridge,
        Massachusetts, USA. So, we use 60 Hz as the power line
        frequency.

        Applies bandpass filter between 8 Hz and 30 Hz. Bandpass
        filters are used to remove unwanted frequencies from the
        raw data. The lower cutoff frequency is 8 Hz, and the upper
        cutoff frequency is 30 Hz. The FIR filter design is used
        to apply the bandpass filter. The 'firwin' design is used
        to design the filter coefficients. The 'skip_by_annotation'
        parameter is set to 'edge' to avoid filtering the data
        near the edges of the raw data.

        Args:
            self (EEGProcessor): The EEGProcessor instance.

        Returns:
            None
        """
        self.raw.notch_filter(60, method="iir", verbose=False)
        self.raw.filter(
            l_freq=8,
            h_freq=30,
            fir_design='firwin',
            skip_by_annotation='edge',
            verbose=False
        )

    def _re_reference_raw(self):
        """
        Re-references the raw data to the average reference.
        Its process of recalculating the EEG signal by
        subtracting the average of all electrodes from each
        individual electrode. This process is used to remove
        common noise sources from the raw data.

        Each channel's signal will be recalculated as the
        difference from mean signal of all channels. This helps
        balance out common noise sources across all channels.

        Projection parameter is set to True to apply the
        projection vectors to the raw data. This is used to
        remove artifacts from the raw data.

        Args:
            self (EEGProcessor): The EEGProcessor instance.

        Returns:
            None
        """
        self.raw.set_eeg_reference(
            ref_channels='average',
            projection=True,
            verbose=False
        )
        self.raw.apply_proj(verbose=False)

    def _downsample_raw(self, sfreq):
        """
        Down samples the raw data to the specified sampling
        frequency. This means recording fewer data points
        per second, which can help reduce the computational
        cost of processing the data. The 'auto' parameter
        is used to automatically determine the number of
        points to pad the data with before resampling.

        npad: helps mitigate the edge effects that can occur
        when resampling the data. These edge effects can
        cause distortions in the data, so padding the data
        can help reduce these distortions.

        Args:
            self (EEGProcessor): The EEGProcessor instance.
            sfreq (int): The sampling frequency to downsample to.

        Returns:
            None
        """
        self.raw.resample(sfreq, npad="auto", verbose=False)

    def _standardize_channels(self):
        """
        Standardizes the raw data by applying the standard
        10-20 electrode system. This system is used to
        standardize the placement of electrodes on the
        scalp. The 10-20 system is used to ensure that
        electrodes are placed at consistent locations
        across different subjects. Our data is recorded
        using the 10-10 system. Since there is no '10-10'
        montage available in MNE-Python, we use the
        closest available montage, which is the standard
        10-20 system.

        Args:
            self (EEGProcessor): The EEGProcessor instance.

        Returns:
            None
        """
        eegbci.standardize(self.raw)
        self.raw.set_montage(
            montage=make_standard_montage('standard_1020'),
            verbose=False
        )
        if self.plot:
            self.plotter.plot_raw_data(self.raw)
            self.plotter.plot_standard_montage(raw=self.raw)

    def _annotate_raw(self):
        """
        Annotates the raw data based on the event descriptions
        provided by the event_description function. The event
        descriptions are used to label the raw data with
        descriptions of the events that occurred during the
        recording. The annotations are used to mark the start
        and end times of each event in the raw data.

        This allows us to differentiate between different
        experimental conditions and identify when each
        condition occurred during the recording.

        Args:
            self (EEGProcessor): The EEGProcessor instance.

        Returns:
            None
        """
        events, _ = events_from_annotations(
            raw=self.raw,
            event_id=dict(T0=1, T1=2, T2=3),
            verbose=False
        )
        annotations = annotations_from_events(
            events=events,
            event_desc=self._event_description(),
            sfreq=self.raw.info['sfreq'],
            orig_time=self.raw.info['meas_date'],
            verbose=False
        )
        self.raw.set_annotations(annotations=annotations, verbose=False)

        if self.plot:
            self.plotter.plot_annotations(raw=self.raw)

    def _process(self, data: list):
        """
        Pre-processing sequence for the raw data. The function loads
        the raw data, standardizes the channels, annotates the data,
        filters the data, applies Independent Component Analysis (ICA),
        re-references the data, and down-samples the data.

        Args:
            self (EEGProcessor): The EEGProcessor instance.
            data (RawEDF): The raw data to process.

        Returns:
            None
        """

        self.raw = concatenate_raws([
            read_raw_edf(f, preload=True, verbose=False) for f in data
        ])
        self._standardize_channels()
        self._annotate_raw()
        self._filter_raw()
        if self.ica:
            self._ica_filter()
        self._re_reference_raw()
        self._downsample_raw(sfreq=160)

    def _preprocess_subject(self):
        """
        Pre-processing sequence for the raw data. The function loads
        the raw data, standardizes the channels, annotates the data,
        filters the data, applies Independent Component Analysis (ICA),
        re-references the data, and down samples the data.

        Apply the sequence to each subject and run combination
        provided in the arguments. The raw data for each subject
        and run combination is concatenated into a single raw object.

        Args:
            self (EEGProcessor): The EEGProcessor instance.

        Returns:
            RawEDF: The pre-processed raw data for all
                run combinations provided in the arguments.
        """
        data = eegbci.load_data(
            subject=self.subject,
            runs=self.runs,
            path='data/',
            verbose=0
        )
        self._process(data=data)

    def _create_epochs(self):
        """
        Creates epochs from the raw data. Epochs are time-locked
        segments of the raw data that are extracted based on the
        event markers in the data. The epochs are extracted from
        the raw data based on the event markers provided in the
        annotations.

        The epochs are extracted from the raw data with a time
        window of -0.2 to 4.0 seconds relative to the event
        markers. The baseline is set to None to avoid applying
        baseline correction to the data. The data is preloaded
        to speed up processing.

        The epochs are created for EEG channels only, excluding
        EOG and Stim channels. Bad channels are also excluded
        from the epochs.

        Args:
            self (EEGProcessor): The EEGProcessor instance.

        Returns:
            None
        """
        events, event_id = events_from_annotations(raw=self.raw, verbose=False)
        picks = pick_types(self.raw.info, eeg=True, exclude='bads')
        metadata, _, _ = make_metadata(
            events=events,
            event_id=event_id,
            tmin=-0.2,
            tmax=4.0,
            sfreq=self.raw.info['sfreq']
        )
        self.epochs = Epochs(
            raw=self.raw,
            events=events,
            event_id=event_id,
            tmin=-0.2,
            tmax=4.0,
            picks=picks,
            baseline=(None, 0),
            preload=True,
            metadata=metadata,
            verbose=False,
        )
        if self.plot:
            self.plotter.plot_epochs(epochs=self.epochs, event_id=event_id)

    def _extract_features(self):
        """
        Extracts features and labels from the epochs. The features
        are the EEG data from the epochs, and the labels are the
        event markers associated with each epoch. The features and
        labels are returned as numpy arrays. The copy parameter is
        set to False, to avoid copying the data, which can save
        memory and processing time.

        Args:
            self (EEGProcessor): The EEGProcessor instance.
            epochs (Epochs): The epochs to extract features and
                labels from.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the
                features and labels extracted from the epochs.
        """
        self.features = self.epochs.get_data(copy=False)
        self.labels = self.epochs.events[:, -1]
        if self.plot:
            self.plotter.plot_csp_results(self.features, self.labels)

    def preprocess_data(self):
        """
        Pre-processes the data for the specified subjects and runs.
        The pre-processing steps include loading the raw data,
        standardizing the channels, annotating the data, filtering
        the data, applying Independent Component Analysis (ICA),
        re-referencing the data, and down-sampling the data.

        The pre-processed data is then used to create epochs, which
        are time-locked segments of the data extracted based on the
        event markers in the data. The epochs are created for EEG
        channels only, excluding EOG and Stim channels. Bad channels
        are also excluded from the epochs.

        After creating the epochs, features and labels are extracted
        from the epochs. The features are the EEG data from the epochs,
        and the labels are the event markers associated with each epoch.

        Args:
            self (EEGProcessor): The EEGProcessor instance.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the
                features and labels extracted from the epochs.
        """
        try:
            self._preprocess_subject()
            self._create_epochs()
            self._extract_features()
            if self.plot:
                self.plotter.report.save(
                    fname='images/plots/plots.html',
                    overwrite=True)

        except Exception as e:
            raise e
