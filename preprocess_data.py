import numpy as np

from mne.preprocessing import ICA
from mne.io import read_raw_edf
from mne.datasets import eegbci
from mne.io.edf.edf import RawEDF
from mne import events_from_annotations
from mne.channels import make_standard_montage
from mne import concatenate_raws, Epochs, annotations_from_events, pick_types

from utils.utils import get_experiment_name
from utils.decorators import time_limit


def event_description(run: int) -> dict:
    """
    Returns a dictionary mapping integers to descriptions
    based on the experiment type.

    Args:
        run (int): The run number.

    Returns:
        dict: A dictionary mapping integers to descriptions
            based on the experiment type.

    Raises:
        ValueError: If the run is invalid.
    """
    experiment = get_experiment_name(run=run)

    if experiment == 'left_right_fist':
        return {1: 'bads', 2: 'real_left', 3: 'real_right'}
    elif experiment == 'imagine_left_right_fist':
        return {1: 'bads', 2: 'imaginary_left', 3: 'imaginary_right'}
    elif experiment == 'fists_feet':
        return {1: 'bads', 2: 'real_fists', 3: 'real_feet'}
    elif experiment == 'imagine_fists_feet':
        return {1: 'bads', 2: 'imaginary_fists', 3: 'imaginary_feet'}
    else:
        raise ValueError(f'Invalid run {run}')


@time_limit(limit=10)
def ica_filter(raw: RawEDF):
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

    Args:
        raw (RawEDF): The raw data to filter.

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
    ica.fit(inst=raw, verbose=False)

    eog_susceptible_channels = ['Fp1', 'Fp2', 'Fz', 'F4', 'F8', 'Fpz']
    for channel in eog_susceptible_channels:
        eog_indices, eog_scores = ica.find_bads_eog(
            inst=raw,
            ch_name=channel,
            threshold='auto',
            l_freq=1,
            h_freq=7,
            verbose=False
        )
        ica.exclude.extend(eog_indices)

    ica.apply(inst=raw, exclude=ica.exclude, verbose=False)


def filter_raw(raw: RawEDF):
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
        raw (RawEDF): The raw data to be filtered.

    Returns:
        None
    """
    raw.notch_filter(60, method="iir", verbose=False)
    raw.filter(
        l_freq=8,
        h_freq=30,
        fir_design='firwin',
        skip_by_annotation='edge',
        verbose=False
    )


def re_reference_raw(raw: RawEDF):
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
        raw (RawEDF): The raw data to re-reference.

    Returns:
        None
    """
    raw.set_eeg_reference(
        ref_channels='average',
        projection=True,
        verbose=False
    )
    raw.apply_proj(verbose=False)


def downsample_raw(raw, sfreq):
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
        raw (RawEDF): The raw data to downsample.
        sfreq (int): The sampling frequency to downsample to.

    Returns:
        None
    """
    raw.resample(sfreq, npad="auto", verbose=False)


def standardize_channels(raw: RawEDF):
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
        raw (RawEDF): The raw data to standardize.

    Returns:
        None
    """
    eegbci.standardize(raw)
    raw.set_montage(
        montage=make_standard_montage('standard_1020'),
        verbose=False
    )


def annotate_raw(raw: RawEDF, run: int):
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
        raw (RawEDF): The raw data to annotate.
        run (int): The run number.

    Returns:
        RawEDF: The annotated raw data.
    """
    events, _ = events_from_annotations(
        raw=raw,
        event_id=dict(T0=1, T1=2, T2=3),
        verbose=False
    )
    annotations = annotations_from_events(
        events=events,
        event_desc=event_description(run=run),
        sfreq=raw.info['sfreq'],
        orig_time=raw.info['meas_date'],
        verbose=False
    )

    raw.set_annotations(annotations=annotations, verbose=False)
    return raw


def preprocess_subject(subject: int, runs: list[int], ica: bool) -> RawEDF:
    """
    Pre-processing sequence for the raw data. The function loads
    the raw data, standardizes the channels, annotates the data,
    filters the data, applies Independent Component Analysis (ICA),
    re-references the data, and down samples the data.

    Apply the sequence to each subject and run combination
    provided in the arguments. The raw data for each subject
    and run combination is concatenated into a single raw object.

    Args:
        subject (int): The subject number.
        runs (list[int]): List of run numbers.
        ica (bool): Whether to apply Independent Component Analysis
            (ICA) to the raw data.

    Returns:
        RawEDF: The pre-processed raw data for all
            run combinations provided in the arguments.
    """
    raws_list = []
    for run in runs:
        data = eegbci.load_data(
            subject=subject,
            runs=run,
            path='data/',
            verbose=False
        )
        raws = concatenate_raws([
            read_raw_edf(f, preload=True, verbose=False) for f in data
        ])
        standardize_channels(raw=raws)
        annotate_raw(raw=raws, run=run)
        filter_raw(raw=raws)
        if ica:
            ica_filter(raw=raws)
        re_reference_raw(raw=raws)
        downsample_raw(raw=raws, sfreq=160)
        raws_list.append(raws)

    return concatenate_raws(raws_list)


def create_epochs(raw: RawEDF) -> Epochs:
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
        raw (RawEDF): The raw data to create epochs from.

    Returns:
        Epochs: The epochs extracted from the raw data.
    """
    events, event_ids = events_from_annotations(raw=raw, verbose=False)
    picks = pick_types(raw.info, eeg=True, exclude='bads')
    return Epochs(
        raw=raw,
        events=events,
        event_id=event_ids,
        tmin=-0.2,
        tmax=4.0,
        picks=picks,
        baseline=(None, 0),
        preload=True,
        verbose=False,
    )


def extract_features(epochs: Epochs) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts features and labels from the epochs. The features
    are the EEG data from the epochs, and the labels are the
    event markers associated with each epoch. The features and
    labels are returned as numpy arrays. The copy parameter is
    set to False, to avoid copying the data, which can save
    memory and processing time.

    Args:
        epochs (Epochs): The epochs to extract features and
            labels from.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the
            features and labels extracted from the epochs.
    """
    features = epochs.get_data(copy=False)
    labels = epochs.events[:, -1]
    return features, labels


def preprocess_data(subject: int, runs: list[int], ica: bool) -> tuple:
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
        subject (int): The subject number.
        runs (list[int]): List of run numbers.
        ica (bool): Whether to apply Independent Component Analysis
            (ICA) to the raw data.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the
            features and labels extracted from the epochs.
    """
    try:
        raw = preprocess_subject(subject=subject, runs=runs, ica=ica)
        return extract_features(epochs=create_epochs(raw=raw))

    except Exception as e:
        raise e
