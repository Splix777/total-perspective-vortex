import mne

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from mne import concatenate_raws, Epochs
from mne.datasets import eegbci
from mne.io import read_raw_edf
from mne.io.edf.edf import RawEDF
from mne import events_from_annotations
from mne.channels import make_standard_montage

from utils import define_experiment, get_program_config


def event_map(run: int, config: dict) -> dict:
    experiment = define_experiment(run=run, config=config)
    if experiment == 'left_right_fist':
        return dict(T0=1, T1=2, T2=3)
    elif experiment == 'imagine_left_right_fist':
        return dict(T0=1, T1=2, T2=3)
    elif experiment == 'fists_feet':
        return dict(T0=1, T1=2, T2=3)
    elif experiment == 'imagine_fists_feet':
        return dict(T0=1, T1=2, T2=3)
    else:
        raise ValueError(f'Invalid experiment {experiment}')


def event_description(run: int, config: dict) -> dict:
    experiment = define_experiment(run=run, config=config)
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


def ica_filter(raw: RawEDF):
    ica = mne.preprocessing.ICA(
        n_components=20,
        max_iter=100_000,
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
    raw.notch_filter(60, method="iir", verbose=False)
    raw.filter(
        l_freq=8,
        h_freq=30,
        fir_design='firwin',
        skip_by_annotation='edge',
        verbose=False
    )
    return raw


def re_reference_raw(raw: RawEDF):
    raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)
    raw.apply_proj(verbose=False)
    return raw


def downsample_raw(raw, sfreq):
    raw.resample(sfreq, npad="auto", verbose=False)
    return raw


def standardize_channels(raw: RawEDF):
    eegbci.standardize(raw)
    raw.set_montage(
        montage=make_standard_montage('standard_1020'),
        verbose=False
    )
    return raw


def annotate_raw(raw: RawEDF, run: int):
    config = get_program_config()
    events, _ = events_from_annotations(
        raw=raw,
        event_id=event_map(run=run, config=config),
        verbose=False
    )
    annotations = mne.annotations_from_events(
        events=events,
        event_desc=event_description(run=run, config=config),
        sfreq=raw.info['sfreq'],
        orig_time=raw.info['meas_date'],
        verbose=False
    )

    raw.set_annotations(annotations=annotations, verbose=False)
    return raw


def preprocess_subject(subjects, runs):
    raws_list = []
    for subject in subjects:
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
            # ica_filter(raw=raws)
            re_reference_raw(raw=raws)
            downsample_raw(raw=raws, sfreq=160)
            raws_list.append(raws)

    return concatenate_raws(raws_list)


def create_epochs(raw: RawEDF):
    events, event_ids = events_from_annotations(raw=raw, verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude='bads')
    epochs = Epochs(
        raw=raw,
        events=events,
        event_id=event_ids,
        tmin=-0.2,
        tmax=4.0,
        picks=picks,
        baseline=(None, 0),
        preload=True,
        verbose=False
    )
    # epochs.equalize_event_counts(event_ids=event_ids, method='mintime')
    return epochs


def extract_features(epochs: Epochs):
    features = epochs.get_data(copy=False)
    labels = epochs.events[:, -1]
    return features, labels


def preprocess_data(subjects: list[int], runs: list[int]):
    raw = preprocess_subject(subjects=subjects, runs=runs)
    return extract_features(epochs=create_epochs(raw=raw))
