import os
import sys
import pickle
import argparse
import contextlib
import multiprocessing

import numpy as np

from itertools import product

# For testings with sklearn's various sklearn classifiers
# from mne.decoding import CSP, SPoC
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import (
#     RandomForestClassifier,
#     GradientBoostingClassifier
# )
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

from srcs.reduction_algorithms.csp import CustomCSP
from srcs.data_processing.eeg_preprocessor import EEGProcessor
from srcs.utils.utils import (
    get_program_config,
    get_experiment,
    verify_inputs,
    print_error_tree,
    download_subjects
)
from srcs.utils.decorators import time_it


def save_model(directory: str, file: str, model: Pipeline):
    """
    Save the model and the score to a file to
    the specified directory.

    Args:
        directory (str): The directory to save the model.
        file (str): The name of the file to save the model.
        model (Pipeline): The model to save.

    Returns:
        None

    Raises:
        ValueError: If there is an error saving the model.
    """
    try:
        os.makedirs(name=directory, exist_ok=True)
        with open(f'{directory}/{file}.pkl', 'wb') as f:
            pickle.dump(model, f)

    except Exception as e:
        raise ValueError(f'Error saving model: {e}') from e


def load_model(directory: str, file: str) -> Pipeline:
    """
    Load a model from a file in the specified directory.

    Args:
        directory (str): The directory to load the model from.
        file (str): The name of the file to load the model from.

    Returns:
        Pipeline: The model if it exists.

    Raises:
        ValueError: If there is an error loading the model.
    """
    try:
        with open(f'{directory}/{file}.pkl', 'rb') as f:
            return pickle.load(f)

    except FileNotFoundError as e:
        raise ValueError(f'Model {file} not found, train the model') from e
    except Exception as e:
        raise ValueError(f'Error loading model: {e}') from e


def create_pipelines() -> list[tuple[str, Pipeline]]:
    """
    Create a list of pipelines to train the model.
    Different combinations of decoders and classifiers
    are used to create the pipelines. This allows us
    to compare the performance of different models.

    Returns:
        list: A list of pipelines to train the model.
    """
    # For testing with various classifiers
    # log_reg = LogisticRegression(penalty='l2', solver='liblinear')
    # rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    # svm = SVC(kernel='linear', C=1.0)
    # gbc = GradientBoostingClassifier(
    #     n_estimators=100,
    #     learning_rate=0.01,
    #     max_depth=3)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    mlp = MLPClassifier(
        hidden_layer_sizes=(32,),
        max_iter=5_000,
        early_stopping=True,
        learning_rate_init=0.001,
        learning_rate='invscaling',
        n_iter_no_change=10
    )
    # estimators = [lda, log_reg, rfc, svm, gbc, mlp]
    estimators = [lda, mlp]

    # For testing with sklearn's CSP and SPoC
    # csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    # spoc = SPoC(n_components=15, log=True, reg='oas', rank='full')
    custom_csp = CustomCSP(n_components=32)
    # reduction_algorithms = [custom_csp, csp, spoc]
    preprocessing_methods = [custom_csp]

    pipelines = []
    for decoder, classifier in product(preprocessing_methods, estimators):
        pipeline = make_pipeline(decoder, classifier)
        decoder_name = decoder.__class__.__name__
        classifier_name = classifier.__class__.__name__
        pipeline_name = f"{decoder_name}_{classifier_name}"
        pipelines.append((pipeline_name, pipeline))

    return pipelines


def get_training_data(subject: int, runs: int, ica: bool = True,
                      plot: bool = False) -> tuple:
    """
    Get the training data for a given subject and run(s).

    Args:
        subject (int): The subject number.
        runs (int) | (list[int]): The run number(s).
        ica (bool): Whether to use ICA.
        plot (bool): Whether to plot the data.

    Returns:
        EEGProcessor: The processed data.
    """
    preprocessor = EEGProcessor(
        runs=[runs] if isinstance(runs, int) else runs,
        subject=subject,
        ica=ica,
        plot=plot,
    )
    return preprocessor.preprocess_data()


def train_model(epochs: np.ndarray, labels: np.ndarray, pipeline: Pipeline):
    """
    Train the model using cross-validation and return the
    average accuracy score.

    Args:
        epochs (np.ndarray): The epoch data.
        labels (np.ndarray): The labels' data.
        pipeline (Pipeline): The pipeline to train.

    Returns:
        np.ndarray: The accuracy scores for each fold.
    """
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            for train_index, test_index in cv.split(epochs):
                X_train = epochs[train_index]
                y_train = labels[train_index]
                pipeline.fit(X_train, y_train)

            return cross_val_score(
                estimator=pipeline,
                X=epochs,
                y=labels,
                cv=cv,
                scoring='accuracy')


def train_subject(subject: int, runs: int | list[int], save: bool = True,
                  ica: bool = True, plot: bool = False):
    """
    Train the model for a given subject and run(s).

    Args:
        subject (int): The subject number.
        runs (int) | (list[int]): The run number(s).
        save (bool): Whether to save the data
        ica (bool): Whether to use ICA.
        plot (bool): Whether to plot the data.

    Returns:
        None
    """
    best_pipeline = None
    best_score = -np.inf

    features, labels = get_training_data(
        subject=subject,
        runs=runs,
        ica=ica,
        plot=plot
    )

    for pipeline_name, pipeline in create_pipelines():
        score = train_model(
            epochs=features,
            labels=labels,
            pipeline=pipeline)
        if np.mean(score) > np.mean(best_score):
            best_pipeline = pipeline
            best_score = score

    if save:
        model_name = f"subject_{subject}_{get_experiment(runs)['name']}"
        save_model(
            directory='models',
            file=model_name,
            model=best_pipeline
        )

    return np.mean(best_score), best_score


@time_it
def predict_subject(subject: int, run: int):
    """
    Predict the labels for a given subject and run.

    Args:
        subject (int): The subject number.
        run (int): The run number.

    Returns:
        None
    """
    model = load_model(
        directory='models',
        file=f"subject_{subject}_{get_experiment(run)['name']}"
    )

    features, labels = get_training_data(
        subject=subject,
        runs=run,
        ica=False,
        plot=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42
    )

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"epoch nb: [prediction] [truth] equal?\n{'-' * 36}")
    for i, (pred, truth) in enumerate(zip(predictions, y_test)):
        equal = pred == truth
        print(f"epoch {i:02}: [{pred}]{' ' * 10}[{truth}]{' ' * 5}{equal}")

    print(f"{'-' * 36}\nAccuracy: {accuracy * 100:.2f}%")


def train_or_predict_single_subject(subject: int, run: int, mode: str,
                                    plot: bool = False):
    """
    Train or predict the model for a given subject and run.

    Args:
        subject (int): The subject number.
        run (int): The run number.
        mode (str): The mode of the program.
        plot (bool): Whether to plot the data.

    Returns:
        None

    Raises:
        ValueError: If the mode is not 'train' or 'predict'.
    """
    if mode == 'predict':
        predict_subject(subject=subject, run=run)
    elif mode == 'train':
        mean, scores = train_subject(subject=subject, runs=run, plot=plot)
        for i, score in enumerate(scores):
            print(f"Fold {i + 1} Accuracy: {score * 100:.2f}%")
        print(f"{'-' * 25}\nCross Val Score: {mean * 100:.2f}%")


def train_all_subjects():
    """
    Train the model for all subjects and runs.

    Returns:
        None
    """
    # For testing, we won't use all 109 subjects
    subjects = list(range(1, 21))
    experiments = get_program_config()['experiments']

    average_scores = {}
    for experiment in experiments:
        for subject in subjects:
            runs = experiment['runs']
            mean, scores = train_subject(subject, runs, save=False)
            experiment_name = experiment['description'].title()
            print(
                f"Experiment: {experiment_name:52}| Subject: {subject:03} "
                f"Accuracy = {mean * 100:.2f}%"
            )
            average_scores[experiment_name] = mean

    print(f"{'-' * 80}")
    for experiment, mean in average_scores.items():
        experiment_name = experiment.replace('_', ' ').title()
        print(
            f"Experiment: {experiment_name:52}| "
            f"Average Accuracy = {mean * 100:.2f}%"
        )

    mean_all_experiments = np.mean(list(average_scores.values()))
    print(f"{'-' * 80}")
    print(f"Mean Accuracy all experiments: {mean_all_experiments * 100:.2f}%")


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    subject, run, mode = verify_inputs(
        subject=args.subject,
        run=args.run,
        mode=args.mode
    )

    train_or_predict_single_subject(
        subject=subject,
        run=run,
        mode=mode,
        plot=args.plot
    )


def bci():
    """
    Main function to run the program.

    Trains the model for all subjects if no arguments are provided.

    If arguments are provided, the program will train or predict
    the model for a given subject and run number.

    Returns:
        None
    """
    dl_process = None
    try:
        if len(sys.argv) == 1:
            dl_process = multiprocessing.Process(target=download_subjects)
            dl_process.start()
            train_all_subjects()
        else:
            get_input_args()

    except KeyboardInterrupt as e:
        raise KeyboardInterrupt("Program interrupted by user.") from e
    except Exception as e:
        raise e
    finally:
        if dl_process and dl_process.is_alive():
            dl_process.terminate()
            dl_process.join()


def main():
    """🧠"""
    try:
        bci()
    except KeyboardInterrupt as e:
        print("Program interrupted by user.")
    except Exception as e:
        print_error_tree(e)


if __name__ == '__main__':
    main()
