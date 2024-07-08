import os
import sys
import pickle
import argparse
import contextlib

import numpy as np

from itertools import product
from multiprocessing import Pool

# For testings with sklearn's CSP and SPoC
# from mne.decoding import CSP, SPoC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

from csp import CustomCSP
from preprocess_data import preprocess_data
from utils.utils import get_program_config, get_experiment_name, verify_inputs


def save_model(directory: str, file: str, model: Pipeline, score: float):
    """
    Save the model and the score to a file to
    the specified directory.

    Args:
        directory (str): The directory to save the model.
        file (str): The name of the file to save the model.
        model (Pipeline): The model to save.
        score (float): The score of the model.

    Returns:
        None
    """
    os.makedirs(name=directory, exist_ok=True)

    with open(f'{directory}/{file}.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open(f'{directory}/{file}.txt', 'w') as f:
        final_score = f'Final score for model: %{score:.2f}%'
        f.write(f'{final_score}')


def load_model(directory: str, file: str) -> Pipeline | None:
    """
    Load a model from a file in the specified directory.

    Args:
        directory (str): The directory to load the model from.
        file (str): The name of the file to load the model from.

    Returns:
        Pipeline | None: The model if it exists, otherwise None.
    """
    try:
        with open(f'{directory}/{file}.pkl', 'rb') as f:
            return pickle.load(f)

    except FileNotFoundError:
        return None
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
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    log_reg = LogisticRegression(penalty='l2', solver='liblinear')
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC(kernel='linear', C=1.0)
    gbc = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=3)
    mlp = MLPClassifier(
        hidden_layer_sizes=(32,),
        max_iter=100_000,
        early_stopping=True,
        learning_rate_init=0.001,
        learning_rate='adaptive',
        n_iter_no_change=200
    )
    estimators = [lda, log_reg, rfc, svm, gbc, mlp]

    # For testing with sklearn's CSP and SPoC
    # csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    # spoc = SPoC(n_components=15, log=True, reg='oas', rank='full')
    custom_csp = CustomCSP(n_components=16)
    # preprocessing_methods = [custom_csp, csp, spoc]
    preprocessing_methods = [custom_csp]

    pipelines = []
    for decoder, classifier in product(preprocessing_methods, estimators):
        pipeline = make_pipeline(decoder, classifier)
        decoder_name = decoder.__class__.__name__
        classifier_name = classifier.__class__.__name__
        pipeline_name = f"{decoder_name}_{classifier_name}"
        pipelines.append((pipeline_name, pipeline))

    return pipelines


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


def train_subject(subject: int, runs: int | list[int], save: bool = True):
    """
    Train the model for a given subject and run(s).

    Args:
        subject (int): The subject number.
        runs (int) | (list[int]): The run number(s).
        save (bool): Whether to save the data

    Returns:
        None
    """
    experiment_name = get_experiment_name(runs)

    best_mean = -np.inf
    best_pipeline = None
    best_score = None

    epochs, labels = preprocess_data(
        runs=[runs] if isinstance(runs, int)
        else runs, subject=subject, ica=True
    )

    pipelines = create_pipelines()

    for pipeline_name, pipeline in pipelines:
        score = train_model(epochs, labels, pipeline)
        mean_score = np.mean(score)
        if mean_score > best_mean:
            best_mean = mean_score
            best_pipeline = pipeline
            best_score = score

    if save:
        save_model(
            directory='models',
            file=f'subject_{subject}_{experiment_name}',
            model=best_pipeline,
            score=best_mean)

    return best_mean, best_score


def predict_subject(subject: int, run: int):
    """
    Predict the labels for a given subject and run.

    Args:
        subject (int): The subject number.
        run (int): The run number.

    Returns:
        None
    """
    experiment_name = get_experiment_name(run)

    model = load_model(
        directory='models',
        file=f'subject_{subject}_{experiment_name}')

    if model is None:
        print(f'Model not found: subject {subject} and run {run}')
        response = input('Would you like to train the model? (y/n): ')
        if response.lower() != 'y':
            return
        train_subject(subject, run)
        model = load_model(
            directory='models',
            file=f'subject_{subject}_{experiment_name}')

    epochs, labels = preprocess_data(subject=subject, runs=[run], ica=True)

    X_train, X_test, y_train, y_test = train_test_split(
        epochs,
        labels,
        test_size=0.2,
        random_state=42
    )

    predictions = model.predict(X_test)
    accuracy_manual = accuracy_score(y_test, predictions)

    print(f"epoch nb: [prediction] [truth] equal?\n{'-' * 36}")
    for i, (pred, truth) in enumerate(zip(predictions, y_test)):
        equal = pred == truth
        print(f"epoch {i:02}: [{pred}]{' ' * 10}[{truth}]{' ' * 5}{equal}")

    print(f"{'-' * 36}\nAccuracy: {accuracy_manual * 100:.2f}%")


def train_or_predict_single_subject(subject: int, run: int, mode: str):
    """
    Train or predict the model for a given subject and run.

    Args:
        subject (int): The subject number.
        run (int): The run number.
        mode (str): The mode of the program.

    Returns:
        None

    Raises:
        ValueError: If the mode is not 'train' or 'predict'.
    """
    if mode == 'predict':
        predict_subject(subject, run)
    elif mode == 'train':
        mean, scores = train_subject(subject=subject, runs=run)
        for i, score in enumerate(scores):
            print(f"Fold {i + 1} Accuracy: {score * 100:.2f}%")
        print(f"{'-' * 25}\nCross Val Score: {mean * 100:.2f}%")


def train_subject_parallel(subject: int, runs: int, experiment_name: str):
    """
    Train a subject and return mean accuracy.

    Args:
        subject (int): The subject number.
        runs (int): The run number.
        experiment_name (str): The name of the experiment.

    Returns:
        float: The mean accuracy of the model.
    """
    mean, score = train_subject(subject=subject, runs=runs, save=False)

    max_experiment_name_length = 30
    experiment_name = experiment_name.replace('_', ' ').title()
    print(f"Experiment: {experiment_name:{max_experiment_name_length}}"
          f"| Subject: {subject:03} "
          f"Accuracy = {mean * 100:.2f}%")
    return mean


def train_all_subjects():
    """
    Train the model for all subjects and runs.

    Returns:
        None
    """
    # For testing, we won't use all 109 subjects
    subjects = list(range(1, 3))
    experiments = get_program_config()['experiments']

    average_scores = {}
    for experiment in experiments:
        runs = experiment['runs']
        experiment_name = experiment['name']
        scores = [train_subject_parallel(subject, runs, experiment_name)
                  for subject in subjects]
        average_scores[experiment_name] = np.mean(scores)

    print(f"{'-' * 50}")
    for experiment, mean in average_scores.items():
        max_experiment_name_length = 30
        experiment_name = experiment.replace('_', ' ').title()
        print(f"Experiment: {experiment_name:{max_experiment_name_length}} "
              f"| Average Accuracy = {mean * 100:.2f}%")

    mean_all_experiments = np.mean(list(average_scores.values()))
    print(f"{'-' * 50}")
    print(f"Mean Accuracy all experiments: {mean_all_experiments * 100:.2f}%")


def bci():
    """
    Main function to run the program.

    Trains the model for all subjects if no arguments are provided.

    If arguments are provided, the program will train or predict
    the model for a given subject and run number.

    Returns:
        None
    """
    if len(sys.argv) == 1:
        train_all_subjects()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--subject', type=int, required=True)
        parser.add_argument('--run', type=int, required=True)
        parser.add_argument('--mode', type=str, required=True)
        args = parser.parse_args()

        subject, run, mode = verify_inputs(
            subject=args.subject,
            run=args.run,
            mode=args.mode
        )

        train_or_predict_single_subject(subject=subject, run=run, mode=mode)


def main():
    """🧠"""
    try:
        bci()

    except KeyboardInterrupt:
        print('Program terminated by user')
        sys.exit(1)
    except Exception as e:
        import traceback
        error_function = traceback.extract_tb(e.__traceback__)[-1].name
        print(f'Error in function `{error_function}`: {e}')


if __name__ == '__main__':
    main()
