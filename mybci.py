import os
import pickle
import argparse
import contextlib
import numpy as np

from itertools import product
from mne.decoding import CSP, SPoC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC

from csp import CustomCSP
from preprocess_data import preprocess_data
from utils import get_program_config, get_experiment_name, verify_inputs


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


def load_model(directory: str, file: str) -> Pipeline:
    """
    Load a model from a file in the specified directory.

    Args:
        directory (str): The directory to load the model from.
        file (str): The name of the file to load the model from.

    Returns:
        Pipeline: The model loaded from the file.
    """
    with open(f'{directory}/{file}.pkl', 'rb') as f:
        return pickle.load(f)


def train_model(epochs: np.ndarray, labels: np.ndarray, pipeline: Pipeline):
    """
    Train the model using cross-validation and return the
    average accuracy score.

    Args:
        epochs (np.ndarray): The epochs data.
        labels (np.ndarray): The labels data.
        pipeline (Pipeline): The pipeline to train.

    Returns:
        float: The average accuracy score.
    """
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    scores = []

    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            for train_index, test_index in cv.split(epochs):
                X_train, X_test = epochs[train_index], epochs[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                pipeline.fit(X_train, y_train)

            score = cross_val_score(
                pipeline,
                epochs,
                labels,
                cv=cv,
                scoring='accuracy'
            )
            scores.append(score)

    return np.mean(scores)


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
    classifiers = [lda, log_reg, rfc, svm, gbc, mlp]

    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    spoc = SPoC(n_components=15, log=True, reg='oas', rank='full')
    custom_csp = CustomCSP(n_components=8)
    decoders = [custom_csp]

    pipelines = []
    for decoder, classifier in product(decoders, classifiers):
        pipeline = make_pipeline(decoder, classifier)
        decoder_name = decoder.__class__.__name__
        classifier_name = classifier.__class__.__name__
        pipeline_name = f"{decoder_name}_{classifier_name}"
        pipelines.append((pipeline_name, pipeline))

    return pipelines


def train_subject(subject: int, runs: int | list[int]):
    """
    Train the model for a given subject and run(s).

    Args:
        subject (int): The subject number.
        runs (int) | (list[int]): The run number(s).

    Returns:
        float: The average accuracy score.
    """
    experiment_name = get_experiment_name(runs)

    best_score = -np.inf
    best_pipeline = None

    epochs, labels = preprocess_data(
        runs=[runs] if isinstance(runs, int)
        else runs, subject=subject

    )
    pipelines = create_pipelines()
    for pipeline_name, pipeline in pipelines:
        score = train_model(epochs, labels, pipeline)
        if score > best_score:
            best_score = score
            best_pipeline = pipeline

    save_model(
        directory='models',
        file=f'subject_{subject}_{experiment_name}',
        model=best_pipeline,
        score=best_score)

    return best_score


def predict_subject(subject: int, run: int):
    """
    Predict the labels for a given subject and run.

    Args:
        subject (int): The subject number.
        run (int): The run number.

    Returns:
        float: The accuracy score of
            the model on the test set.
    """
    experiment_name = get_experiment_name(run)

    model = load_model(
        directory='models',
        file=f'subject_{subject}_{experiment_name}')

    if model is None:
        raise ValueError(f'Model not found: subject {subject} and run {run}')

    epochs, labels = preprocess_data(subject=subject, runs=[run])

    X_train, X_test, y_train, y_test = train_test_split(
        epochs,
        labels,
        test_size=0.2,
        random_state=42)

    # Evaluate the model on the unseen test set
    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    print("epoch nb: [prediction] [truth] equal?")
    for i, (pred, truth) in enumerate(zip(predictions, y_test)):
        equal = pred == truth
        print(f"epoch {i:02}: [{pred}] [{truth}] {equal}")

    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


def train_all_subjects():
    """
    Train the model for all subjects and runs.

    Returns:
        None
    """
    subjects = list(range(1, 11))
    config = get_program_config()
    experiments = config['experiments']

    average_scores = {}
    for experiment in experiments:
        run = experiment['runs']
        scores = []
        experiment_name = experiment['name']
        for subject in subjects:
            score = train_subject(subject, run)
            print(f"Experiment: {experiment_name}"
                  f" -- Subject: {subject} "
                  f"Accuracy = {score * 100:.2f}%")
            scores.append(score)

        average_scores[experiment_name] = np.mean(scores)

    for experiment, score in average_scores.items():
        print(f"Experiment: {experiment} "
              f"Average Accuracy = {score * 100:.2f}%")

    mean_all_experiments = np.mean(list(average_scores.values()))
    print(f"Mean Accuracy all experiments: {mean_all_experiments * 100:.2f}%")


def bci(subject=None, run=None, mode=None):
    """
    Train or predict the model based on the input arguments.
    If no arguments are provided, train the model for all
    subjects and runs.

    Args:
        subject (int): The subject number.
        run (int): The run number.
        mode (str): The mode of the program.

    Returns:
        None
    """
    if all([subject, run, mode]) is not None:
        if mode == 'train':
            score = train_subject(subject=subject, runs=run)
            print(f"Cross Val Score: {score * 100:.2f}%")
        if mode == 'predict':
            return predict_subject(subject, run)
    else:
        train_all_subjects()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, default=None)
    parser.add_argument('--run', type=int, default=None)
    parser.add_argument('--mode', type=str, default=None)
    args = parser.parse_args()

    subject, run, mode = verify_inputs(args.subject, args.run, args.mode)
    bci(subject, run, mode)


if __name__ == '__main__':
    main()
