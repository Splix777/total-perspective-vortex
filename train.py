import contextlib
import json
import os
import pickle
import numpy as np

from itertools import product
from mne.decoding import CSP, SPoC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, KFold
from sklearn.svm import SVC
from tqdm import tqdm

from csp import CustomCSP
from preprocess_data import preprocess_data
from utils import get_program_config, define_experiment


def save_model(directory: str, file: str, model: Pipeline, score: float):
    os.makedirs(f'{directory}', exist_ok=True)

    with open(f'{directory}/{file}.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open(f'{directory}/{file}.txt', 'w') as f:
        final_score = f'Final score for model: %{score:.2f}%'
        f.write(f'{final_score}')


def load_model(directory: str, file: str):
    with open(f'{directory}/{file}.pkl', 'rb') as f:
        return pickle.load(f)


def train_model(epochs, labels, pipeline):
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


def create_pipelines():
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


def train_subject(subject: int, runs: list[int], config: dict):
    experiment_name = define_experiment(runs[-1], config)

    average_score = 0
    best_score = -np.inf
    best_pipeline = None

    for run in runs:
        epochs, labels = preprocess_data(subjects=[subject], runs=[run])
        pipelines = create_pipelines()

        for pipeline_name, pipeline in pipelines:
            score = train_model(epochs, labels, pipeline)
            if score > best_score:
                best_score = score
                best_pipeline = pipeline

        average_score += best_score

    average_score /= len(runs)

    save_model(
        directory='models',
        file=f'subject_{subject}_{experiment_name}',
        model=best_pipeline,
        score=best_score)

    return average_score


def predict_subject(subject: int, run: int, config: dict):
    experiment_name = define_experiment(run, config)

    model = load_model(directory='models', file=f'subject_{subject}_{experiment_name}')

    if model is None:
        raise ValueError(f'Model not found for subject {subject} and run {run}')

    epochs, labels = preprocess_data(subjects=[subject], runs=[run])

    X_train, X_test, y_train, y_test = train_test_split(
        epochs,
        labels,
        test_size=0.2,
        random_state=42)

    # Evaluate the model on the test set
    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    print(f"epoch nb: [prediction] [truth] equal?")
    for i, (pred, truth) in enumerate(zip(predictions, y_test)):
        equal = pred == truth
        print(f"epoch {i:02}: [{pred}] [{truth}] {equal}")

    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


def train_all_subjects(config: dict):
    subjects = [id for id in range(1, 11)]
    experiments = config['experiments']

    average_scores = {}
    for experiment in experiments:
        run = experiment['runs']
        scores = []
        experiment_name = experiment['name']
        for subject in subjects:
            score = train_subject(subject, run, config)
            print(f"Experiment: {experiment_name.replace('_', ' ').capitalize()}"
                  f" -- Subject: {subject} "
                  f"Accuracy = {score * 100:.2f}%")
            scores.append(score)

        average_scores[experiment_name] = np.mean(scores)

    for experiment, score in average_scores.items():
        print(f"Experiment: {experiment} "
              f"Average Accuracy = {score * 100:.2f}%")

    mean_all_experiments = np.mean(list(average_scores.values()))
    print(f"Mean Accuracy of all experiments: {mean_all_experiments * 100:.2f}%")


def main(subject=None, run=None, mode=None):
    config = get_program_config()
    if subject is not None and run is not None:
        if mode == 'train':
            score = train_subject(subject, [run], config)
            print(f"Cross Val Score: {score * 100:.2f}%")
        if mode == 'predict':
            return predict_subject(subject, run, config)

    else:
        train_all_subjects(config)


if __name__ == '__main__':
    main(subject=3, run=3, mode='train')
    # main(subject=3, run=3, mode='predict')
    # main()
