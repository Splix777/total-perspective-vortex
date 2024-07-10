import contextlib
import os
import signal
import sys
import json
import traceback

from pathlib import Path

from mne.datasets.eegbci import eegbci


def get_program_config(config_path: str = 'config.json'):
    """
    Loads and returns the program configuration from a JSON file.

    Args:
        config_path (str): Path to the JSON configuration
            file. Defaults to 'config.json'.

    Returns:
        dict: The program configuration loaded from the JSON file.

    Raises:
        ValueError: If the config_path is empty, not a JSON file,
            or not a valid path.
        FileNotFoundError: If the config_path does not exist.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        raise ValueError('config_path must be a valid json file') from e
    except FileNotFoundError as e:
        raise FileNotFoundError(f'{config_path} does not exist') from e
    except Exception as e:
        raise ValueError('config_path must be a valid path') from e


def get_directory(directory: str) -> Path:
    """
    Retrieves the directory path from the configuration file.

    Args:
        directory (str): The directory key to retrieve.
        config_path (str): Path to the JSON configuration
            file. Defaults to 'config.json'.

    Returns:
        Path: The directory path.

    Raises:
        ValueError: If the directory key is not found in the
            configuration file.
    """
    config = get_program_config()
    try:
        directory_path = config['directories'][directory]
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / directory_path
    except KeyError as e:
        raise ValueError(f'Config file must contain "{directory}" key') from e


def check_runs_same_exp(runs: list[int], experiments: list[dict]) -> str:
    """
    Checks if all runs in the list belong to the same experiment.

    Args:
        runs (list[int]): List of run numbers to check.
        experiments (list[dict]): List of experiments
            from configuration.

    Returns:
        str: The name of the experiment

    Raises:
        ValueError: If runs do not belong to the same experiment.
    """
    if not runs:
        raise ValueError('The runs list cannot be empty')

    experiment_names = set()
    for run in runs:
        experiment_name = next(
            (
                experiment['name']
                for experiment in experiments
                if run in experiment['runs']),
            None
        )
        if experiment_name is None:
            raise ValueError(f'Invalid run {run}')
        experiment_names.add(experiment_name)

    if len(experiment_names) > 1:
        raise ValueError(f'Runs {runs} belong to different '
                         f'experiments: {experiment_names}')

    return experiment_names.pop()


def get_experiment(run: int | list[int]) -> dict:
    """
    Determines the experiment name based on the run
    number and configuration.

    Args:
        run (int) | (list[int]): The run number(s) to
            determine the experiment name for.

    Returns:
        dict: The experiment configuration.

    Raises:
        ValueError: If the run number does not match
            any experiment.
    """
    config = get_program_config()

    try:
        experiments = config['experiments']
        name = check_runs_same_exp(
            runs=[run] if isinstance(run, int) else run,
            experiments=experiments,
        )
        return next(
            (experiment for experiment in experiments
             if experiment['name'] == name),
            None
        )

    except KeyError as e:
        raise ValueError('Config file must contain "experiments" key') from e
    except Exception as e:
        raise ValueError(f'Invalid run {run}: {e}') from e


def verify_inputs(subject: int, run: int, mode: str):
    """
    Verifies the input arguments for the program.

    Args:
        subject (str): The subject number.
        run (str): The run number.
        mode (str): The mode of the program.

    Returns:
        tuple: A tuple containing the subject, run, and mode
            as integers and strings.

    Raises:
        ValueError: If the subject, run, or mode are not valid.
    """
    if subject < 1 or subject > 109:
        raise ValueError('Subject must be between 1 and 109')
    if run < 3 or run > 14:
        raise ValueError('Run must be between 3 and 14')
    if mode in {'train', 'predict'}:
        return subject, run, mode
    raise ValueError('Mode must be either "train" or "predict"')


def print_error_tree(e: Exception):
    """
    Prints the error tree from the exception.

    Args:
        e (Exception): The exception to print the error tree.

    Returns:
        None
    """
    tb = traceback.extract_tb(e.__traceback__)
    print('Traceback (most recent call last):')
    for frame in tb:
        print(f'File "{frame.filename}", line {frame.lineno}, in {frame.name}')
        print(f'  {frame.line}')
    print(f'Error: {e}')
    sys.exit(1)


def download_subjects():
    """
    Downloads the subjects from the dataset.
    """
    subjects = list(range(1, 110))

    def sigterm_handler(signum, frame):
        """
        Signal handler for SIGTERM.
        """
        print("Received SIGTERM. Terminating download process.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stderr(fnull):
                for subject in subjects:
                    eegbci.load_data(
                        subject=subject,
                        runs=list(range(3, 15)),
                        path='data/',
                        verbose=0
                    )

    except Exception as e:
        print(f"Error in downloading subjects: {e}")
        sys.exit(1)
    else:
        print("-----Finished downloading subjects-----")
        sys.exit(0)
