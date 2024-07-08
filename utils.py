import json
import os


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
    if not config_path:
        raise ValueError('config_path must be a valid path to a config file')
    if not os.path.exists(config_path):
        raise FileNotFoundError('config_path must be a valid path')
    if not config_path.endswith('.json'):
        raise ValueError('config_path must be a valid json file')

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def define_experiment(run: int, config: dict):
    """
    Determines the experiment name based on the run
    number and configuration.

    Args:
        run (int): The run number.
        config (dict): Configuration settings containing
            experiments and their corresponding runs.

    Returns:
        str: The name of the experiment associated
            with the given run number.

    Raises:
        ValueError: If the run number does not match
            any experiment.
    """
    experiments = config['experiments']
    experiment_name = next(
        (
            experiment['name']
            for experiment in experiments
            if run in experiment['runs']
        ),
        None,
    )
    if experiment_name is None:
        raise ValueError(f'Invalid run {run}')

    return experiment_name
