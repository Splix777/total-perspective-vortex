import json
import os


def get_program_config(config_path: str = 'config.json'):
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
    experiments = config['experiments']
    experiment_name = None
    for experiment in experiments:
        if run in experiment['runs']:
            experiment_name = experiment['name']
            break

    if experiment_name is None:
        raise ValueError(f'Invalid run {run}')

    return experiment_name