import json


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
            config = json.load(f)

        return config

    except json.JSONDecodeError as e:
        raise ValueError('config_path must be a valid json file') from e
    except FileNotFoundError as e:
        raise FileNotFoundError(f'{config_path} does not exist') from e
    except Exception as e:
        raise ValueError('config_path must be a valid path') from e


def check_runs_same_experiment(runs: list[int], experiments: list[dict]):
    """
    Checks if all runs in the list belong to the same experiment.

    Args:
        runs (list[int]): List of run numbers to check.
        experiments (list[dict]): List of experiments
            from configuration.

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


def get_experiment_name(run: int | list[int]):
    """
    Determines the experiment name based on the run
    number and configuration.

    Args:
        run (int) | (list[int]): The run number(s) to
            determine the experiment name for.

    Returns:
        str: The name of the experiment associated
            with the given run number.

    Raises:
        ValueError: If the run number does not match
            any experiment.
    """
    config = get_program_config()

    try:
        experiments = config['experiments']
        return check_runs_same_experiment(
            runs=[run] if isinstance(run, int) else run,
            experiments=experiments,
        )
    except KeyError as e:
        raise ValueError('Config file must contain "experiments" key') from e
    except Exception as e:
        raise ValueError(f'Invalid run {run}: {e}') from e


def verify_inputs(subject: str, run: str, mode: str):
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
    try:
        subject = int(subject)
        run = int(run)
        if subject < 1 or subject > 109:
            raise ValueError('Subject must be between 1 and 109')
        if run < 3 or run > 14:
            raise ValueError('Run must be between 3 and 14')
    except ValueError as e:
        raise ValueError('Subject and run must be integers') from e

    if mode in {'train', 'predict'}:
        return subject, run, mode
    else:
        raise ValueError('Mode must be "train" or "predict"')
