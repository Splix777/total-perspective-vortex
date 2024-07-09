import signal
from functools import wraps


class CustomTimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise CustomTimeoutError("Function call timed out")


def time_limit(limit: int, verbose: bool = False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(limit)

            try:
                result = func(*args, **kwargs)
            except CustomTimeoutError:
                if verbose:
                    print(f"{func.__name__} timed out after {limit} secs")
                return None
            finally:
                signal.alarm(0)

            return result

        return wrapper

    return decorator


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        elapsed_time = round(end - start, 2)
        function_name = func.__name__.replace("_", " ").title()
        print(f"{function_name} took {elapsed_time} secs")

        return result

    return wrapper
