import _thread as thread
import threading
from functools import wraps
import platform
import signal
import os

import pandas as pd
import numpy as np


def get_pandas_version():
    major, minor = pd.__version__.split(".")[:2]
    return int(major), int(minor)


def iterate_by_df(df, idx, axis):
    if axis:
        for i in idx:
            yield df.iloc[i, :]
    else:
        for i in idx:
            yield df.iloc[:, i]


def get_split_data(df, axis, split_size):
    split_size = min(split_size, df.shape[1 - axis])
    idx_split = np.array_split(np.arange(df.shape[1 - axis]), split_size)
    tasks = iterate_by_df(df, idx_split, axis)
    return tasks


def stop_function():
    if platform.system() == 'Windows':
        thread.interrupt_main()
    else:
        os.kill(os.getpid(), signal.SIGINT)


def stopit_after_timeout(s, raise_exception=True):
    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer = threading.Timer(s, stop_function)
            try:
                timer.start()
                result = func(*args, **kwargs)
            except KeyboardInterrupt:
                msg = f'function \"{func.__name__}\" took longer than {s} s.'
                if raise_exception:
                    raise TimeoutError(msg)
                result = msg
            finally:
                timer.cancel()
            return result

        return wrapper

    return actual_decorator


def _wrapped_func(func, s, raise_exception, *args, **kwargs):
    return stopit_after_timeout(s, raise_exception=raise_exception)(func)(*args, **kwargs)
