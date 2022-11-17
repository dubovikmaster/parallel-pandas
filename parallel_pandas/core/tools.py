import _thread as thread
import threading
from multiprocessing import cpu_count
from functools import wraps
from itertools import combinations

import platform
import signal
import os

import pandas as pd
import numpy as np
import time


def time_of_function(function):
    def wrapped(*args, **kwargs):
        start_time = time.time()
        res = function(*args, **kwargs)
        print('Time of function {} is {:.3f} s.'.format(function.__name__, time.time() - start_time))
        return res

    return wrapped


def get_pandas_version():
    major, minor = pd.__version__.split(".")[:2]
    return int(major), int(minor)


def get_comb_cnt(x):
    return len(list(combinations(range(x), 2)))


def get_col_combinations(df):
    iterator = combinations(range(df.shape[1]), 2)
    for idx in iterator:
        yield df.iloc[:, idx[0]], df.iloc[:, idx[1]]


def get_split_size(n_cpu, split_factor):
    if n_cpu is None:
        n_cpu = cpu_count()
    if split_factor is None:
        split_factor = 4
    return n_cpu * split_factor


def iterate_by_df(df, idx, axis, offset):
    if axis:
        for i in idx:
            start = max(0, i[0] - offset)
            yield df.iloc[start:i[-1] + 1]
    else:
        for i in idx:
            yield df.iloc[:, i[0]:i[-1] + 1]


def get_split_data(df, axis, split_size, offset=0):
    split_size = min(split_size, df.shape[1 - axis])
    idx_split = np.array_split(np.arange(df.shape[1 - axis]), split_size)
    tasks = iterate_by_df(df, idx_split, axis, offset)
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
