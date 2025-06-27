import _thread as thread
import threading
from multiprocessing import cpu_count
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

import platform
import signal
import os

import pandas as pd
import numpy as np
import time


def get_time_chunks(data, window, n_chunks):

    data = data.sort_index()
    if isinstance(window, str):
        window = pd.Timedelta(window)

    idx = data.index
    cuts = np.linspace(0, len(idx), n_chunks + 1, dtype=int)

    chunks = []
    cut_times = []

    for i in range(n_chunks):
        start, end = cuts[i], cuts[i + 1]

        # Для первого чанка не делаем перекрытие
        if i == 0:
            overlap_start = start
        else:
            overlap_start_time = idx[start] - window
            overlap_start = idx.searchsorted(overlap_start_time, side="right")

        chunk = data.iloc[overlap_start:end]
        cut_time = idx[start]  # официальная граница начала чанка

        chunks.append(chunk)
        cut_times.append(cut_time)

    return chunks, cut_times


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


def _rank(mat):
    return np.argsort(np.argsort(mat, axis=0), axis=0)


def parallel_rank(mat, n_cpu):
    matrix_parts = np.array_split(mat, n_cpu, axis=1)
    with ThreadPoolExecutor(n_cpu) as pool:
        return np.hstack(list(pool.map(_rank, matrix_parts)))


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
    if isinstance(offset, (str, pd.Timedelta, pd.offsets.BaseOffset)):
        return get_time_chunks(df, window=offset, n_chunks=split_size)
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
