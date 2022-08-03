from functools import partial
from multiprocessing import cpu_count, Manager

import pandas as pd
import dill

from .progress_imap import progress_imap
from .progress_imap import progress_udf_wrapper
from .tools import get_split_data


def _do_apply(data, dill_func, workers_queue, axis, raw, result_type, args, kwargs):
    return data.apply(progress_udf_wrapper(dill_func, workers_queue, data.shape[1 - axis]), axis=axis, raw=raw,
                      result_type=result_type, args=args, **kwargs)


def _get_split_size(n_cpu):
    if n_cpu is None:
        n_cpu = cpu_count()
    return n_cpu * 4


def parallelize_apply(n_cpu=None, disable_pr_bar=False, error_behavior='raise', set_error_value=None):
    def parallel_apply(data, func, axis=0, raw=False, result_type=None, args=(), **kwargs):
        workers_queue = Manager().Queue()
        split_size = _get_split_size(n_cpu)
        tasks = get_split_data(data, axis, split_size)
        dill_func = dill.dumps(func)
        result = progress_imap(partial(_do_apply, axis=axis, raw=raw, result_type=result_type, dill_func=dill_func,
                                       workers_queue=workers_queue, args=args, kwargs=kwargs),
                               tasks, workers_queue, n_cpu=n_cpu, total=data.shape[1 - axis], disable=disable_pr_bar,
                               set_error_value=set_error_value, error_behavior=error_behavior)
        concat_axis = 0
        if result:
            if isinstance(result[0], pd.DataFrame):
                concat_axis = 1 - axis
        return pd.concat(result, axis=concat_axis)

    return parallel_apply


def _do_split_apply(data, dill_func, workers_queue, args, kwargs):
    return progress_udf_wrapper(dill_func, workers_queue, 1)(data, *args, **kwargs)


def parallelize_split(n_cpu=None, disable_progress_bar=False, error_behavior='raise', set_error_value=None):
    def split_apply(data, func, split_size=None, args=(), **kwargs):
        workers_queue = Manager().Queue()
        if split_size is None:
            split_size = _get_split_size(n_cpu)
        tasks = get_split_data(data, 1, split_size)
        dill_func = dill.dumps(func)
        result = progress_imap(partial(_do_split_apply, dill_func=dill_func, args=args, kwargs=kwargs), tasks,
                               workers_queue, total=split_size, n_cpu=n_cpu, disable=disable_progress_bar,
                               set_error_value=set_error_value, error_behavior=error_behavior)
        return pd.concat(result)

    return split_apply
