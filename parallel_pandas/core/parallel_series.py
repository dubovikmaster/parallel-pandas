from __future__ import annotations

from functools import partial
from multiprocessing import Manager

import pandas as pd
from pandas.util._decorators import doc

import dill

from .progress_imap import progress_imap
from .progress_imap import progress_udf_wrapper
from .tools import (
    get_split_data,
    get_split_size,
)

DOC = 'Parallel analogue of the pd.Series.{func} method\nSee pandas Series docstring for more ' \
      'information\nhttps://pandas.pydata.org/pandas-docs/stable/reference/series.html'


def _do_apply(data, dill_func, workers_queue, convert_dtype, args, kwargs):
    func = dill.loads(dill_func)
    return data.apply(progress_udf_wrapper(func, workers_queue, data.shape[0]),
                      convert_dtype=convert_dtype, args=args, **kwargs)


def series_parallelize_apply(n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
    @doc(DOC, func='apply')
    def p_apply(data, func, executor='processes', convert_dtype=True, args=(), **kwargs):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        tasks = get_split_data(data, 1, split_size)
        dill_func = dill.dumps(func)
        result = progress_imap(partial(_do_apply, convert_dtype=convert_dtype, dill_func=dill_func,
                                       workers_queue=workers_queue, args=args, kwargs=kwargs),
                               tasks, workers_queue, n_cpu=n_cpu, total=data.shape[0], disable=disable_pr_bar,
                               show_vmem=show_vmem, executor=executor, desc=func.__name__.upper())

        return pd.concat(result, copy=False)

    return p_apply


def _do_map(data, dill_arg, workers_queue, na_action):
    func = dill.loads(dill_arg)
    def foo():
        return data.map(func, na_action=na_action)
    return progress_udf_wrapper(foo, workers_queue, 1)()


def series_parallelize_map(n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
    @doc(DOC, func='map')
    def p_map(data, arg, executor='threads', na_action=None):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        tasks = get_split_data(data, 1, split_size)
        dill_arg = dill.dumps(arg)
        result = progress_imap(partial(_do_map, dill_arg=dill_arg,
                                       workers_queue=workers_queue, na_action=na_action),
                               tasks, workers_queue, n_cpu=n_cpu, total=split_size, disable=disable_pr_bar,
                               show_vmem=show_vmem, executor=executor, desc='map'.upper())

        return pd.concat(result, copy=False)

    return p_map
