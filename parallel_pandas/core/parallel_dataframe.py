from __future__ import annotations

from functools import partial
from multiprocessing import Manager

import numpy as np
import pandas as pd
from pandas._libs import lib
from pandas.core import nanops
from pandas.util._decorators import doc
from pandas._typing import (
    IndexLabel,
    Suffixes
)
import dill

from .progress_imap import progress_imap
from .progress_imap import progress_udf_wrapper
from .tools import (
    get_split_data,
    get_split_size,
)

DOC = 'Parallel analogue of the DataFrame.{func} method\nSee pandas DataFrame docstring for more ' \
      'information\nhttps://pandas.pydata.org/docs/reference/frame.html '


def _do_apply(data, dill_func, workers_queue, axis, raw, result_type, args, kwargs):
    func = dill.loads(dill_func)
    return data.apply(progress_udf_wrapper(func, workers_queue, data.shape[1 - axis]), axis=axis, raw=raw,
                      result_type=result_type, args=args, **kwargs)


def parallelize_apply(n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
    @doc(DOC, func='apply')
    def p_apply(data, func, executor='processes', axis=0, raw=False, result_type=None, args=(), **kwargs):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        tasks = get_split_data(data, axis, split_size)
        dill_func = dill.dumps(func)
        result = progress_imap(partial(_do_apply, axis=axis, raw=raw, result_type=result_type, dill_func=dill_func,
                                       workers_queue=workers_queue, args=args, kwargs=kwargs),
                               tasks, workers_queue, n_cpu=n_cpu, total=data.shape[1 - axis], disable=disable_pr_bar,
                               show_vmem=show_vmem, executor=executor, desc=func.__name__.upper())
        concat_axis = 0
        if result:
            if isinstance(result[0], pd.DataFrame):
                concat_axis = 1 - axis
        return pd.concat(result, axis=concat_axis, copy=False)

    return p_apply


def _do_chunk_apply(data, dill_func, workers_queue, args, kwargs):
    func = dill.loads(dill_func)

    def foo():
        return func(data, *args, **kwargs)

    return progress_udf_wrapper(foo, workers_queue, 1)()


def parallelize_chunk_apply(n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
    def chunk_apply(data, func, executor='processes', axis=0, split_by_col=None, args=(), **kwargs):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        if split_by_col:
            idx_split = np.array_split(data[split_by_col].unique(), split_size)
            group = data.groupby(split_by_col)
            tasks = (pd.concat([group.get_group(j) for j in i], copy=False) for i in idx_split)
        else:
            tasks = get_split_data(data, 1 - axis, split_size)
        dill_func = dill.dumps(func)
        result = progress_imap(partial(_do_chunk_apply, dill_func=dill_func,
                                       workers_queue=workers_queue, args=args, kwargs=kwargs),
                               tasks, workers_queue, n_cpu=n_cpu, total=split_size, disable=disable_pr_bar,
                               show_vmem=show_vmem, executor=executor, desc='chunk_apply'.upper())
        return pd.concat(result, axis=axis, copy=False)

    return chunk_apply


def _do_replace(df, workers_queue, **kwargs):
    def foo():
        return df.replace(**kwargs)

    return progress_udf_wrapper(foo, workers_queue, 1)()


def parallelize_replace(n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
    @doc(DOC, func='replace')
    def p_replace(data, to_replace=None, value=lib.no_default, limit=None,
                  regex: bool = False, method: str | lib.NoDefault = lib.no_default):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        tasks = get_split_data(data, 1, split_size)
        result = progress_imap(partial(_do_replace, to_replace=to_replace, value=value, limit=limit, regex=regex,
                                       method=method, workers_queue=workers_queue), tasks, workers_queue,
                               total=split_size, n_cpu=n_cpu, disable=disable_pr_bar, show_vmem=show_vmem,
                               desc='REPLACE')

        return pd.concat(result, copy=False)

    return p_replace


def do_applymap(df, workers_queue, dill_func, na_action, kwargs):
    func = dill.loads(dill_func)

    return df.applymap(progress_udf_wrapper(func, workers_queue, df.size), na_action=na_action, **kwargs)


def parallelize_applymap(n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
    @doc(DOC, func='applymap')
    def p_applymap(data, func, na_action=None, **kwargs):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        tasks = get_split_data(data, 1, split_size)
        dill_func = dill.dumps(func)
        result = progress_imap(
            partial(do_applymap, workers_queue=workers_queue, dill_func=dill_func, na_action=na_action,
                    kwargs=kwargs), tasks, workers_queue, n_cpu=n_cpu, disable=disable_pr_bar, show_vmem=show_vmem,
            total=data.size, executor='processes', desc='APPLYMAP')
        return pd.concat(result, copy=False)

    return p_applymap


def do_describe(df, workers_queue, percentiles, include, exclude, datetime_is_numeric):
    if isinstance(df, pd.Series):
        df = df.to_frame()

    def foo():
        return df.describe(percentiles=percentiles, include=include, exclude=exclude,
                           datetime_is_numeric=datetime_is_numeric)

    return progress_udf_wrapper(foo, workers_queue, 1)()


def parallelize_describe(n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
    @doc(DOC, func='describe')
    def p_describe(data, percentiles=None, include=None, exclude=None,
                   datetime_is_numeric=False):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        tasks = get_split_data(data, 0, split_size)
        result = progress_imap(
            partial(do_describe, workers_queue=workers_queue, percentiles=percentiles, include=include, exclude=exclude,
                    datetime_is_numeric=datetime_is_numeric), tasks, workers_queue, n_cpu=n_cpu, disable=disable_pr_bar,
            show_vmem=show_vmem, total=min(split_size, data.shape[1]), desc='DESCRIBE')
        return pd.concat(result, copy=False, axis=1)

    return p_describe


def parallelize_nunique(n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
    @doc(DOC, func='nunique')
    def p_nunique(data, executor='threads', axis=0, dropna=True):
        return parallelize_apply(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar, show_vmem=show_vmem,
                                 split_factor=split_factor)(data, pd.Series.nunique, executor=executor,
                                                            axis=axis, dropna=dropna)

    return p_nunique


def do_mad(df, workers_queue, axis, skipna, level):
    def foo():
        return df.mad(axis=axis, skipna=skipna, level=level)

    return progress_udf_wrapper(foo, workers_queue, 1)()


def parallelize_mad(n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
    @doc(DOC, func='mad')
    def p_mad(data, axis=0, skipna=True, level=None):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        tasks = get_split_data(data, axis, split_size)
        total = min(split_size, data.shape[1 - axis])
        result = progress_imap(
            partial(do_mad, workers_queue=workers_queue, axis=axis, skipna=skipna, level=level), tasks, workers_queue,
            n_cpu=n_cpu, disable=disable_pr_bar, show_vmem=show_vmem, total=total, desc='MAD'
        )
        return pd.concat(result, copy=False)

    return p_mad


def do_idxmax(df, workers_queue, axis, skipna):
    def foo():
        return df.idxmax(axis=axis, skipna=skipna)

    return progress_udf_wrapper(foo, workers_queue, 1)()


def parallelize_idxmax(n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
    @doc(DOC, func='idxmax')
    def p_idxmax(data, axis=0, skipna=True):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        tasks = get_split_data(data, axis, split_size)
        total = min(split_size, data.shape[1 - axis])
        result = progress_imap(
            partial(do_idxmax, workers_queue=workers_queue, axis=axis, skipna=skipna), tasks, workers_queue,
            n_cpu=n_cpu, disable=disable_pr_bar, show_vmem=show_vmem, total=total, desc='IDXMAX'
        )
        return pd.concat(result, copy=False)

    return p_idxmax


def do_idxmin(df, workers_queue, axis, skipna):
    def foo():
        return df.idxmin(axis=axis, skipna=skipna)

    return progress_udf_wrapper(foo, workers_queue, 1)()


def parallelize_idxmin(n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
    @doc(DOC, func='idxmin')
    def p_idxmin(data, axis=0, skipna=True):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        tasks = get_split_data(data, axis, split_size)
        total = min(split_size, data.shape[1 - axis])
        result = progress_imap(
            partial(do_idxmin, workers_queue=workers_queue, axis=axis, skipna=skipna), tasks, workers_queue,
            n_cpu=n_cpu, disable=disable_pr_bar, show_vmem=show_vmem, total=total, desc='IDXMIN'
        )
        return pd.concat(result, copy=False)

    return p_idxmin


def do_rank(df, workers_queue, axis, method, numeric_only, na_option, ascending, pct):
    def foo():
        return df.rank(axis=axis, method=method, numeric_only=numeric_only, na_option=na_option, ascending=ascending,
                       pct=pct)

    return progress_udf_wrapper(foo, workers_queue, 1)()


def parallelize_rank(n_cpu=None, disable_pr_bar=False, split_factor=1,
                     show_vmem=False):
    @doc(DOC, func='rank')
    def p_rank(data, axis=0, method: str = "average", numeric_only=lib.no_default, na_option="keep", ascending=True,
               pct=False):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        tasks = get_split_data(data, axis, split_size)
        total = min(split_size, data.shape[1 - axis])
        result = progress_imap(
            partial(do_rank, workers_queue=workers_queue, axis=axis, method=method, numeric_only=numeric_only,
                    na_option=na_option, ascending=ascending, pct=pct), tasks, workers_queue,
            n_cpu=n_cpu, disable=disable_pr_bar, show_vmem=show_vmem, total=total, desc='RANK'
        )
        return pd.concat(result, axis=1 - axis, copy=False)

    return p_rank


def do_quantile(df, workers_queue, axis, q, numeric_only, interpolation):
    def foo():
        return df.quantile(axis=axis, q=q, numeric_only=numeric_only, interpolation=interpolation)

    return progress_udf_wrapper(foo, workers_queue, 1)()


def parallelize_quantile(n_cpu=None, disable_pr_bar=False, split_factor=1,
                         show_vmem=False):
    @doc(DOC, func='quantile')
    def p_quantile(data, axis=0, q=0.5, numeric_only: bool = True, interpolation: str = "linear"):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        tasks = get_split_data(data, axis, split_size)
        total = min(split_size, data.shape[1 - axis])
        result = progress_imap(
            partial(do_quantile, workers_queue=workers_queue, axis=axis, numeric_only=numeric_only, q=q,
                    interpolation=interpolation), tasks, workers_queue,
            n_cpu=n_cpu, disable=disable_pr_bar, show_vmem=show_vmem, total=total, desc='QUANTILE'
        )
        if not lib.is_list_like(q):
            return pd.concat(result, copy=False)
        return pd.concat(result, axis=1, copy=False)

    return p_quantile


def do_mode(df, workers_queue, axis, numeric_only, dropna):
    def foo():
        return df.mode(axis=axis, numeric_only=numeric_only, dropna=dropna)

    return progress_udf_wrapper(foo, workers_queue, 1)()


def parallelize_mode(n_cpu=None, disable_pr_bar=False, split_factor=1,
                     show_vmem=False):
    @doc(DOC, func='mode')
    def p_mode(data, executor='processes', axis=0, numeric_only: bool = False, dropna=True):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        tasks = get_split_data(data, axis, split_size)
        total = min(split_size, data.shape[1 - axis])
        result = progress_imap(
            partial(do_mode, workers_queue=workers_queue, axis=axis, numeric_only=numeric_only, dropna=dropna
                    ), tasks, workers_queue,
            n_cpu=n_cpu, disable=disable_pr_bar, show_vmem=show_vmem, total=total, executor=executor, desc='MODE'
        )
        return pd.concat(result, axis=1 - axis, copy=False)

    return p_mode


def do_merge(df, workers_queue, right, **kwargs):
    def foo():
        return df.merge(right, **kwargs)

    return progress_udf_wrapper(foo, workers_queue, 1)()


def parallelize_merge(n_cpu=None, disable_pr_bar=False, split_factor=1,
                      show_vmem=False):
    @doc(DOC, func='mode')
    def p_merge(data,
                right: pd.DataFrame | pd.Series,
                how: str = "inner",
                on: IndexLabel | None = None,
                left_on: IndexLabel | None = None,
                right_on: IndexLabel | None = None,
                left_index: bool = False,
                right_index: bool = False,
                sort: bool = False,
                suffixes: Suffixes = ("_x", "_y"),
                copy: bool = True,
                indicator: bool = False,
                validate: str | None = None, ):
        workers_queue = Manager().Queue()
        split_size = get_split_size(n_cpu, split_factor)
        tasks = get_split_data(data, 1, split_size)
        total = min(split_size, data.shape[0])
        result = progress_imap(
            partial(do_merge, workers_queue=workers_queue, how=how, right=right, on=on,
                    ), tasks, workers_queue,
            n_cpu=n_cpu, disable=disable_pr_bar, show_vmem=show_vmem, total=total, executor='threads', desc='MERGE'
        )
        return pd.concat(result, copy=False)

    return p_merge


class ParallelizeStatFunc:
    def __init__(self, n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
        self.n_cpu = n_cpu
        self.disable_pr_bar = disable_pr_bar
        self.show_vmem = show_vmem
        self.split_factor = split_factor

    @staticmethod
    def get_nanops_arg(name):
        if name == 'min':
            return nanops.nanmin
        if name == 'max':
            return nanops.nanmax
        if name == 'mean':
            return nanops.nanmean
        if name == 'median':
            return nanops.nanmedian
        if name == 'skew':
            return nanops.nanskew
        if name == 'kurt':
            return nanops.nankurt

    def _stat_func(self, df, workers_queue, name, axis, skipna, level, numeric_only, kwargs):
        def closure():
            return df._stat_function(name, self.get_nanops_arg(name), axis, skipna, level, numeric_only, **kwargs)

        return progress_udf_wrapper(closure, workers_queue, 1)()

    def _parallel_stat_func(self, data, name, kwargs, axis=0, skipna=True, level=None, numeric_only=None):
        workers_queue = Manager().Queue()
        split_size = get_split_size(self.n_cpu, self.split_factor)
        tasks = get_split_data(data, axis, split_size)
        total = min(split_size, data.shape[1 - axis])
        result = progress_imap(
            partial(self._stat_func, workers_queue=workers_queue, name=name, axis=axis, skipna=skipna,
                    level=level, numeric_only=numeric_only, kwargs=kwargs), tasks, workers_queue,
            total=total, n_cpu=self.n_cpu, disable=self.disable_pr_bar, show_vmem=self.show_vmem, desc=name.upper())
        return pd.concat(result, copy=False)

    def do_parallel(self, name):
        if name == 'min':
            @doc(DOC, func=name)
            def p_min(data, axis=0, skipna=True, level=None, numeric_only=None, **kwargs):
                return self._parallel_stat_func(data, name=name, axis=axis, skipna=skipna,
                                                level=level, numeric_only=numeric_only, kwargs=kwargs)

            return p_min
        if name == 'max':
            @doc(DOC, func=name)
            def p_max(data, axis=0, skipna=True, level=None, numeric_only=None, **kwargs):
                return self._parallel_stat_func(data, name=name, axis=axis, skipna=skipna,
                                                level=level, numeric_only=numeric_only, kwargs=kwargs)

            return p_max
        if name == 'mean':
            @doc(DOC, func=name)
            def p_mean(data, axis=0, skipna=True, level=None, numeric_only=None, **kwargs):
                return self._parallel_stat_func(data, name=name, axis=axis, skipna=skipna,
                                                level=level, numeric_only=numeric_only, kwargs=kwargs)

            return p_mean
        if name == 'median':
            @doc(DOC, func=name)
            def p_median(data, axis=0, skipna=True, level=None, numeric_only=None, **kwargs):
                return self._parallel_stat_func(data, name=name, axis=axis, skipna=skipna,
                                                level=level, numeric_only=numeric_only, kwargs=kwargs)

            return p_median
        if name == 'kurt':
            @doc(DOC, func=name)
            def p_kurt(data, axis=0, skipna=True, level=None, numeric_only=None, **kwargs):
                return self._parallel_stat_func(data, name=name, axis=axis, skipna=skipna,
                                                level=level, numeric_only=numeric_only, kwargs=kwargs)

            return p_kurt

        if name == 'skew':
            @doc(DOC, func=name)
            def p_skew(data, axis=0, skipna=True, level=None, numeric_only=None, **kwargs):
                return self._parallel_stat_func(data, name=name, axis=axis, skipna=skipna,
                                                level=level, numeric_only=numeric_only, kwargs=kwargs)

            return p_skew


class ParallelizeStatFuncDdof:
    def __init__(self, n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
        self.n_cpu = n_cpu
        self.disable_pr_bar = disable_pr_bar
        self.show_vmem = show_vmem
        self.split_factor = split_factor

    @staticmethod
    def get_nanops_arg(name):
        if name == 'sem':
            return nanops.nansem
        if name == 'var':
            return nanops.nanvar
        if name == 'std':
            return nanops.nanstd

    def _stat_func_ddof(self, df, workers_queue, name, axis, skipna, level, ddof, numeric_only, kwargs):
        def closure():
            return df._stat_function_ddof(name, self.get_nanops_arg(name), axis, skipna, level, ddof, numeric_only,
                                          **kwargs)

        return progress_udf_wrapper(closure, workers_queue, 1)()

    def _parallel_stat_func_ddof(self, data, name, kwargs, axis=0, skipna=True, level=None, ddof=1,
                                 numeric_only=None):
        workers_queue = Manager().Queue()
        split_size = get_split_size(self.n_cpu, self.split_factor)
        tasks = get_split_data(data, axis, split_size)
        total = min(split_size, data.shape[1 - axis])
        result = progress_imap(
            partial(self._stat_func_ddof, workers_queue=workers_queue, name=name, axis=axis, skipna=skipna,
                    level=level, ddof=ddof, numeric_only=numeric_only, kwargs=kwargs), tasks, workers_queue,
            total=total, n_cpu=self.n_cpu, disable=self.disable_pr_bar, show_vmem=self.show_vmem, desc=name.upper())
        return pd.concat(result, copy=False)

    def do_parallel(self, name):
        if name == 'var':
            @doc(DOC, func=name)
            def p_var(data, axis=0, skipna=True, level=None, ddof=1, numeric_only=None, **kwargs):
                return self._parallel_stat_func_ddof(data, name=name, axis=axis,
                                                     skipna=skipna, level=level, ddof=ddof, numeric_only=numeric_only,
                                                     kwargs=kwargs)

            return p_var
        if name == 'std':
            @doc(DOC, func=name)
            def p_std(data, axis=0, skipna=True, level=None, ddof=1, numeric_only=None, **kwargs):
                return self._parallel_stat_func_ddof(data, name=name, axis=axis,
                                                     skipna=skipna, level=level, ddof=ddof, numeric_only=numeric_only,
                                                     kwargs=kwargs)

            return p_std
        if name == 'sem':
            @doc(DOC, func=name)
            def p_sem(data, axis=0, skipna=True, level=None, ddof=1, numeric_only=None, **kwargs):
                return self._parallel_stat_func_ddof(data, name=name, axis=axis,
                                                     skipna=skipna, level=level, ddof=ddof, numeric_only=numeric_only,
                                                     kwargs=kwargs)

            return p_sem


class ParallelizeMinCountStatFunc:
    def __init__(self, n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
        self.n_cpu = n_cpu
        self.disable_pr_bar = disable_pr_bar
        self.show_vmem = show_vmem
        self.split_factor = split_factor

    @staticmethod
    def get_nanops_arg(name):
        if name == 'sum':
            return nanops.nansum
        if name == 'prod':
            return nanops.nanprod

    def _min_count_stat_func(self, df, workers_queue, name, axis, skipna, level, numeric_only, min_count, kwargs):
        def closure():
            return df._min_count_stat_function(name, self.get_nanops_arg(name), axis, skipna, level, numeric_only,
                                               min_count, **kwargs
                                               )

        return progress_udf_wrapper(closure, workers_queue, 1)()

    def _parallel_min_count_stat_func(self, data, name, kwargs, axis=0, skipna=True, level=None,
                                      numeric_only=None, min_count=0):
        workers_queue = Manager().Queue()
        split_size = get_split_size(self.n_cpu, self.split_factor)
        tasks = get_split_data(data, axis, split_size)
        total = min(split_size, data.shape[1 - axis])
        result = progress_imap(
            partial(self._min_count_stat_func, workers_queue=workers_queue, name=name, axis=axis, skipna=skipna,
                    level=level, min_count=min_count, numeric_only=numeric_only, kwargs=kwargs), tasks, workers_queue,
            total=total, n_cpu=self.n_cpu, disable=self.disable_pr_bar, show_vmem=self.show_vmem, desc=name.upper())
        return pd.concat(result, copy=False)

    def do_parallel(self, name):
        if name == 'sum':
            @doc(DOC, func=name)
            def p_sum(data, axis=0, skipna=True, level=None, numeric_only=None,
                      min_count=0, **kwargs):
                return self._parallel_min_count_stat_func(data, name=name, axis=axis,
                                                          skipna=skipna, level=level, min_count=min_count,
                                                          numeric_only=numeric_only, kwargs=kwargs)

            return p_sum
        if name == 'prod':
            @doc(DOC, func=name)
            def p_prod(data, axis=0, skipna=True, level=None, numeric_only=None,
                       min_count=0, **kwargs):
                return self._parallel_min_count_stat_func(data, name=name, axis=axis,
                                                          skipna=skipna, level=level, min_count=min_count,
                                                          numeric_only=numeric_only, kwargs=kwargs)

            return p_prod


class ParallelizeAccumFunc:
    def __init__(self, n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
        self.n_cpu = n_cpu
        self.disable_pr_bar = disable_pr_bar
        self.show_vmem = show_vmem
        self.split_factor = split_factor

    @staticmethod
    def get_func(name):
        if name == 'cummin':
            return np.minimum.accumulate
        if name == 'cummax':
            return np.maximum.accumulate
        if name == 'cumsum':
            return np.cumsum
        if name == 'cumprod':
            return np.cumprod

    def _concat_by_columns(self, columns):
        first = columns[0]
        for d in columns[1:]:
            first.update(d)
        return pd.DataFrame(first)

    def _accum_func(self, df, workers_queue, name, axis, skipna, args, kwargs):
        def closure():
            if not axis:
                return df._accum_func(name, self.get_func(name), axis, skipna, *args, **kwargs).to_dict(orient='series')
            return df._accum_func(name, self.get_func(name), axis, skipna, *args, **kwargs)

        return progress_udf_wrapper(closure, workers_queue, 1)()

    def _parallel_accum_func(self, data, name, args, kwargs, axis=0, skipna=True):
        workers_queue = Manager().Queue()
        split_size = get_split_size(self.n_cpu, self.split_factor)
        tasks = get_split_data(data, axis, split_size)
        total = min(split_size, data.shape[1 - axis])
        result = progress_imap(
            partial(self._accum_func, workers_queue=workers_queue, name=name, axis=axis, skipna=skipna,
                    args=args, kwargs=kwargs), tasks, workers_queue,
            total=total, n_cpu=self.n_cpu, disable=self.disable_pr_bar, show_vmem=self.show_vmem, desc=name.upper())
        if not axis:
            return self._concat_by_columns(result)
        return pd.concat(result, axis=1 - axis, copy=False)

    def do_parallel(self, name):
        if name == 'cumsum':
            @doc(DOC, func=name)
            def p_cumsum(data, axis=0, skipna=True, *args, **kwargs):
                return self._parallel_accum_func(data, name=name, axis=axis,
                                                 skipna=skipna, args=args, kwargs=kwargs)

            return p_cumsum

        if name == 'cumprod':
            @doc(DOC, func=name)
            def p_cumprod(data, axis=0, skipna=True, *args, **kwargs):
                return self._parallel_accum_func(data, name=name, axis=axis,
                                                 skipna=skipna, args=args, kwargs=kwargs)

            return p_cumprod

        if name == 'cummin':
            @doc(DOC, func=name)
            def p_cummin(data, axis=0, skipna=True, *args, **kwargs):
                return self._parallel_accum_func(data, name=name, axis=axis,
                                                 skipna=skipna, args=args, kwargs=kwargs)

            return p_cummin

        if name == 'cummax':
            @doc(DOC, func=name)
            def p_cummax(data, axis=0, skipna=True, *args, **kwargs):
                return self._parallel_accum_func(data, name=name, axis=axis,
                                                 skipna=skipna, args=args, kwargs=kwargs)

            return p_cummax
