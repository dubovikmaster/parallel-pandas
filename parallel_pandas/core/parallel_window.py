from __future__ import annotations

from functools import partial
from multiprocessing import cpu_count, Manager
import pandas as pd
from pandas.util._decorators import doc
import dill
from .progress_imap import progress_imap
from .progress_imap import progress_udf_wrapper
from .tools import get_split_data

DOC = 'Parallel analogue of the {func} method\nSee pandas DataFrame docstring for more ' \
      'information\nhttps://pandas.pydata.org/docs/reference/window.html'


class ParallelRolling:
    def __init__(self, n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
        self.n_cpu = n_cpu if n_cpu else cpu_count()
        self.disable_pr_bar = disable_pr_bar
        self.show_vmem = show_vmem
        self.split_factor = split_factor

    @staticmethod
    def _get_method(df, name, kwargs):
        return getattr(df.rolling(**kwargs), name)

    def do_method(self, df, workers_queue, name, window_attr, args, kwargs):
        if name == 'apply':
            # need to deserialize the function
            args = (dill.loads(args[0]),)

        def foo():
            method = self._get_method(df, name, window_attr)
            return method(*args, **kwargs)

        return progress_udf_wrapper(foo, workers_queue, 1)()

    @staticmethod
    def _get_axis_and_offset(data):
        axis = 1
        offset = data.window
        if isinstance(data.obj, pd.DataFrame):
            axis = data.axis
            offset = 0
        return axis, offset

    def _get_split_data(self, data):
        axis, offset = self._get_axis_and_offset(data)
        return get_split_data(data.obj, axis, self.n_cpu * self.split_factor, offset=offset)

    def _get_total_tasks(self, data):
        axis = data.axis
        if isinstance(data.obj, pd.Series):
            axis = 1
        return min(self.n_cpu * self.split_factor, data.obj.shape[1 - axis])

    def _data_reduce(self, result, data):
        axis, offset = self._get_axis_and_offset(data)
        if offset:
            result = [result[0]] + [s[offset:] for s in result[1:]]
            return pd.concat(result, axis=1 - axis, copy=False, ignore_index=True)
        return pd.concat(result, axis=1 - axis, copy=False)

    def parallelize_method(self, data, name, executor, *args, **kwargs):
        attributes = {attribute: getattr(data, attribute) for attribute in data._attributes}
        attributes.pop("_grouper", None)
        workers_queue = Manager().Queue()
        tasks = self._get_split_data(data)
        total = self._get_total_tasks(data)
        if name == 'apply':
            # need to serialize the function
            args = (dill.dumps(args[0]),)
        result = progress_imap(
            partial(self.do_method, workers_queue=workers_queue, args=args, kwargs=kwargs, name=name,
                    window_attr=attributes),
            tasks, workers_queue, n_cpu=self.n_cpu, disable=self.disable_pr_bar, show_vmem=self.show_vmem,
            total=total, desc=name.upper(), executor=executor,
        )
        return self._data_reduce(result, data)

    def do_parallel(self, name):
        if name == 'mean':
            @doc(DOC, func=name)
            def p_mean(data, *args, executor='threads', engine=None, engine_kwargs=None, **kwargs):
                return self.parallelize_method(data, name, executor, *args, engine=engine, engine_kwargs=engine_kwargs,
                                               **kwargs)

            return p_mean
        if name == 'median':
            @doc(DOC, func=name)
            def p_median(data, executor='threads', engine=None, engine_kwargs=None, **kwargs):
                return self.parallelize_method(data, name, executor, engine=engine, engine_kwargs=engine_kwargs,
                                               **kwargs)

            return p_median

        if name == 'sum':
            @doc(DOC, func=name)
            def p_sum(data, *args, executor='threads', engine=None, engine_kwargs=None, **kwargs):
                return self.parallelize_method(data, name, executor, *args, engine=engine, engine_kwargs=engine_kwargs,
                                               **kwargs)

            return p_sum

        if name == 'min':
            @doc(DOC, func=name)
            def p_min(data, *args, executor='threads', engine=None, engine_kwargs=None, **kwargs):
                return self.parallelize_method(data, name, executor, *args, engine=engine, engine_kwargs=engine_kwargs,
                                               **kwargs)

            return p_min

        if name == 'max':
            @doc(DOC, func=name)
            def p_max(data, *args, executor='threads', engine=None, engine_kwargs=None, **kwargs):
                return self.parallelize_method(data, name, executor, *args, engine=engine, engine_kwargs=engine_kwargs,
                                               **kwargs)

            return p_max

        if name == 'std':
            @doc(DOC, func=name)
            def p_std(data, executor='threads', ddof=1, *args, engine=None, engine_kwargs=None, **kwargs):
                return self.parallelize_method(data, name, executor, ddof=ddof, *args, engine=engine,
                                               engine_kwargs=engine_kwargs,
                                               **kwargs)

            return p_std

        if name == 'var':
            @doc(DOC, func=name)
            def p_var(data, executor='threads', ddof=1, *args, engine=None, engine_kwargs=None, **kwargs):
                return self.parallelize_method(data, name, executor, ddof=ddof, *args, engine=engine,
                                               engine_kwargs=engine_kwargs,
                                               **kwargs)

            return p_var

        if name == 'sem':
            @doc(DOC, func=name)
            def p_sem(data, executor='threads', ddof=1, *args, **kwargs):
                return self.parallelize_method(data, name, executor, ddof=ddof, *args, **kwargs)

            return p_sem

        if name == 'skew':
            @doc(DOC, func=name)
            def p_skew(data, executor='threads', **kwargs):
                return self.parallelize_method(data, name, executor, **kwargs)

            return p_skew

        if name == 'kurt':
            @doc(DOC, func=name)
            def p_kurt(data, executor='threads', **kwargs):
                return self.parallelize_method(data, name, executor, **kwargs)

            return p_kurt

        if name == 'rank':
            @doc(DOC, func=name)
            def p_rank(data, executor='threads', method='average', ascending=True, pct=False, **kwargs):
                return self.parallelize_method(data, name, executor, method=method, ascending=ascending, pct=pct,
                                               **kwargs)

            return p_rank

        if name == 'quantile':
            @doc(DOC, func=name)
            def p_quantile(data, quantile, executor='threads', interpolation="linear", **kwargs):
                return self.parallelize_method(data, name, executor, quantile, interpolation=interpolation, **kwargs)

            return p_quantile

        if name == 'apply':
            @doc(DOC, func=name)
            def p_apply(data, func, executor='threads', raw=False, engine=None, engine_kwargs=None, args=None,
                        kwargs=None):
                return self.parallelize_method(data, name, executor, func, raw=raw, engine=engine,
                                               engine_kwargs=engine_kwargs,
                                               args=args, kwargs=kwargs)

            return p_apply


class ParallelGroupbyMixin(ParallelRolling):

    def do_method(self, data, workers_queue, name, window_attr, args, kwargs):
        if name == 'apply':
            # need to deserialize the function
            args = (dill.loads(args[0]),)

        def foo():
            method = self._get_method(data[1], name, window_attr)
            result = method(*args, **kwargs)
            idx = [[i] for i in data[0]] + [result.index.tolist()]

            result.index = pd.MultiIndex.from_product(idx)
            return result

        return progress_udf_wrapper(foo, workers_queue, 1)()

    def _get_split_data(self, data):
        return data._grouper.get_iterator(data.obj)

    def _get_total_tasks(self, data):
        return data._grouper.ngroups

    def _data_reduce(self, result, data):
        out = pd.concat(result, copy=False)
        out.rename_axis(data._grouper.names + [data._grouper.axis.name], inplace=True)
        return out


class ParallelRollingGroupby(ParallelGroupbyMixin, ParallelRolling):
    pass


class ParallelExpanding(ParallelRolling):

    @staticmethod
    def _get_method(df, name, kwargs):
        return getattr(df.expanding(**kwargs), name)

    @staticmethod
    def _get_axis_and_offset(data):
        if isinstance(data.obj, pd.DataFrame):
            axis = data.axis
            offset = 0
        else:
            raise NotImplementedError('Parallel methods for Series objects are not implemented.')
        return axis, offset


class ParallelExpandingGroupby(ParallelGroupbyMixin, ParallelExpanding):
    pass


class ParallelEWM(ParallelRolling):

    @staticmethod
    def _get_axis_and_offset(data):
        if isinstance(data.obj, pd.DataFrame):
            axis = data.axis
            offset = 0
        else:
            raise NotImplementedError('Parallel methods for Series objects are not implemented.')
        return axis, offset

    @staticmethod
    def _get_method(df, name, kwargs):
        return getattr(df.ewm(**kwargs), name)


class ParallelEWMGroupby(ParallelGroupbyMixin, ParallelEWM):
    pass
