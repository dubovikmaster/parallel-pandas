from __future__ import annotations

import inspect
from functools import partial, cached_property
from multiprocessing import cpu_count

import pandas as pd
import dill

from .progress_imap import progress_imap
from .progress_imap import progress_udf_wrapper
from .progress_imap import get_workers_queue
from .tools import (
    get_split_data,
    resolve_split_size,
)

# Runtime configuration shared with ParallelPandas.initialize. The ``.str``/``.dt``
# parallel accessors are reached through the ``.parallel`` namespace, which has no
# closure of its own, so the chunking/pool settings live here.
_CONFIG = {
    'n_cpu': None,
    'split_factor': None,
    'disable_pr_bar': False,
    'show_vmem': False,
}


def set_parallel_config(n_cpu=None, split_factor=None, disable_pr_bar=False, show_vmem=False):
    _CONFIG.update(
        n_cpu=n_cpu,
        split_factor=split_factor,
        disable_pr_bar=disable_pr_bar,
        show_vmem=show_vmem,
    )


def _do_strdt_chunk(chunk, dill_fn, workers_queue):
    fn = dill.loads(dill_fn)

    def foo():
        return fn(chunk)

    return progress_udf_wrapper(foo, workers_queue, 1)()


def _run_chunked(series, fn, executor, desc):
    n_cpu = _CONFIG['n_cpu']
    split_factor = _CONFIG['split_factor']
    split_size = resolve_split_size(series, 1, n_cpu, split_factor)

    # Not worth spinning up a pool when we cannot even fill the workers.
    resolved_cpu = n_cpu or cpu_count()
    if split_size <= 1 or len(series) < resolved_cpu:
        return fn(series)

    workers_queue = get_workers_queue()
    tasks = get_split_data(series, 1, split_size)
    dill_fn = dill.dumps(fn, recurse=True)
    result = progress_imap(
        partial(_do_strdt_chunk, dill_fn=dill_fn, workers_queue=workers_queue),
        tasks, workers_queue, n_cpu=n_cpu, total=split_size,
        disable=_CONFIG['disable_pr_bar'], show_vmem=_CONFIG['show_vmem'],
        executor=executor, desc=desc,
    )
    return pd.concat(result)


class ParallelStrDt:
    """Parallel proxy for the ``.str`` / ``.dt`` accessors of a Series.

    Reached through the ``.parallel`` namespace, e.g. ``s.parallel.str.lower()``
    or ``s.parallel.dt.year``. Every string/datetime op is element-wise, so the
    Series is split into row-chunks, the real pandas accessor runs on each chunk
    in a worker pool and the pieces are concatenated back together.

    Methods (``lower``, ``extract``, ``floor``, ...) return a callable; accessor
    *properties* (``dt.year``, ``dt.month``, ``str`` has none) are computed
    eagerly and return the result directly.
    """

    __slots__ = ('_series', '_accessor', '_acc_obj')

    def __init__(self, series, accessor):
        self._series = series
        self._accessor = accessor
        # Instantiating the accessor is cheap (it only validates the dtype) and
        # lets us introspect method-vs-property without touching the data.
        self._acc_obj = getattr(series, accessor)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            # getattr_static returns the descriptor without triggering a property.
            raw = inspect.getattr_static(self._acc_obj, name)
        except AttributeError:
            raise AttributeError(
                f"'{self._accessor}' accessor has no attribute {name!r}"
            ) from None

        accessor = self._accessor
        is_property = isinstance(raw, (property, cached_property)) or not callable(raw)

        if is_property:
            fn = _make_property_fn(accessor, name)
            fn.__name__ = f'{accessor}.{name}'
            return _run_chunked(self._series, fn, 'processes', name.upper())

        def method(*args, **kwargs):
            fn = _make_method_fn(accessor, name, args, kwargs)
            fn.__name__ = f'{accessor}.{name}'
            return _run_chunked(self._series, fn, 'processes', name.upper())

        method.__name__ = name
        return method

    def __dir__(self):
        return sorted(n for n in dir(self._acc_obj) if not n.startswith('_'))


def _make_method_fn(accessor, name, args, kwargs):
    def fn(chunk):
        return getattr(getattr(chunk, accessor), name)(*args, **kwargs)
    return fn


def _make_property_fn(accessor, name):
    def fn(chunk):
        return getattr(getattr(chunk, accessor), name)
    return fn
