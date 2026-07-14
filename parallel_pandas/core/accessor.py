"""A ``.parallel`` accessor for pandas Series/DataFrame.

It lets you call the parallel methods as ``df.parallel.mean()`` instead of
``df.p_mean()``: every attribute ``x`` is dispatched to the monkey-patched
``p_x`` method on the underlying object, so the accessor automatically tracks
whatever ``ParallelPandas.initialize`` registered.
"""
import warnings

import pandas as pd

from .parallel_str_dt import ParallelStrDt


class ParallelAccessor:
    """``.parallel`` namespace dispatching to the ``p_*`` methods.

    Examples
    --------
    >>> df.parallel.mean()                 # -> df.p_mean()
    >>> df.parallel.apply(func, axis=1)     # -> df.p_apply(func, axis=1)
    >>> df.parallel.chunk_apply(func)       # -> df.chunk_apply(func)
    >>> s.parallel.str.lower()              # parallel Series.str.lower()
    >>> s.parallel.dt.year                  # parallel Series.dt.year
    """

    # Methods that intentionally keep their own name (no ``p_`` prefix).
    _ALIASES = {'chunk_apply': 'chunk_apply'}

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def str(self):
        """Parallel ``Series.str`` accessor (Series only)."""
        if not isinstance(self._obj, pd.Series):
            raise AttributeError("'str' accessor is only available on a Series")
        return ParallelStrDt(self._obj, 'str')

    @property
    def dt(self):
        """Parallel ``Series.dt`` accessor (Series only)."""
        if not isinstance(self._obj, pd.Series):
            raise AttributeError("'dt' accessor is only available on a Series")
        return ParallelStrDt(self._obj, 'dt')

    def __getattr__(self, name):
        # Guard against recursion / dunder lookups before ``_obj`` is set.
        if name.startswith('_'):
            raise AttributeError(name)
        target = self._ALIASES.get(name, f'p_{name}')
        try:
            return getattr(self._obj, target)
        except AttributeError:
            raise AttributeError(
                f"'parallel' accessor has no method {name!r}. Make sure "
                f"ParallelPandas.initialize() has been called; the method is "
                f"exposed as {target!r} on the object once initialized."
            ) from None

    def __dir__(self):
        names = [attr[2:] for attr in dir(type(self._obj)) if attr.startswith('p_')]
        names += list(self._ALIASES)
        if isinstance(self._obj, pd.Series):
            names += ['str', 'dt']
        return sorted(set(names))


_REGISTERED = False


def register_parallel_accessor():
    """Register the ``.parallel`` accessor on Series and DataFrame (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    # Silence pandas' "overriding an existing accessor" warning; registering the
    # same class under the same name is a no-op we deliberately allow.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pd.api.extensions.register_dataframe_accessor('parallel')(ParallelAccessor)
        pd.api.extensions.register_series_accessor('parallel')(ParallelAccessor)
    _REGISTERED = True
