from functools import partial

import numpy as np
import pandas as pd

from .progress_imap import progress_imap
from .progress_imap import get_workers_queue
from pandas.core.groupby.ops import _is_indexed_like
from pandas.util._decorators import doc

import dill

from .progress_imap import progress_udf_wrapper
from .tools import (
    get_pandas_version,
    get_split_size,
)

DOC = 'Parallel analogue of the GroupBy.{func} method\nSee pandas GroupBy docstring for more ' \
      'information\nhttps://pandas.pydata.org/docs/reference/groupby.html'

MAJOR_PANDAS_VERSION, MINOR_PANDAS_VERSION = get_pandas_version()

# pandas 3 dropped the trailing `axis` argument of _is_indexed_like.
_IS_INDEXED_LIKE_NARGS = 2 if MAJOR_PANDAS_VERSION >= 3 else 3


def _do_group_apply(data, dill_func, workers_queue, args, kwargs):
    func = dill.loads(dill_func)
    result = progress_udf_wrapper(func, workers_queue, 1)(data, *args, **kwargs)
    return result, data.axes, 0


def _prepare_result(data):
    mutated = False
    result = list()
    for d in data:
        if not mutated and not _is_indexed_like(*d[:_IS_INDEXED_LIKE_NARGS]):
            mutated = True
        result.append(d[0])
    return result, mutated


def _get_grouper(data):
    # `_grouper` on the groupby object only exists since pandas 2.2;
    # older versions expose it as the public `grouper` attribute.
    grouper = getattr(data, '_grouper', None)
    if grouper is None:
        grouper = data.grouper
    return grouper


def _get_group_iterator(data, include_groups):
    grouper = _get_grouper(data)
    if MAJOR_PANDAS_VERSION >= 3:
        # pandas 3 removed the `axis` argument from _get_splitter and the
        # `axis` attribute from the groupby object; only axis=0 exists.
        obj = data._selected_obj if include_groups else data._obj_with_exclusions
        return iter(grouper._get_splitter(obj))
    if MAJOR_PANDAS_VERSION == 2 and MINOR_PANDAS_VERSION >= 2 and not include_groups:
        return iter(grouper._get_splitter(data._obj_with_exclusions, data.axis))
    else:
        return iter(grouper._get_splitter(data._selected_obj, data.axis))


def parallelize_groupby_apply(n_cpu=None, disable_pr_bar=False):
    @doc(DOC, func='apply')
    def p_apply(data, func, executor='processes', include_groups=True, args=(), **kwargs):
        workers_queue = get_workers_queue()
        gr_count = data.ngroups
        iterator = _get_group_iterator(data, include_groups)
        dill_func = dill.dumps(func, recurse=True)
        result = progress_imap(
            partial(_do_group_apply, dill_func=dill_func, workers_queue=workers_queue, args=args, kwargs=kwargs),
            iterator, workers_queue, total=gr_count, n_cpu=n_cpu, disable=disable_pr_bar, executor=executor,
            desc=func.__name__.upper()
        )
        result, mutated = _prepare_result(result)

        # due to a bug in the get_iterator method of the Basegrouper class
        # that was only fixed in pandas 1.4.0, earlier versions are not yet supported

        # pandas_version = get_pandas_version()
        #
        # if pandas_version < (1, 3):
        #     return data._wrap_applied_output(data.grouper._get_group_keys(), result,
        #                                      not_indexed_same=mutated or data.mutated)
        # elif pandas_version < (1, 4):
        #     return data._wrap_applied_output(data.grouper._get_group_keys(), data.grouper._get_group_keys(), result,
        #                                      not_indexed_same=mutated or data.mutated)
        return data._wrap_applied_output(data._selected_obj, result, not_indexed_same=mutated)

    return p_apply


def _do_group_transform(task, dill_func, workers_queue, args, kwargs):
    sub, codes = task
    func = dill.loads(dill_func)

    def foo():
        # Re-group the chunk by the original (integer) group codes and let pandas
        # run the real transform: this reproduces groupby.transform semantics
        # exactly (column fast-path, broadcasting, string ops, ...) because every
        # group lives entirely inside a single chunk and transform is independent
        # across groups.
        return sub.groupby(codes, sort=False).transform(func, *args, **kwargs)

    return progress_udf_wrapper(foo, workers_queue, 1)()


def _chunk_of_row(ids, ngroups, n_chunks):
    """Assign every row to a chunk via its group code, keeping groups intact.

    ``ids`` are the per-row group codes (as returned by ``GroupBy.ngroup``);
    rows with a missing key have code ``-1`` / ``NaN`` and are assigned ``-1``
    (excluded from every chunk, so they stay NaN in the result, matching pandas).
    """
    ids = np.asarray(ids)
    valid = np.isfinite(ids) & (ids >= 0)
    chunk = np.full(ids.shape[0], -1, dtype=np.int64)
    gid = ids[valid].astype(np.int64)
    # Contiguous ranges of group ids map to the same chunk (like np.array_split).
    chunk[valid] = gid * n_chunks // ngroups
    return chunk


def _assemble_transform(obj, positions, pieces):
    """Scatter the per-chunk transform results back into original row order.

    Uses integer positions (not index labels) so it is correct even when the
    original index has duplicate labels.
    """
    combined = pd.concat(pieces, axis=0)
    pos = np.concatenate(positions) if positions else np.array([], dtype=np.int64)
    n = len(obj)
    fully_covered = len(pos) == n

    if isinstance(combined, pd.Series):
        if fully_covered:
            out = pd.Series(index=obj.index, name=combined.name, dtype=combined.dtype)
        else:
            out = pd.Series(np.nan, index=obj.index, name=combined.name)
        out.iloc[pos] = combined.to_numpy()
        return out

    out = pd.DataFrame(np.nan, index=obj.index, columns=combined.columns)
    out.iloc[pos, :] = combined.to_numpy()
    if fully_covered:
        out = out.astype(combined.dtypes.to_dict())
    return out


def parallelize_groupby_transform(n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=None):
    @doc(DOC, func='transform')
    def p_transform(data, func, executor='processes', args=(), **kwargs):
        obj = data._obj_with_exclusions
        ngroups = data.ngroups
        if ngroups == 0:
            return data.transform(func, *args, **kwargs)

        ids = data.ngroup().to_numpy()
        n_chunks = min(get_split_size(n_cpu, split_factor), ngroups)
        chunk_of_row = _chunk_of_row(ids, ngroups, n_chunks)

        masks = [chunk_of_row == c for c in range(n_chunks)]
        masks = [m for m in masks if m.any()]
        if not masks:
            return data.transform(func, *args, **kwargs)

        positions = [np.where(m)[0] for m in masks]
        tasks = ((obj.iloc[m], ids[m]) for m in masks)

        workers_queue = get_workers_queue()
        dill_func = dill.dumps(func, recurse=True)
        desc = (func.upper() if isinstance(func, str) else func.__name__.upper())
        pieces = progress_imap(
            partial(_do_group_transform, dill_func=dill_func, workers_queue=workers_queue,
                    args=args, kwargs=kwargs),
            tasks, workers_queue, total=len(masks), n_cpu=n_cpu, disable=disable_pr_bar,
            show_vmem=show_vmem, executor=executor, desc=desc
        )
        return _assemble_transform(obj, positions, pieces)

    return p_transform
