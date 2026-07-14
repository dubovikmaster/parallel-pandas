from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np


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


def get_pandas_version():
    major, minor = pd.__version__.split(".")[:2]
    return int(major), int(minor)


def get_obj_axis(obj, default=0):
    # pandas >= 3 dropped the `axis` attribute on window/groupby objects;
    # only axis=0 is supported there, so fall back to the default.
    return getattr(obj, 'axis', default)


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
        split_factor = 1
    return n_cpu * split_factor


# Target payload per chunk for process-based methods. Derived from a chunk-size
# sweep across data shapes and UDF weights: throughput is stable around
# 8-16 MB/chunk, because the fixed per-chunk pickle/pipe cost is amortised while
# transfer still overlaps compute. See the auto-chunking PR for the measurements.
_TARGET_CHUNK_BYTES = 8 * 1024 * 1024
# Never create more than this many chunks per CPU, to keep per-task overhead
# bounded on very large frames (a huge frame lands near ~20 MB/chunk instead).
_MAX_CHUNKS_PER_CPU = 64


def _approx_nbytes(data):
    """Cheap dtype-based byte estimate for a Series/DataFrame (deep=False)."""
    usage = data.memory_usage(index=False, deep=False)
    try:
        return int(usage.sum())  # DataFrame -> per-column Series
    except AttributeError:
        return int(usage)        # Series -> scalar


def auto_split_size(data, axis, n_cpu):
    """Pick the number of chunks so each is ~``_TARGET_CHUNK_BYTES`` of data.

    Bounded below by ``n_cpu`` (keep every worker busy) and above by
    ``_MAX_CHUNKS_PER_CPU * n_cpu`` and by the length of the split dimension.
    """
    if n_cpu is None:
        n_cpu = cpu_count()
    total_bytes = _approx_nbytes(data)
    desired = max(1, -(-total_bytes // _TARGET_CHUNK_BYTES))  # ceil division
    n_chunks = min(max(desired, n_cpu), _MAX_CHUNKS_PER_CPU * n_cpu)
    split_dim = data.shape[1 - axis] if getattr(data, 'ndim', 1) > 1 else data.shape[0]
    return max(1, min(n_chunks, split_dim))


def resolve_split_size(data, axis, n_cpu, split_factor):
    """Chunk count for the process-transport methods.

    ``split_factor is None`` triggers the byte-size heuristic; an explicit factor
    keeps the classic ``n_cpu * split_factor`` behaviour.
    """
    if split_factor is None:
        return auto_split_size(data, axis, n_cpu)
    return get_split_size(n_cpu, split_factor)


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
