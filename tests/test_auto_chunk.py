import numpy as np
import pandas as pd
import pytest

from parallel_pandas import ParallelPandas
from parallel_pandas.core.tools import (
    auto_split_size,
    resolve_split_size,
    get_split_size,
    _TARGET_CHUNK_BYTES,
    _MAX_CHUNKS_PER_CPU,
)

ParallelPandas.initialize(n_cpu=4)


def test_get_split_size_none_is_one_factor():
    # None must behave like the historical default split_factor=1 (n_cpu chunks),
    # so the non-transport reductions keep their previous chunking.
    assert get_split_size(4, None) == 4
    assert get_split_size(4, 1) == 4
    assert get_split_size(4, 3) == 12


def test_resolve_explicit_factor_ignores_data_size():
    df = pd.DataFrame(np.ones((1_000_000, 20)))
    assert resolve_split_size(df, 1, 4, 2) == get_split_size(4, 2) == 8
    assert resolve_split_size(df, 1, 4, 1) == 4


def test_auto_small_frame_floors_at_n_cpu():
    # A tiny frame has far less than one target chunk of data -> exactly n_cpu
    # chunks (but never more than the split dimension length).
    df = pd.DataFrame(np.ones((100, 4)))
    n_cpu = 4
    got = auto_split_size(df, 1, n_cpu)
    assert got == n_cpu


def test_auto_never_exceeds_split_dim():
    # Split dimension shorter than n_cpu must clamp the chunk count.
    df = pd.DataFrame(np.ones((3, 4)))
    got = auto_split_size(df, 1, 8)  # split dim = rows = 3
    assert got == 3


def test_auto_targets_bytes_per_chunk_for_big_frame():
    n_cpu = 4
    rows, cols = 2_000_000, 10  # ~160 MB of float64
    df = pd.DataFrame(np.ones((rows, cols)))
    total_bytes = df.memory_usage(index=False, deep=False).sum()
    expected = int(np.ceil(total_bytes / _TARGET_CHUNK_BYTES))
    expected = min(max(expected, n_cpu), _MAX_CHUNKS_PER_CPU * n_cpu)
    got = auto_split_size(df, 1, n_cpu)
    assert got == expected
    # sanity: each chunk is roughly the target size
    assert _TARGET_CHUNK_BYTES // 2 <= total_bytes / got <= _TARGET_CHUNK_BYTES * 2


def test_auto_caps_chunks_per_cpu():
    n_cpu = 2
    # Force desired chunk count above the per-cpu cap via a large split dimension.
    df = pd.DataFrame(np.ones((5_000_000, 40)))  # ~1.6 GB -> ~190 chunks desired
    got = auto_split_size(df, 1, n_cpu)
    assert got == _MAX_CHUNKS_PER_CPU * n_cpu


def test_auto_series():
    s = pd.Series(np.ones(500))
    assert auto_split_size(s, 1, 4) == 4  # tiny -> n_cpu
    short = pd.Series(np.ones(2))
    assert auto_split_size(short, 1, 4) == 2  # clamped to length


@pytest.mark.parametrize("shape", [(10_000, 8), (200_000, 5)])
def test_p_apply_auto_matches_pandas(shape):
    df = pd.DataFrame(np.random.random(shape))
    res = df.p_apply(lambda col: col.sum(), axis=0)
    expected = df.apply(lambda col: col.sum(), axis=0)
    pd.testing.assert_series_equal(res, expected)


def test_p_apply_auto_and_explicit_agree():
    df = pd.DataFrame(np.random.random((50_000, 6)))
    auto = df.p_apply(lambda col: col.mean(), axis=0)
    ParallelPandas.initialize(n_cpu=4, split_factor=1)
    explicit = df.p_apply(lambda col: col.mean(), axis=0)
    pd.testing.assert_series_equal(auto, explicit)
    ParallelPandas.initialize(n_cpu=4)  # restore auto default for other tests
