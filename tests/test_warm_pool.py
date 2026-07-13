import pandas as pd
import pytest

from parallel_pandas import set_reuse_pool
from parallel_pandas.core import progress_imap as pi


def _square(x):
    return x * x


@pytest.fixture(autouse=True)
def _reset_pools():
    """Every test starts and ends with a clean, reuse-enabled pool cache."""
    pi._shutdown_pools()
    set_reuse_pool(True)
    yield
    pi._shutdown_pools()
    set_reuse_pool(True)


def _run(executor, n_cpu=2):
    q = pi.get_workers_queue()
    tasks = list(range(20))
    return pi.progress_imap(
        _square, tasks, q, executor=executor, n_cpu=n_cpu, total=len(tasks), disable=True
    )


@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_results_are_correct(executor):
    assert _run(executor) == [x * x for x in range(20)]


@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_pool_is_reused_across_calls(executor):
    _run(executor)
    pool_first = pi._POOLS[(executor, 2)]
    _run(executor)
    pool_second = pi._POOLS[(executor, 2)]
    assert pool_first is pool_second
    assert len(pi._POOLS) == 1


def test_pools_keyed_by_executor_and_ncpu():
    _run("threads", n_cpu=2)
    _run("processes", n_cpu=2)
    _run("threads", n_cpu=3)
    assert set(pi._POOLS) == {("threads", 2), ("processes", 2), ("threads", 3)}


@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_reuse_disabled_does_not_cache(executor):
    set_reuse_pool(False)
    assert _run(executor) == [x * x for x in range(20)]
    assert pi._POOLS == {}


def test_disabling_shuts_down_existing_pools():
    _run("threads")
    assert pi._POOLS
    set_reuse_pool(False)
    assert pi._POOLS == {}


@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_broken_pool_is_dropped_and_recreated(executor):
    _run(executor)
    broken = pi._POOLS[(executor, 2)]
    broken.terminate()
    broken.join()
    # A call against the terminated pool must eventually leave the cache healthy:
    # either it raises and drops the pool, or (threads) it just recreates cleanly.
    try:
        _run(executor)
    except Exception:
        pass
    # After recovery a subsequent call always succeeds with a fresh pool.
    assert _run(executor) == [x * x for x in range(20)]
    assert pi._POOLS[(executor, 2)] is not broken


def test_repeated_p_apply_warm(df):
    # Two consecutive process-executor applies must both work with a warm pool
    # and agree with pandas.
    expected = df.apply(lambda row: row.sum(), axis=1)
    r1 = df.p_apply(lambda row: row.sum(), axis=1, executor="processes")
    r2 = df.p_apply(lambda row: row.sum(), axis=1, executor="processes")
    pd.testing.assert_series_equal(r1, expected)
    pd.testing.assert_series_equal(r2, expected)
