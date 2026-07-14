import numpy as np
import pandas as pd
import pytest

from parallel_pandas import ParallelPandas

ParallelPandas.initialize(n_cpu=4, disable_pr_bar=True)


def _demean(g):
    return g - g.mean()


def _zscore(g):
    return (g - g.mean()) / (g.std(ddof=0) + 1e-9)


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 5000
    return pd.DataFrame({
        'k': rng.integers(0, 50, n),
        'k2': rng.integers(0, 3, n),
        'x': rng.random(n),
        'y': rng.random(n) * 10,
    })


@pytest.mark.parametrize('executor', ['processes', 'threads'])
def test_dataframe_transform_callable(df, executor):
    expected = df.groupby('k')[['x', 'y']].transform(_demean)
    result = df.groupby('k')[['x', 'y']].p_transform(_demean, executor=executor)
    pd.testing.assert_frame_equal(result[expected.columns], expected)


def test_dataframe_transform_zscore(df):
    expected = df.groupby('k')[['x', 'y']].transform(_zscore)
    result = df.groupby('k')[['x', 'y']].p_transform(_zscore)
    pd.testing.assert_frame_equal(result[expected.columns], expected)


def test_series_transform_callable(df):
    expected = df.groupby('k')['x'].transform(_demean)
    result = df.groupby('k')['x'].p_transform(_demean)
    pd.testing.assert_series_equal(result, expected)


def test_scalar_broadcast(df):
    expected = df.groupby('k')['x'].transform(lambda s: s.mean())
    result = df.groupby('k')['x'].p_transform(lambda s: s.mean())
    pd.testing.assert_series_equal(result, expected)


def test_string_func(df):
    expected = df.groupby('k')[['x', 'y']].transform('mean')
    result = df.groupby('k')[['x', 'y']].p_transform('mean')
    pd.testing.assert_frame_equal(result[expected.columns], expected)


def test_multi_key(df):
    expected = df.groupby(['k', 'k2'])[['x', 'y']].transform(_demean)
    result = df.groupby(['k', 'k2'])[['x', 'y']].p_transform(_demean)
    pd.testing.assert_frame_equal(result[expected.columns], expected)


def test_nan_key_rows_stay_nan():
    df = pd.DataFrame({
        'k': ['a', 'b', 'a', 'b', np.nan, 'a'],
        'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    expected = df.groupby('k')['x'].transform(_demean)
    result = df.groupby('k')['x'].p_transform(_demean)
    pd.testing.assert_series_equal(result, expected)
    assert np.isnan(result.iloc[4])


def test_duplicate_index_labels():
    df = pd.DataFrame(
        {'k': ['a', 'b', 'a', 'b'], 'x': [1.0, 2.0, 3.0, 4.0]},
        index=[7, 7, 7, 7],
    )
    expected = df.groupby('k')['x'].transform(_demean)
    result = df.groupby('k')['x'].p_transform(_demean)
    pd.testing.assert_series_equal(result, expected)


def test_args_passed_through(df):
    def add_const(g, c):
        return g - g.mean() + c

    expected = df.groupby('k')['x'].transform(add_const, 100)
    result = df.groupby('k')['x'].p_transform(add_const, args=(100,))
    pd.testing.assert_series_equal(result, expected)
