import numpy as np
import pandas as pd
import pytest

from parallel_pandas import ParallelPandas

ParallelPandas.initialize(n_cpu=4, disable_pr_bar=True)


def _range(s):
    return s.max() - s.min()


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 5000
    return pd.DataFrame({
        'k': rng.integers(0, 40, n),
        'k2': rng.integers(0, 3, n),
        'x': rng.random(n),
        'y': rng.random(n) * 10,
    })


@pytest.mark.parametrize('executor', ['processes', 'threads'])
def test_agg_callable(df, executor):
    expected = df.groupby('k').agg(_range)
    result = df.groupby('k').p_agg(_range, executor=executor)
    pd.testing.assert_frame_equal(result, expected)


def test_agg_string(df):
    expected = df.groupby('k').agg('mean')
    result = df.groupby('k').p_agg('mean')
    pd.testing.assert_frame_equal(result, expected)


def test_agg_list(df):
    expected = df.groupby('k').agg(['mean', 'sum'])
    result = df.groupby('k').p_agg(['mean', 'sum'])
    pd.testing.assert_frame_equal(result, expected)


def test_agg_dict(df):
    expected = df.groupby('k').agg({'x': 'mean', 'y': _range})
    result = df.groupby('k').p_agg({'x': 'mean', 'y': _range})
    pd.testing.assert_frame_equal(result, expected)


def test_agg_named(df):
    expected = df.groupby('k').agg(mx=('x', 'max'), sy=('y', 'sum'))
    result = df.groupby('k').p_agg(mx=('x', 'max'), sy=('y', 'sum'))
    pd.testing.assert_frame_equal(result, expected)


def test_agg_multikey(df):
    expected = df.groupby(['k', 'k2']).agg(_range)
    result = df.groupby(['k', 'k2']).p_agg(_range)
    pd.testing.assert_frame_equal(result, expected)


def test_agg_multikey_as_index_false(df):
    expected = df.groupby(['k', 'k2'], as_index=False).agg(_range)
    result = df.groupby(['k', 'k2'], as_index=False).p_agg(_range)
    pd.testing.assert_frame_equal(result, expected)


def test_agg_as_index_false(df):
    expected = df.groupby('k', as_index=False).agg(_range)
    result = df.groupby('k', as_index=False).p_agg(_range)
    pd.testing.assert_frame_equal(result, expected)


def test_agg_sort_false(df):
    expected = df.groupby('k', sort=False).agg(_range)
    result = df.groupby('k', sort=False).p_agg(_range)
    pd.testing.assert_frame_equal(result, expected)


def test_agg_series(df):
    expected = df.groupby('k')['x'].agg(_range)
    result = df.groupby('k')['x'].p_agg(_range)
    pd.testing.assert_series_equal(result, expected)


def test_agg_nan_key():
    df = pd.DataFrame({
        'k': ['a', 'b', 'a', 'b', np.nan, 'a'],
        'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    expected = df.groupby('k').agg(_range)
    result = df.groupby('k').p_agg(_range)
    pd.testing.assert_frame_equal(result, expected)


def test_agg_args_passed_through(df):
    def clipped_sum(s, lo):
        return s.clip(lower=lo).sum()

    expected = df.groupby('k')['x'].agg(clipped_sum, 0.5)
    result = df.groupby('k')['x'].p_agg(clipped_sum, args=(0.5,))
    pd.testing.assert_series_equal(result, expected)
