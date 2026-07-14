import numpy as np
import pandas as pd
import pytest

from parallel_pandas import ParallelPandas

ParallelPandas.initialize(n_cpu=4, disable_pr_bar=True)


@pytest.fixture
def s_str():
    rng = np.random.default_rng(0)
    words = np.array(['Alpha', 'beta', 'GAMMA', 'delta42', 'Eps-ilon', 'zeta '])
    return pd.Series(rng.choice(words, size=3000))


@pytest.fixture
def s_dt():
    return pd.Series(pd.date_range('2020-01-01', periods=3000, freq='7h', tz='UTC'))


@pytest.mark.parametrize('op,args,kwargs', [
    ('lower', (), {}),
    ('upper', (), {}),
    ('len', (), {}),
    ('strip', (), {}),
    ('contains', ('a',), {}),
    ('startswith', ('A',), {}),
    ('replace', ('a', 'X'), {}),
    ('slice', (0, 3), {}),
    ('count', ('a',), {}),
    ('zfill', (10,), {}),
])
def test_str_methods(s_str, op, args, kwargs):
    expected = getattr(s_str.str, op)(*args, **kwargs)
    result = getattr(s_str.parallel.str, op)(*args, **kwargs)
    pd.testing.assert_series_equal(result, expected)


def test_str_extract_returns_frame(s_str):
    expected = s_str.str.extract(r'([A-Za-z]+)(\d*)')
    result = s_str.parallel.str.extract(r'([A-Za-z]+)(\d*)')
    pd.testing.assert_frame_equal(result, expected)


def test_str_split_expand(s_str):
    expected = s_str.str.split('-', expand=True)
    result = s_str.parallel.str.split('-', expand=True)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize('prop', ['year', 'month', 'day', 'dayofweek', 'hour', 'quarter'])
def test_dt_properties(s_dt, prop):
    expected = getattr(s_dt.dt, prop)
    result = getattr(s_dt.parallel.dt, prop)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize('op,args,kwargs', [
    ('floor', ('D',), {}),
    ('ceil', ('h',), {}),
    ('strftime', ('%Y-%m-%d',), {}),
    ('tz_convert', ('Europe/Berlin',), {}),
    ('day_name', (), {}),
])
def test_dt_methods(s_dt, op, args, kwargs):
    expected = getattr(s_dt.dt, op)(*args, **kwargs)
    result = getattr(s_dt.parallel.dt, op)(*args, **kwargs)
    pd.testing.assert_series_equal(result, expected)


def test_str_with_nan():
    s = pd.Series(['aXa', None, 'bb', 'aa'] * 1000)
    expected = s.str.contains('a')
    result = s.parallel.str.contains('a')
    pd.testing.assert_series_equal(result, expected)


def test_small_series_falls_back(s_str):
    small = s_str.iloc[:2]
    # fewer rows than workers -> native path, still correct
    pd.testing.assert_series_equal(small.parallel.str.upper(), small.str.upper())


def test_str_on_dataframe_raises():
    df = pd.DataFrame({'a': ['x', 'y']})
    with pytest.raises(AttributeError):
        _ = df.parallel.str


def test_unknown_str_method_raises(s_str):
    with pytest.raises(AttributeError):
        _ = s_str.parallel.str.not_a_method
