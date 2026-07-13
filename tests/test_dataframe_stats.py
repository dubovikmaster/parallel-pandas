"""Parallel DataFrame reductions must match plain pandas."""
import pandas as pd
import pytest
from pandas.testing import assert_series_equal, assert_frame_equal


REDUCTIONS = ["mean", "median", "min", "max", "sum", "prod", "std", "var", "sem", "skew", "kurt"]


@pytest.mark.parametrize("op", REDUCTIONS)
def test_reduction_matches_pandas(df, op):
    expected = getattr(df, op)()
    actual = getattr(df, f"p_{op}")()
    assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)


@pytest.mark.parametrize("op", REDUCTIONS)
def test_reduction_with_nans(df_with_nans, op):
    expected = getattr(df_with_nans, op)()
    actual = getattr(df_with_nans, f"p_{op}")()
    assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)


CUMULATIVE = ["cumsum", "cumprod", "cummin", "cummax"]


@pytest.mark.parametrize("op", CUMULATIVE)
def test_cumulative_axis0_matches_pandas(df, op):
    expected = getattr(df, op)(axis=0)
    actual = getattr(df, f"p_{op}")(axis=0)
    assert_frame_equal(actual[expected.columns], expected)


@pytest.mark.parametrize("op", CUMULATIVE)
def test_cumulative_axis1_matches_pandas(df, op):
    expected = getattr(df, op)(axis=1)
    actual = getattr(df, f"p_{op}")(axis=1)
    assert_frame_equal(actual[expected.columns], expected)


def test_nunique(df_with_nans):
    assert_series_equal(df_with_nans.p_nunique().sort_index(),
                        df_with_nans.nunique().sort_index(), check_names=False)


def test_idxmax_idxmin(df):
    assert_series_equal(df.p_idxmax().sort_index(), df.idxmax().sort_index(), check_names=False)
    assert_series_equal(df.p_idxmin().sort_index(), df.idxmin().sort_index(), check_names=False)


def test_rank(df):
    assert_frame_equal(df.p_rank()[df.columns], df.rank())


@pytest.mark.parametrize("q", [0.5, 0.9, [0.1, 0.5, 0.9]])
def test_quantile(df, q):
    expected = df.quantile(q=q)
    actual = df.p_quantile(q=q)
    if isinstance(expected, pd.Series):
        assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)
    else:
        assert_frame_equal(actual[expected.columns], expected)


def test_describe(df):
    expected = df.describe()
    actual = df.p_describe()
    assert_frame_equal(actual[expected.columns].loc[expected.index], expected)


def test_isin(df):
    values = [0, 1]
    assert_frame_equal(df.p_isin(values)[df.columns], df.isin(values))
