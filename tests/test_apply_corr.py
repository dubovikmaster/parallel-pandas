"""Parallel apply / agg / corr must match plain pandas."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal, assert_frame_equal


def _row_sum(row):
    return row.sum()


def _scale(x):
    return x * 2 + 1


@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_df_apply_axis0(df, executor):
    expected = df.apply(_row_sum, axis=0)
    actual = df.p_apply(_row_sum, axis=0, executor=executor)
    assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)


@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_df_apply_axis1(df, executor):
    expected = df.apply(_row_sum, axis=1)
    actual = df.p_apply(_row_sum, axis=1, executor=executor)
    assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)


def test_df_agg(df):
    expected = df.agg(["mean", "std", "min", "max"])
    actual = df.p_agg(["mean", "std", "min", "max"])
    assert_frame_equal(actual[expected.columns].loc[expected.index], expected)


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_corr(df, method):
    expected = df.corr(method=method)
    actual = df.p_corr(method=method)
    assert_frame_equal(actual.loc[expected.index, expected.columns], expected,
                       check_exact=False, atol=1e-8)


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_corr_with_nans(df_with_nans, method):
    expected = df_with_nans.corr(method=method)
    actual = df_with_nans.p_corr(method=method)
    assert_frame_equal(actual.loc[expected.index, expected.columns], expected,
                       check_exact=False, atol=1e-8)


def test_series_apply(series):
    expected = series.apply(_scale)
    actual = series.p_apply(_scale)
    assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)


def test_series_map(series):
    expected = series.map(_scale)
    actual = series.p_map(_scale)
    assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)


def test_series_isin(series):
    values = [round(v, 3) for v in series.head(5).tolist()]
    assert_series_equal(series.p_isin(values).sort_index(),
                        series.isin(values).sort_index(), check_names=False)
