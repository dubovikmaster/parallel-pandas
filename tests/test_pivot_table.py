import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def pivot_df(rng):
    n = 3000
    return pd.DataFrame(
        {
            "A": rng.integers(0, 8, n),
            "B": rng.choice(list("xyz"), n),
            "C": rng.choice(["p", "q", "r", "s"], n),
            "v1": rng.standard_normal(n),
            "v2": rng.standard_normal(n) * 10,
        }
    )


@pytest.mark.parametrize("aggfunc", ["mean", "sum", "count", "max", "min", "std"])
def test_single_value_single_column(pivot_df, aggfunc):
    expected = pivot_df.pivot_table(index="A", columns="B", values="v1", aggfunc=aggfunc)
    actual = pivot_df.p_pivot_table(index="A", columns="B", values="v1", aggfunc=aggfunc)
    pd.testing.assert_frame_equal(actual, expected)


def test_no_columns(pivot_df):
    expected = pivot_df.pivot_table(index="A", values="v1", aggfunc="mean")
    actual = pivot_df.p_pivot_table(index="A", values="v1", aggfunc="mean")
    pd.testing.assert_frame_equal(actual, expected)


def test_multi_index(pivot_df):
    expected = pivot_df.pivot_table(index=["A", "B"], columns="C", values="v1", aggfunc="sum")
    actual = pivot_df.p_pivot_table(index=["A", "B"], columns="C", values="v1", aggfunc="sum")
    pd.testing.assert_frame_equal(actual, expected)


def test_multiple_values(pivot_df):
    expected = pivot_df.pivot_table(index="A", columns="B", values=["v1", "v2"], aggfunc="mean")
    actual = pivot_df.p_pivot_table(index="A", columns="B", values=["v1", "v2"], aggfunc="mean")
    pd.testing.assert_frame_equal(actual, expected, check_like=True)


def test_list_aggfunc(pivot_df):
    expected = pivot_df.pivot_table(index="A", columns="B", values="v1", aggfunc=["mean", "sum"])
    actual = pivot_df.p_pivot_table(index="A", columns="B", values="v1", aggfunc=["mean", "sum"])
    pd.testing.assert_frame_equal(actual, expected, check_like=True)


def test_dict_aggfunc(pivot_df):
    expected = pivot_df.pivot_table(index="A", values=["v1", "v2"], aggfunc={"v1": "mean", "v2": "sum"})
    actual = pivot_df.p_pivot_table(index="A", values=["v1", "v2"], aggfunc={"v1": "mean", "v2": "sum"})
    pd.testing.assert_frame_equal(actual, expected, check_like=True)


def test_fill_value(pivot_df):
    expected = pivot_df.pivot_table(index="A", columns="C", values="v1", aggfunc="sum", fill_value=0)
    actual = pivot_df.p_pivot_table(index="A", columns="C", values="v1", aggfunc="sum", fill_value=0)
    pd.testing.assert_frame_equal(actual, expected)


def test_callable_aggfunc(pivot_df):
    peak_to_peak = lambda x: x.max() - x.min()
    expected = pivot_df.pivot_table(index="A", columns="B", values="v1", aggfunc=peak_to_peak)
    actual = pivot_df.p_pivot_table(index="A", columns="B", values="v1", aggfunc=peak_to_peak)
    pd.testing.assert_frame_equal(actual, expected, check_like=True)


def test_threads_executor(pivot_df):
    expected = pivot_df.pivot_table(index="A", columns="B", values="v1", aggfunc="mean")
    actual = pivot_df.p_pivot_table(index="A", columns="B", values="v1", aggfunc="mean", executor="threads")
    pd.testing.assert_frame_equal(actual, expected)


def test_with_nans(pivot_df, rng):
    d = pivot_df.copy()
    d.loc[rng.random(len(d)) < 0.1, "v1"] = np.nan
    expected = d.pivot_table(index="A", columns="B", values="v1", aggfunc="mean")
    actual = d.p_pivot_table(index="A", columns="B", values="v1", aggfunc="mean")
    pd.testing.assert_frame_equal(actual, expected)


def test_fallback_margins(pivot_df):
    expected = pivot_df.pivot_table(index="A", columns="B", values="v1", aggfunc="sum", margins=True)
    actual = pivot_df.p_pivot_table(index="A", columns="B", values="v1", aggfunc="sum", margins=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_fallback_index_none(pivot_df):
    expected = pivot_df.pivot_table(columns="B", values="v1", aggfunc="mean")
    actual = pivot_df.p_pivot_table(columns="B", values="v1", aggfunc="mean")
    pd.testing.assert_frame_equal(actual, expected)


def test_sort_false(pivot_df):
    expected = pivot_df.pivot_table(index="A", columns="B", values="v1", aggfunc="mean", sort=False)
    actual = pivot_df.p_pivot_table(index="A", columns="B", values="v1", aggfunc="mean", sort=False)
    pd.testing.assert_frame_equal(actual, expected, check_like=True)
