"""Parallel rolling / expanding / ewm / groupby-window must match plain pandas."""
import pytest
from pandas.testing import assert_frame_equal


ROLL_OPS = ["mean", "sum", "std", "var", "min", "max", "median"]


@pytest.mark.parametrize("op", ROLL_OPS)
def test_rolling(df, op):
    expected = getattr(df.rolling(10), op)()
    actual = getattr(df.rolling(10), f"p_{op}")()
    assert_frame_equal(actual[expected.columns], expected)


@pytest.mark.parametrize("op", ROLL_OPS)
def test_expanding(df, op):
    expected = getattr(df.expanding(), op)()
    actual = getattr(df.expanding(), f"p_{op}")()
    assert_frame_equal(actual[expected.columns], expected)


@pytest.mark.parametrize("op", ["mean", "sum", "std", "var"])
def test_ewm(df, op):
    expected = getattr(df.ewm(span=5), op)()
    actual = getattr(df.ewm(span=5), f"p_{op}")()
    assert_frame_equal(actual[expected.columns], expected)


@pytest.mark.parametrize("op", ["mean", "sum", "std", "var"])
def test_rolling_groupby(grouped_df, op):
    gb = grouped_df.groupby("group")
    expected = getattr(gb.rolling(5), op)()
    actual = getattr(gb.rolling(5), f"p_{op}")()
    assert_frame_equal(actual.sort_index()[expected.columns], expected.sort_index())
