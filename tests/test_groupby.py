"""Parallel groupby apply must match plain pandas."""
from pandas.testing import assert_frame_equal, assert_series_equal


def _demean(g):
    return g[["a", "b"]] - g[["a", "b"]].mean()


def _agg(g):
    return g.sum()


def test_groupby_apply_frame(grouped_df):
    gb = grouped_df.groupby("group")
    expected = gb.apply(_demean, include_groups=False)
    actual = gb.p_apply(_demean, include_groups=False)
    assert_frame_equal(actual.sort_index(), expected.sort_index())


def test_groupby_apply_series(grouped_df):
    gb = grouped_df.groupby("group")["a"]
    expected = gb.apply(_agg)
    actual = gb.p_apply(_agg)
    assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)
