"""Parallel groupby apply must match plain pandas."""
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

# `include_groups` was added to GroupBy.apply in pandas 2.2.
_PANDAS = tuple(int(x) for x in pd.__version__.split(".")[:2])
_HAS_INCLUDE_GROUPS = _PANDAS >= (2, 2)


def _demean(g):
    return g[["a", "b"]] - g[["a", "b"]].mean()


def _agg(g):
    return g.sum()


def test_groupby_apply_frame(grouped_df):
    gb = grouped_df.groupby("group")
    kw = {"include_groups": False} if _HAS_INCLUDE_GROUPS else {}
    expected = gb.apply(_demean, **kw)
    actual = gb.p_apply(_demean, include_groups=False)
    assert_frame_equal(actual.sort_index(), expected.sort_index())


def test_groupby_apply_series(grouped_df):
    gb = grouped_df.groupby("group")["a"]
    expected = gb.apply(_agg)
    actual = gb.p_apply(_agg)
    assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)
