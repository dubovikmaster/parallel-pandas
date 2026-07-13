"""The .parallel accessor must dispatch to the p_* methods."""
import pytest
from pandas.testing import assert_series_equal, assert_frame_equal


def test_dataframe_accessor_matches_p_method(df):
    assert_series_equal(df.parallel.mean().sort_index(),
                        df.p_mean().sort_index(), check_names=False)


def test_dataframe_accessor_matches_pandas(df):
    assert_series_equal(df.parallel.sum().sort_index(),
                        df.sum().sort_index(), check_names=False)


def test_accessor_apply_passes_args(df):
    expected = df.apply(lambda row: row.sum(), axis=1)
    actual = df.parallel.apply(lambda row: row.sum(), axis=1, executor="processes")
    assert_series_equal(actual, expected)


def test_series_accessor(series):
    expected = series.apply(lambda x: x + 1)
    actual = series.parallel.apply(lambda x: x + 1, executor="processes")
    assert_series_equal(actual, expected)


def test_accessor_chunk_apply_alias(df):
    result = df.parallel.chunk_apply(lambda d: d + 1, executor="processes")
    expected = df.chunk_apply(lambda d: d + 1, executor="processes")
    assert_frame_equal(result[df.columns], expected[df.columns])


def test_unknown_method_raises(df):
    with pytest.raises(AttributeError, match="no method 'definitely_not_a_method'"):
        df.parallel.definitely_not_a_method()


def test_dir_lists_parallel_methods(df):
    listing = dir(df.parallel)
    assert "mean" in listing
    assert "apply" in listing
    assert "chunk_apply" in listing


def test_accessor_is_same_object_type(df):
    from parallel_pandas.core.accessor import ParallelAccessor
    assert isinstance(df.parallel, ParallelAccessor)
