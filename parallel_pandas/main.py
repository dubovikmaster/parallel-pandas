import pandas as pd

from .core import parallelize_split
from .core import parallelize_apply
from .core import parallelize_groupby_apply


class ParallelPandas:
    @staticmethod
    def initialize(n_cpu=None, disable_pr_bar=False, error_behavior='raise', set_error_value=None, initialize=None,
                   initargs=(), process_timeout=None):
        # add parallel methods to Series
        pd.Series.parallel_apply = parallelize_apply(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                     set_error_value=set_error_value, error_behavior=error_behavior)

        # add parallel methods to DataFrame
        pd.DataFrame.parallel_apply = parallelize_apply(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                        set_error_value=set_error_value, error_behavior=error_behavior)
        pd.DataFrame.split_apply = parallelize_split(n_cpu=n_cpu, disable_progress_bar=disable_pr_bar,
                                                     set_error_value=set_error_value, error_behavior=error_behavior)

        # add parallel methods to DataFrameGroupBy and SeriesGroupBy
        pd.core.groupby.DataFrameGroupBy.parallel_apply = parallelize_groupby_apply(n_cpu=n_cpu,
                                                                                    disable_pr_bar=disable_pr_bar)
        pd.core.groupby.SeriesGroupBy.parallel_apply = parallelize_groupby_apply(n_cpu=n_cpu,
                                                                                 disable_pr_bar=disable_pr_bar)
