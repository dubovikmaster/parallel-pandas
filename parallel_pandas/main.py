import pandas as pd

from .core import parallelize_apply
from .core import parallelize_replace
from .core import ParallelizeStatFunc
from .core import ParallelizeStatFuncDdof
from .core import parallelize_groupby_apply
from .core import parallelize_applymap
from .core import parallelize_describe
from .core import parallelize_nunique
from .core import parallelize_mad
from .core import parallelize_idxmax
from .core import parallelize_idxmin
from .core import parallelize_rank
from .core import ParallelizeMinCountStatFunc
from .core import ParallelizeAccumFunc
from .core import parallelize_quantile
from .core import parallelize_mode
# from .core import parallelize_pct_change


class ParallelPandas:
    @staticmethod
    def initialize(n_cpu=None, disable_pr_bar=False, show_vmem=False, split_factor=1):
        # add parallel methods to Series
        # pd.Series.parallel_apply = parallelize_apply(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
        #                                              set_error_value=set_error_value, error_behavior=error_behavior,
        #                                              show_vmem=show_vmem)

        # add parallel methods to DataFrame
        pd.DataFrame.p_apply = parallelize_apply(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar, show_vmem=show_vmem,
                                                 split_factor=split_factor)

        pd.DataFrame.p_replace = parallelize_replace(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                     show_vmem=show_vmem, split_factor=split_factor)

        pd.DataFrame.p_min = ParallelizeStatFunc(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                 show_vmem=show_vmem, split_factor=split_factor).do_parallel('min')

        pd.DataFrame.p_max = ParallelizeStatFunc(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                 show_vmem=show_vmem, split_factor=split_factor).do_parallel('max')

        pd.DataFrame.p_mean = ParallelizeStatFunc(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                  show_vmem=show_vmem, split_factor=split_factor).do_parallel('mean')
        pd.DataFrame.p_median = ParallelizeStatFunc(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                    show_vmem=show_vmem,
                                                    split_factor=split_factor).do_parallel('median')
        pd.DataFrame.p_skew = ParallelizeStatFunc(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                  show_vmem=show_vmem, split_factor=split_factor).do_parallel('skew')

        pd.DataFrame.p_kurt = ParallelizeStatFunc(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                  show_vmem=show_vmem, split_factor=split_factor).do_parallel('kurt')

        pd.DataFrame.p_std = ParallelizeStatFuncDdof(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                     show_vmem=show_vmem, split_factor=split_factor).do_parallel('std')

        pd.DataFrame.p_var = ParallelizeStatFuncDdof(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                     show_vmem=show_vmem, split_factor=split_factor).do_parallel('var')

        pd.DataFrame.p_sem = ParallelizeStatFuncDdof(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                     show_vmem=show_vmem, split_factor=split_factor).do_parallel('sem')

        pd.DataFrame.p_sum = ParallelizeMinCountStatFunc(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                         show_vmem=show_vmem,
                                                         split_factor=split_factor).do_parallel('sum')

        pd.DataFrame.p_prod = ParallelizeMinCountStatFunc(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                          show_vmem=show_vmem,
                                                          split_factor=split_factor).do_parallel('prod')

        pd.DataFrame.p_cumprod = ParallelizeAccumFunc(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                      show_vmem=show_vmem,
                                                      split_factor=split_factor).do_parallel('cumprod')

        pd.DataFrame.p_cummin = ParallelizeAccumFunc(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                     show_vmem=show_vmem,
                                                     split_factor=split_factor).do_parallel('cummin')

        pd.DataFrame.p_cummax = ParallelizeAccumFunc(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                     show_vmem=show_vmem,
                                                     split_factor=split_factor).do_parallel('cummax')

        pd.DataFrame.p_cumsum = ParallelizeAccumFunc(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                     show_vmem=show_vmem,
                                                     split_factor=split_factor).do_parallel('cumsum')

        pd.DataFrame.p_applymap = parallelize_applymap(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                       show_vmem=show_vmem)
        pd.DataFrame.p_describe = parallelize_describe(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                       show_vmem=show_vmem,
                                                       split_factor=split_factor)

        pd.DataFrame.p_nunique = parallelize_nunique(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
                                                     show_vmem=show_vmem,
                                                     split_factor=split_factor)

        pd.DataFrame.p_mad = parallelize_mad(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar, show_vmem=show_vmem,
                                             split_factor=split_factor)

        pd.DataFrame.p_idxmax = parallelize_idxmax(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar, show_vmem=show_vmem,
                                                   split_factor=split_factor)

        pd.DataFrame.p_idxmin = parallelize_idxmin(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar, show_vmem=show_vmem,
                                                   split_factor=split_factor)

        pd.DataFrame.p_rank = parallelize_rank(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar, show_vmem=show_vmem,
                                               split_factor=split_factor)

        pd.DataFrame.p_quantile = parallelize_quantile(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar, show_vmem=show_vmem,
                                                       split_factor=split_factor)

        pd.DataFrame.p_mode = parallelize_mode(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar, show_vmem=show_vmem,
                                               split_factor=split_factor)

        # pd.DataFrame.p_pct_change = parallelize_pct_change(n_cpu=n_cpu, disable_pr_bar=disable_pr_bar,
        #                                                    show_vmem=show_vmem, split_factor=split_factor)

        # add parallel methods to DataFrameGroupBy and SeriesGroupBy
        pd.core.groupby.DataFrameGroupBy.p_apply = parallelize_groupby_apply(n_cpu=n_cpu,
                                                                             disable_pr_bar=disable_pr_bar)
        pd.core.groupby.SeriesGroupBy.p_apply = parallelize_groupby_apply(n_cpu=n_cpu,
                                                                          disable_pr_bar=disable_pr_bar)
