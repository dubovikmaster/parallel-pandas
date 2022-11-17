## Parallel-pandas
Makes it easy to parallelize your calculations in pandas on all your CPUs.

## Installation


## Quickstart
```python
import pandas as pd
import numpy as np
from parallel_pandas import ParallelPandas

#initialize parallel-pandas
ParallelPandas.initialize(n_cpu=16, split_factor=4, disable_pr_bar=True)

# create big DataFrame
df = pd.DataFrame(np.random.random((1_000_000, 100)))

# calculate multiple quantiles. Pandas only uses one core of CPU
%%timeit
res = df.quantile(q=[.25, .5, .95], axis=1)
```
3.66 s ± 31.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```python
#p_quantile is parallel analogue of quantile methods. Can use all cores of your CPU.
%%timeit
res = df.p_quantile(q=[.25, .5, .95], axis=1)
```
679 ms ± 10.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

As you can see the `p_quantile` method is 5 times faster!

## Usage

<div class="alert alert-block alert-warning">
<b>Warning:</b> You will only notice performance gains if your data is very large.
</div>
When initializing parallel-pandas you can specify the following options:

Under the hood, **parallel-pandas** works very simply. The Dataframe or Series is split into chunks along the first or second axis. Then these chunks are passed to a pool of processes or threads where the desired method is executed on each part. At the end, the parts are concatenated to get the final result.

1. `n_cpu` - the number of cores of your CPU that you want to use (default `None` - use all cores of CPU)
2. `split_factor` - Affects the number of chunks into which the DataFrame/Series is split according to the formula `chunks_number = split_factor*n_cpu` (default 1).
3. `show_vmem` - Shows a progress bar with available RAM (default `False`)
4. `disable_pr_bar` - Disable the progress bar for parallel tasks (default `False`)

For example

```python
import pandas as pd
import numpy as np
from parallel_pandas import ParallelPandas

#initialize parallel-pandas
ParallelPandas.initialize(n_cpu=16, split_factor=4, disable_pr_bar=True)

# create big DataFrame
df = pd.DataFrame(np.random.random((1_000_000, 100)))
```
### Parallel counterparts for pandas Dataframe methods

| methods       | parallel analogue | executor           |
|---------------|-------------------|--------------------|
| df.mean()     | df.p_mean()       | threads            |
| df.min()      | df.p_min()        | threads            |
| df.max()      | df.p_max()        | threads            |
| df.median()   | df.p_max()        | threads            |
| df.kurt()     | df.p_kurt()       | threads            |
| df.skew()     | df.p_skew()       | threads            |
| df.sum()      | df.p_sum()        | threads            |
| df.prod()     | df.p_prod()       | threads            |
| df.var()      | df.p_var()        | threads            |
| df.sem()      | df.p_sem()        | threads            |
| df.std()      | df.p_std()        | threads            |
| df.cummin()   | df.p_cummin()     | threads            |
| df.cumsum()   | df.p_cumsum()     | threads            |
| df.cummax()   | df.p_cummax()     | threads            |
| df.cumprod()  | df.p_cumprod()    | threads            |
| df.apply()    | df.p_apply()      | threads / processes|
| df.applymap() | df.p_applymap()   | processes          |
| df.replace()  | df.p_replace()    | threads            |
| df.describe() | df.p_describe()   | threads            |
| df.nunique()  | df.p_nunique()    | threads / processes|
| df.mad()      | df.p_mad()        | threads            |
| df.idxmin()   | df.p_idxmin()     | threads            |
| df.idxmax()   | df.p_idxmax()     | threads            |
| df.rank()     | df.p_rank()       | threads            |
| df.mode()     | df.p_mode()       | threads/processes  |

### Parallel counterparts for pandas DataframeGroupBy methods

| methods                  | parallel analogue          | executor             |
|--------------------------|----------------------------|----------------------|
| DataFrameGroupBy.apply() | DataFrameGroupBy.p_apply() | threads / processes  |

### Parallel counterparts for pandas window methods

#### Rolling

| methods                           | parallel analogue                   | executor    |
|-----------------------------------|-------------------------------------|-------------|
| pd.core.window.Rolling.apply()    | pd.core.window.Rolling.p_apply()    | processes   |
| pd.core.window.Rolling.min()      | pd.core.window.Rolling.p_min()      | processes   |
| pd.core.window.Rolling.max()      | pd.core.window.Rolling.p_max()      | processes   |
| pd.core.window.Rolling.mean()     | pd.core.window.Rolling.p_mean()     | processes   |
| pd.core.window.Rolling.sum()      | pd.core.window.Rolling.p_sum()      | processes   |
| pd.core.window.Rolling.var()      | pd.core.window.Rolling.p_var()      | processes   |
| pd.core.window.Rolling.sem()      | pd.core.window.Rolling.p_sem()      | processes   |
| pd.core.window.Rolling.skew()     | pd.core.window.Rolling.p_skew()     | processes   |
| pd.core.window.Rolling.kurt()     | pd.core.window.Rolling.p_kurt()     | processes   |
| pd.core.window.Rolling.median()   | pd.core.window.Rolling.p_median()   | processes   |
| pd.core.window.Rolling.quantile() | pd.core.window.Rolling.p_quantile() | processes   |
| pd.core.window.Rolling.rank()     | pd.core.window.Rolling.p_rank()     | processes   |

#### RollingGroupby

| methods                                  | parallel analogue                          | executor     |
|------------------------------------------|--------------------------------------------|--------------|
| pd.core.window.RollingGroupby.apply()    | pd.core.window.RollingGroupby.p_apply()    | processes    |
| pd.core.window.RollingGroupby.min()      | pd.core.window.RollingGroupby.p_min()      | processes    |
| pd.core.window.RollingGroupby.max()      | pd.core.window.RollingGroupby.p_max()      | processes    |
| pd.core.window.RollingGroupby.mean()     | pd.core.window.RollingGroupby.p_mean()     | processes    |
| pd.core.window.RollingGroupby.sum()      | pd.core.window.RollingGroupby.p_sum()      | processes    |
| pd.core.window.RollingGroupby.var()      | pd.core.window.RollingGroupby.p_var()      | processes    |
| pd.core.window.RollingGroupby.sem()      | pd.core.window.RollingGroupby.p_sem()      | processes    |
| pd.core.window.RollingGroupby.skew()     | pd.core.window.RollingGroupby.p_skew()     | processes    |
| pd.core.window.RollingGroupby.kurt()     | pd.core.window.RollingGroupby.p_kurt()     | processes    |
| pd.core.window.RollingGroupby.median()   | pd.core.window.RollingGroupby.p_median()   | processes    |
| pd.core.window.RollingGroupby.quantile() | pd.core.window.RollingGroupby.p_quantile() | processes    |
| pd.core.window.RollingGroupby.rank()     | pd.core.window.RollingGroupby.p_rank()     | processes    |

#### Expanding

| methods                             | parallel analogue                     | executor    |
|-------------------------------------|---------------------------------------|-------------|
| pd.core.window.Expanding.apply()    | pd.core.window.Expanding.p_apply()    | processes   |
| pd.core.window.Expanding.min()      | pd.core.window.Expanding.p_min()      | processes   |
| pd.core.window.Expanding.max()      | pd.core.window.Expanding.p_max()      | processes   |
| pd.core.window.Expanding.mean()     | pd.core.window.Expanding.p_mean()     | processes   |
| pd.core.window.Expanding.sum()      | pd.core.window.Expanding.p_sum()      | processes   |
| pd.core.window.Expanding.var()      | pd.core.window.Expanding.p_var()      | processes   |
| pd.core.window.Expanding.sem()      | pd.core.window.Expanding.p_sem()      | processes   |
| pd.core.window.Expanding.skew()     | pd.core.window.Expanding.p_skew()     | processes   |
| pd.core.window.Expanding.kurt()     | pd.core.window.Expanding.p_kurt()     | processes   |
| pd.core.window.Expanding.median()   | pd.core.window.Expanding.p_median()   | processes   |
| pd.core.window.Expanding.quantile() | pd.core.window.Expanding.p_quantile() | processes   |
| pd.core.window.Expanding.rank()     | pd.core.window.Expanding.p_rank()     | processes   |


#### ExpandingGroupby

| methods                                    | parallel analogue                            | executor    |
|--------------------------------------------|----------------------------------------------|-------------|
| pd.core.window.ExpandingGroupby.apply()    | pd.core.window.ExpandingGroupby.p_apply()    | processes   |
| pd.core.window.ExpandingGroupby.min()      | pd.core.window.ExpandingGroupby.p_min()      | processes   |
| pd.core.window.ExpandingGroupby.max()      | pd.core.window.ExpandingGroupby.p_max()      | processes   |
| pd.core.window.ExpandingGroupby.mean()     | pd.core.window.ExpandingGroupby.p_mean()     | processes   |
| pd.core.window.ExpandingGroupby.sum()      | pd.core.window.ExpandingGroupby.p_sum()      | processes   |
| pd.core.window.ExpandingGroupby.var()      | pd.core.window.ExpandingGroupby.p_var()      | processes   |
| pd.core.window.ExpandingGroupby.sem()      | pd.core.window.ExpandingGroupby.p_sem()      | processes   |
| pd.core.window.ExpandingGroupby.skew()     | pd.core.window.ExpandingGroupby.p_skew()     | processes   |
| pd.core.window.ExpandingGroupby.kurt()     | pd.core.window.ExpandingGroupby.p_kurt()     | processes   |
| pd.core.window.ExpandingGroupby.median()   | pd.core.window.ExpandingGroupby.p_median()   | processes   |
| pd.core.window.ExpandingGroupby.quantile() | pd.core.window.ExpandingGroupby.p_quantile() | processes   |
| pd.core.window.ExpandingGroupby.rank()     | pd.core.window.ExpandingGroupby.p_rank()     | processes   |

### ExponentialMovingWindow

| methods                                       | parallel analogue                               | executor    |
|-----------------------------------------------|-------------------------------------------------|-------------|
| pd.core.window.ExponentialMovingWindow.mean() | pd.core.window.ExponentialMovingWindow.p_mean() | processes   |
| pd.core.window.ExponentialMovingWindow.sum()  | pd.core.window.ExponentialMovingWindow.p_sum()  | processes   |
| pd.core.window.ExponentialMovingWindow.var()  | pd.core.window.ExponentialMovingWindow.p_var()  | processes   |
| pd.core.window.ExponentialMovingWindow.std()  | pd.core.window.ExponentialMovingWindow.p_std()  | processes   |

### ExponentialMovingWindowGroupby

| methods                                              | parallel analogue                                      | executor    |
|------------------------------------------------------|--------------------------------------------------------|-------------|
| pd.core.window.ExponentialMovingWindowGroupby.mean() | pd.core.window.ExponentialMovingWindowGroupby.p_mean() | processes   |
| pd.core.window.ExponentialMovingWindowGroupby.sum()  | pd.core.window.ExponentialMovingWindowGroupby.p_sum()  | processes   |
| pd.core.window.ExponentialMovingWindowGroupby.var()  | pd.core.window.ExponentialMovingWindowGroupby.p_var()  | processes   |
| pd.core.window.ExponentialMovingWindowGroupby.std()  | pd.core.window.ExponentialMovingWindowGroupby.p_std()  | processes   |











