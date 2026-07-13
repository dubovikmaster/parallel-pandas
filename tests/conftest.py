import numpy as np
import pandas as pd
import pytest

from parallel_pandas import ParallelPandas


@pytest.fixture(scope="session", autouse=True)
def _init_parallel_pandas():
    # Small, deterministic pool so tests are fast and reproducible.
    ParallelPandas.initialize(n_cpu=4, disable_pr_bar=True, split_factor=2)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def df(rng):
    return pd.DataFrame(
        rng.standard_normal((200, 12)),
        columns=[f"col_{i}" for i in range(12)],
    )


@pytest.fixture
def df_with_nans(rng):
    data = rng.standard_normal((200, 12))
    mask = rng.random((200, 12)) < 0.1
    data[mask] = np.nan
    return pd.DataFrame(data, columns=[f"col_{i}" for i in range(12)])


@pytest.fixture
def series(rng):
    return pd.Series(rng.standard_normal(500))


@pytest.fixture
def grouped_df(rng):
    n = 600
    return pd.DataFrame(
        {
            "group": rng.integers(0, 5, n),
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
        }
    )
