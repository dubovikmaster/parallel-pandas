import logging

import pandas as pd
import pytest

from parallel_pandas import ParallelPandas, TqdmToLogger
from parallel_pandas.core import progress_imap


@pytest.fixture
def restore_pp():
    """Restore the shared configuration the rest of the suite relies on."""
    yield
    ParallelPandas.initialize(n_cpu=4, disable_pr_bar=True, split_factor=2)


class _CollectHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_tqdm_to_logger_write_flush():
    logger = logging.getLogger("parallel_pandas.test.direct")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = _CollectHandler()
    logger.addHandler(handler)
    try:
        out = TqdmToLogger(logger)
        out.write("\r  50%|#####     | progress")
        out.flush()
        out.write("\r\n")  # whitespace only -> nothing logged
        out.flush()
    finally:
        logger.removeHandler(handler)
    assert handler.messages == ["50%|#####     | progress"]


def test_progress_redirected_to_logger(rng, restore_pp):
    logger = logging.getLogger("parallel_pandas.test.pbar")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = _CollectHandler()
    logger.addHandler(handler)
    try:
        ParallelPandas.initialize(n_cpu=2, disable_pr_bar=False, split_factor=2, logger=logger)
        df = pd.DataFrame(rng.standard_normal((300, 8)))
        df.p_apply(lambda x: x.mean())
    finally:
        logger.removeHandler(handler)

    assert handler.messages, "expected the progress bar to emit at least one log record"
    assert any("DONE" in m for m in handler.messages)


def test_default_output_when_no_logger(rng, restore_pp):
    ParallelPandas.initialize(n_cpu=2, disable_pr_bar=True, split_factor=2)
    assert progress_imap._PROGRESS_FILE is None
    df = pd.DataFrame(rng.standard_normal((120, 4)))
    result = df.p_apply(lambda x: x.sum())
    assert len(result) == 4


def test_pbar_file_takes_precedence_over_logger(rng, restore_pp):
    import io

    buffer = io.StringIO()
    logger = logging.getLogger("parallel_pandas.test.precedence")
    ParallelPandas.initialize(n_cpu=2, disable_pr_bar=True, split_factor=2,
                              logger=logger, pbar_file=buffer)
    assert progress_imap._PROGRESS_FILE is buffer
