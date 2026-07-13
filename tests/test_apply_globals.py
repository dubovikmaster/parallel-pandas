import subprocess
import sys
import textwrap


# Regression test for https://github.com/dubovikmaster/parallel-pandas/issues/13
# A user function passed to p_apply references a third-party module (numpy) via a
# module global. Under the "spawn" start method the function is shipped to worker
# processes; without dill's recurse=True the referenced global is lost and the
# worker raises `NameError: name 'np' is not defined`.
#
# The failure only shows up for a function living in a __main__ script (an importable
# test module would make the global available in the workers anyway), so we run a real
# script in a subprocess and assert it completes.
_SCRIPT = textwrap.dedent(
    '''
    import numpy as np
    import pandas as pd
    from parallel_pandas import ParallelPandas


    def rolling_udf(x):
        return np.nansum(x) - np.nanmean(x)


    def row_udf(row):
        return np.nanmax(row) - np.nanmin(row)


    if __name__ == "__main__":
        ParallelPandas.initialize(n_cpu=2, disable_pr_bar=True)
        df = pd.DataFrame(np.random.random((500, 4)))

        r1 = df.rolling(10).p_apply(rolling_udf, raw=True, executor="processes")
        assert r1.shape == (500, 4)

        r2 = df.p_apply(row_udf, axis=1, executor="processes")
        assert len(r2) == 500

        s = pd.Series(np.random.random(500))
        r3 = s.p_apply(lambda v: np.expm1(v), executor="processes")
        assert len(r3) == 500

        print("SUCCESS")
    '''
)


def test_apply_with_third_party_global(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(_SCRIPT)
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SUCCESS" in proc.stdout
