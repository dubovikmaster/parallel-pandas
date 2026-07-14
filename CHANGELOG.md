# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project
follows a simple `MAJOR.MINOR` versioning scheme.

## 0.8

### Added
- **Parallel `groupby.transform`** — `DataFrameGroupBy.p_transform` /
  `SeriesGroupBy.p_transform`. Groups are chunked by their integer code and the
  real pandas `transform` runs on each chunk, reproducing pandas semantics
  exactly (column fast-path, broadcasting, string ops, NaN keys). ~5x faster
  than serial `transform` for heavy Python UDFs.
- **Parallel `groupby.agg`** — `DataFrameGroupBy.p_agg` / `SeriesGroupBy.p_agg`.
  Supports callable / string / list / dict / named aggregations and
  `as_index=False`, `sort=False`, MultiIndex keys. ~5x faster than serial `agg`
  for heavy aggfuncs.
- **Parallel `.str` and `.dt` accessors** on the Series `.parallel` namespace:
  `s.parallel.str.<op>()` and `s.parallel.dt.<op>()` (methods return a callable,
  accessor properties like `dt.year` are evaluated eagerly). Best for CPU-heavy
  per-element ops such as regex `extract`/`replace` and `strftime`.
- **`.parallel` accessor** — every `p_*` method is reachable as
  `df.parallel.<name>()` in addition to `df.p_<name>()`.
- **Automatic chunk sizing** — `split_factor` now defaults to `None`, which
  auto-picks the number of chunks from the input byte size (~8 MB per chunk) for
  the process-transport methods (`p_apply`, `p_map`, `p_applymap`,
  `chunk_apply`). Large frames parallelize noticeably better (~1.9x on a ~1.9 GB
  frame). Pass an explicit integer to keep the classic `split_factor * n_cpu`
  behaviour.
- **Warm worker pools** — `reuse_pool=True` (default) keeps the process/thread
  pool alive across calls, removing the per-call spawn overhead. Set
  `reuse_pool=False` to restore per-call pools.
- **Parallel `DataFrame.p_pivot_table`** (closes #3).
- **Progress-bar redirection to a logger** via `ParallelPandas.initialize(logger=...)`
  (and `pbar_file=...` for an arbitrary file-like), closes #15.

### Changed
- Support for **pandas 3.0**; the supported floor is now **pandas 2.0**, both
  covered by the CI version matrix.
- Internal refactors of the stat/window `do_parallel` dispatchers (no API
  change).

### Fixed
- User functions that reference module-level globals (e.g. `numpy`) now work
  under the `spawn` start method (closes #13).
- Various parallel window bugs, backed by a new test suite and GitHub Actions CI
  across a Python/pandas version matrix.
