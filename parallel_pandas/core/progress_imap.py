import atexit
import io
import logging
import time
from itertools import count

import multiprocessing as mp
from threading import Thread

from tqdm.auto import tqdm

from psutil import virtual_memory
from psutil._common import bytes2human


_MANAGER = None

# Cache of worker pools, keyed by (executor, n_cpu), so consecutive parallel
# operations reuse the same warm workers instead of paying the spawn cost every
# call. See ``_get_pool``/``set_reuse_pool``.
_POOLS = {}
_REUSE_POOL = True
_POOLS_ATEXIT_REGISTERED = False

# Where the progress bars write. ``None`` means the tqdm default (stderr).
# Can be any file-like object, e.g. a TqdmToLogger to redirect the bar to a logger.
_PROGRESS_FILE = None


def set_progress_bar_file(file):
    """Set the file-like object the progress bars write to (None = tqdm default)."""
    global _PROGRESS_FILE
    _PROGRESS_FILE = file


class TqdmToLogger(io.StringIO):
    """File-like object that redirects tqdm progress-bar output to a logging.Logger.

    tqdm writes the rendered bar and flushes on every refresh; each flush is emitted
    as one log record, so the bar shows up in the configured logging handlers
    (files, JSON logs, non-tty environments, ...) instead of the terminal.
    """

    def __init__(self, logger, level=logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level
        self._buf = ''

    def write(self, buf):
        self._buf = buf.strip('\r\n\t ')
        return len(buf)

    def flush(self):
        if self._buf:
            self.logger.log(self.level, self._buf)


def _shutdown_manager():
    global _MANAGER
    if _MANAGER is not None:
        _MANAGER.shutdown()
        _MANAGER = None


def _create_pool(executor, n_cpu, initializer, initargs):
    if executor == 'threads':
        return mp.pool.ThreadPool(n_cpu, initializer=initializer, initargs=initargs)
    return mp.Pool(n_cpu, initializer=initializer, initargs=initargs)


def _shutdown_pools():
    global _POOLS
    for pool in list(_POOLS.values()):
        try:
            pool.terminate()
        except Exception:
            pass
    _POOLS = {}


def _drop_pool(executor, n_cpu):
    """Discard a (possibly broken) cached pool so the next call spawns a fresh one."""
    pool = _POOLS.pop((executor, n_cpu), None)
    if pool is not None:
        try:
            pool.terminate()
        except Exception:
            pass


def set_reuse_pool(flag):
    """Enable/disable reusing worker pools across parallel operations.

    When enabled (default) the worker pool for a given ``(executor, n_cpu)`` is
    created once and kept warm for the lifetime of the process, which removes the
    per-call process-spawn overhead. Disabling it restores the old behaviour of
    creating and tearing down a pool on every call and closes any warm pools.
    """
    global _REUSE_POOL
    _REUSE_POOL = bool(flag)
    if not _REUSE_POOL:
        _shutdown_pools()


def _get_pool(executor, n_cpu, initializer, initargs):
    """Return ``(pool, reused)``.

    Only the common case (no custom ``initializer``) is cached, because a cached
    pool runs its initializer exactly once at creation and cannot honour a
    different one on a later call. ``reused=True`` means the caller must NOT close
    the pool afterwards.
    """
    global _POOLS_ATEXIT_REGISTERED
    if not _REUSE_POOL or initializer is not None:
        return _create_pool(executor, n_cpu, initializer, initargs), False
    key = (executor, n_cpu)
    pool = _POOLS.get(key)
    if pool is None:
        pool = _create_pool(executor, n_cpu, None, initargs)
        _POOLS[key] = pool
        if not _POOLS_ATEXIT_REGISTERED:
            atexit.register(_shutdown_pools)
            _POOLS_ATEXIT_REGISTERED = True
    return pool, True


def get_workers_queue():
    """Return a progress queue from a shared, lazily-created Manager.

    Spawning a ``multiprocessing.Manager`` starts a server process, which is
    costly. parallel-pandas runs many small operations in a row, so instead of
    creating a Manager on every call we create it once and hand out fresh
    queues from it.
    """
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = mp.Manager()
        atexit.register(_shutdown_manager)
    return _MANAGER.Queue()


class ProgressBar(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def close(self):
        super().close()
        if hasattr(self, 'disp'):
            if self.total and self.n < self.total:
                self.disp(bar_style='warning')


class MemoryProgressBar(tqdm):

    def refresh(self, **kwargs):
        super().refresh(**kwargs)
        if 70 <= self.n < 90:
            self.colour = 'orange'
        elif self.n >= 90:
            self.colour = 'red'
        else:
            self.colour = 'green'


class ProgressStatus:
    def __init__(self):
        self.next_update = 1
        self.last_update_t = time.perf_counter()
        self.last_update_val = 0


def progress_udf_wrapper(func, workers_queue, total):
    state = ProgressStatus()
    cnt = count(1)

    def wrapped_udf(*args, **kwargs):
        result = func(*args, **kwargs)
        updated = next(cnt)
        if updated == state.next_update:
            time_now = time.perf_counter()

            delta_t = time_now - state.last_update_t
            delta_i = updated - state.last_update_val

            state.next_update += max(int((delta_i / delta_t) * .25), 1)
            state.last_update_val = updated
            state.last_update_t = time_now
            workers_queue.put_nowait((1, delta_i))
        elif updated == total:
            workers_queue.put_nowait((1, updated - state.last_update_val))
        return result

    return wrapped_udf


def _process_status(bar_size, disable, show_vmem, desc, q):
    pbar_file = _PROGRESS_FILE
    bar = ProgressBar(total=bar_size, disable=disable, desc=desc + ' DONE', file=pbar_file)
    vmem = virtual_memory()
    if show_vmem:
        vmem_pbar = MemoryProgressBar(range(100),
                                      bar_format="{desc}: {percentage:.1f}%|{bar}|  " + bytes2human(vmem.total),
                                      initial=vmem.percent, colour='green', position=1, desc='VMEM USAGE',
                                      disable=disable, file=pbar_file,
                                      )
        vmem_pbar.refresh()
    while True:
        flag, upd_value = q.get()
        if not flag:
            bar.close()
            if show_vmem:
                vmem_pbar.close()
            return
        bar.update(upd_value)
        if show_vmem:
            if time.time() - vmem_pbar.last_print_t >= 1:
                vmem = virtual_memory()
                vmem_pbar.update(vmem.percent - vmem_pbar.n)
                vmem_pbar.refresh()


def _do_parallel(func, tasks, initializer, initargs, n_cpu, total, disable, show_vmem,
                 q, executor, desc):
    if not n_cpu:
        n_cpu = mp.cpu_count()
    thread_ = Thread(target=_process_status, args=(total, disable, show_vmem, desc, q))
    thread_.start()
    pool, reused = _get_pool(executor, n_cpu, initializer, initargs)
    try:
        result = list(pool.imap(func, tasks))
    except BaseException:
        # A reused pool may be left in a broken state (e.g. a worker died), so drop
        # it to guarantee the next call starts from a healthy pool.
        if reused:
            _drop_pool(executor, n_cpu)
        raise
    finally:
        if not reused:
            pool.close()
            pool.join()
    q.put((None, None))
    thread_.join()
    return result


def progress_imap(func, tasks, q, executor='threads', initializer=None, initargs=(), n_cpu=None, total=None,
                  disable=False, show_vmem=False, desc=''
                  ):
    if executor not in ['threads', 'processes']:
        raise ValueError('Invalid executor value specified. Must be one of the values: "threads", "processes"')
    try:
        result = _do_parallel(func, tasks, initializer, initargs, n_cpu, total, disable, show_vmem, q, executor, desc)
    except (KeyboardInterrupt, Exception):
        q.put((None, None))
        raise
    return result
