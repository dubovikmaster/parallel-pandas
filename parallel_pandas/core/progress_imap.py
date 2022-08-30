import time
from functools import partial
from itertools import count

import multiprocessing as mp
from threading import Thread

from tqdm.auto import tqdm

from .tools import _wrapped_func

from psutil import virtual_memory
from psutil._common import bytes2human


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
        self.last_update_t = time.monotonic()
        self.last_update_val = 0


def progress_udf_wrapper(func, workers_queue, total):
    state = ProgressStatus()
    cnt = count(1)

    def wrapped_udf(*args, **kwargs):
        result = func(*args, **kwargs)
        updated = next(cnt)
        if updated == state.next_update:
            time_now = time.monotonic()

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
    bar = ProgressBar(total=bar_size, disable=disable, desc=desc + ' DONE')
    vmem = virtual_memory()
    if show_vmem:
        vmem_pbar = MemoryProgressBar(range(100),
                                      bar_format="{desc}: {percentage:.1f}%|{bar}|  " + bytes2human(vmem.total),
                                      initial=vmem.percent, colour='green', position=1, desc='VMEM USAGE',
                                      disable=disable,
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
    if executor == 'threads':
        exc_pool = mp.pool.ThreadPool(n_cpu, initializer=initializer, initargs=initargs)
    else:
        exc_pool = mp.Pool(n_cpu, initializer=initializer, initargs=initargs)
    with exc_pool as p:
        result = list()
        iter_result = p.imap(func, tasks)
        while 1:
            try:
                result.append(next(iter_result))
            except StopIteration:
                break
    q.put((None, None))
    thread_.join()
    return result


def progress_imap(func, tasks, q, executor='threads', initializer=None, initargs=(), n_cpu=None, total=None,
                  disable=False, process_timeout=None, show_vmem=False, desc=''
                  ):
    if executor not in ['threads', 'processes']:
        raise ValueError('Invalid executor value specified. Must be one of the values: "threads", "processes"')
    try:
        if process_timeout:
            func = partial(_wrapped_func, func, process_timeout, True)
        result = _do_parallel(func, tasks, initializer, initargs, n_cpu, total, disable, show_vmem, q, executor, desc)
    except (KeyboardInterrupt, Exception):
        q.put((None, None))
        raise
    return result
