import time
from functools import partial
from itertools import count

import multiprocessing as mp
from threading import Thread

import dill
from tqdm.auto import tqdm

from .tools import _wrapped_func


class ProgressBar(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def close(self):
        super().close()
        if hasattr(self, 'disp'):
            if self.total and self.n < self.total:
                self.disp(bar_style='warning')


class ProgressStatus:
    def __init__(self):
        self.next_update = 1
        self.last_update_t = time.monotonic()
        self.last_update_val = 0


def progress_udf_wrapper(dill_func, workers_queue, total):
    state = ProgressStatus()
    func = dill.loads(dill_func)
    cnt = count(1)

    def wrapped_udf(*args, **kwargs):
        updated = next(cnt)
        if updated == state.next_update:
            time_now = time.monotonic()

            delta_t = time_now - state.last_update_t
            delta_i = updated - state.last_update_val

            state.next_update += max(int((delta_i / delta_t) * .25), 1)
            state.last_update_val = updated
            state.last_update_t = time_now
            workers_queue.put((1, delta_i))
        elif updated == total:
            workers_queue.put((1, updated - state.last_update_val))
        return func(*args, **kwargs)

    return wrapped_udf


def _process_status(bar_size, disable, q):
    bar = ProgressBar(total=bar_size, disable=disable, desc='DONE')
    while True:
        flag, upd_value = q.get()
        if not flag:
            bar.close()
            break

        bar.update(upd_value)


def _update_error_bar(bar_dict, bar_parameters):
    try:
        bar_dict['bar'].update()
    except KeyError:
        bar_dict['bar'] = ProgressBar(**bar_parameters)
        bar_dict['bar'].update()


def _error_behavior(error_handling, msgs, result, set_error_value, q):
    if error_handling == 'raise':
        q.put(None)
        raise
    elif error_handling == 'ignore':
        pass
    elif error_handling == 'coerce':
        if set_error_value is None:
            set_error_value = msgs
        result.append(set_error_value)
    else:
        raise ValueError(
            'Invalid error_handling value specified. Must be one of the values: "raise", "ignore", "coerce"')


def _do_parallel(func, tasks, initializer, initargs, n_cpu, total, disable, error_behavior, set_error_value,
                 q):
    if not n_cpu:
        n_cpu = mp.cpu_count()
    thread_ = Thread(target=_process_status, args=(total, disable, q))
    thread_.start()
    bar_parameters = dict(total=total, disable=disable, position=1, desc='ERROR', colour='red')
    error_bar = {}
    with mp.Pool(n_cpu, initializer=initializer, initargs=initargs) as p:
        result = list()
        iter_result = p.imap(func, tasks)
        while 1:
            try:
                result.append(next(iter_result))
            except StopIteration:
                break
            except Exception as e:
                _update_error_bar(error_bar, bar_parameters)
                _error_behavior(error_behavior, e, result, set_error_value, q)
    if error_bar:
        error_bar['bar'].close()
    q.put((None, None))
    thread_.join()
    return result


def progress_imap(func, tasks, q, initializer=None, initargs=(), n_cpu=None, total=None, disable=False,
                  process_timeout=None, error_behavior='raise', set_error_value=None,
                  ):
    if process_timeout:
        func = partial(_wrapped_func, func, process_timeout, True)
    result = _do_parallel(func, tasks, initializer, initargs, n_cpu, total, disable, error_behavior, set_error_value, q)
    return result
