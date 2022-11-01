from functools import partial
from multiprocessing import Manager

from .progress_imap import progress_imap
from pandas.core.groupby.ops import _is_indexed_like
from pandas.util._decorators import doc

import dill

from .progress_imap import progress_udf_wrapper

DOC = 'Parallel analogue of the GroupBy.{func} method\nSee pandas GroupBy docstring for more ' \
      'information\nhttps://pandas.pydata.org/docs/reference/groupby.html'


def _do_group_apply(data, dill_func, workers_queue, args, kwargs):
    func = dill.loads(dill_func)
    result = progress_udf_wrapper(func, workers_queue, 1)(data[1], *args, **kwargs)
    return result, data[1].axes, 0


def _prepare_result(data):
    mutated = False
    result = list()
    for d in data:
        if not mutated and not _is_indexed_like(*d):
            mutated = True
        result.append(d[0])
    return result, mutated


def parallelize_groupby_apply(n_cpu=None, disable_pr_bar=False):
    @doc(DOC, func='apply')
    def p_apply(data, func, executor='processes', args=(), **kwargs):
        workers_queue = Manager().Queue()
        gr_count = data.ngroups
        iterator = iter(data)
        dill_func = dill.dumps(func)
        result = progress_imap(
            partial(_do_group_apply, dill_func=dill_func, workers_queue=workers_queue, args=args, kwargs=kwargs),
            iterator, workers_queue, total=gr_count, n_cpu=n_cpu, disable=disable_pr_bar, executor=executor,
            desc=func.__name__.upper()
        )
        result, mutated = _prepare_result(result)

        # due to a bug in the get_iterator method of the Basegrouper class
        # that was only fixed in pandas 1.4.0, earlier versions are not yet supported

        # pandas_version = get_pandas_version()
        #
        # if pandas_version < (1, 3):
        #     return data._wrap_applied_output(data.grouper._get_group_keys(), result,
        #                                      not_indexed_same=mutated or data.mutated)
        # elif pandas_version < (1, 4):
        #     return data._wrap_applied_output(data.grouper._get_group_keys(), data.grouper._get_group_keys(), result,
        #                                      not_indexed_same=mutated or data.mutated)
        return data._wrap_applied_output(data._selected_obj, result, not_indexed_same=mutated or data.mutated)

    return p_apply
