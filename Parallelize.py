# http://docs.python.org/2/library/multiprocessing.html
from __future__ import print_function, division
import __builtin__
import multiprocessing as mp
import sys
from os.path import exists
from itertools import izip
import helpers

# Toggleable printing
print = __builtin__.print
print_ = sys.stdout.write


def print_on():
    global print, print_
    print =  __builtin__.print
    print_ = sys.stdout.write


def print_off():
    global print, print_

    def print(*args, **kwargs):
        pass

    def print_(*args, **kwargs):
        pass


# Dynamic module reloading
def rrr():
    import imp
    print('[parallel] reloading ' + __name__)
    imp.reload(sys.modules[__name__])


def _calculate(func, args):
    result = func(*args)
    #arg_names = func.func_code.co_varnames[:func.func_code.co_argcount]
    #arg_list  = [n+'='+str(v) for n,v in izip(arg_names, args)]
    #arg_str = '\n    *** '+str('\n    *** '.join(arg_list))
    #print('[parallel]  * %s finished:\n    ** %s%s' % \
            #(mp.current_process().name,
             #func.__name__,
             #arg_str))
    return result


def _worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = _calculate(func, args)
        output.put(result)


def parallel_compute(func, arg_list, num_procs=None, lazy=True, args=None, common_args=[]):
    if args is not None and num_procs is None:
        num_procs = args.num_procs
    task_list = make_task_list(func, arg_list, lazy=lazy, common_args=common_args)
    nTasks = len(task_list)
    if nTasks == 0:
        print('[parallel] ... No %s tasks left to compute!' % func.func_name)
        return None
    # Do not execute small tasks in parallel
    if nTasks < num_procs / 2 or nTasks == 1:
        num_procs = 1
    num_procs = min(num_procs, nTasks)
    task_lbl = func.func_name
    try:
        ret = parallelize_tasks(task_list, num_procs, task_lbl)
    except Exception as ex:
        sys.stdout.flush()
        print('[parallel!] Problem while parallelizing task: %r' % ex)
        print('[parallel!] task_list: ')
        for task in task_list:
            print('  %r' % (task,))
            break
        print('[parallel!] common_args = %r' % common_args)
        print('[parallel! num_procs = %r ' % (num_procs,))
        print('[parallel!] task_lbl = %r ' % (task_lbl,))
        sys.stdout.flush()
        raise
    return ret


def make_task_list(func, arg_list, lazy=True, common_args=[]):
    '''
    The input should alawyas be argument 1
    The output should always be argument 2
    '''
    has_output = len(arg_list) >= 2
    append_common = lambda _args: tuple(list(_args) + common_args)
    if not (lazy and has_output):
        # does not check existance
        task_list = [(func, append_common(_args)) for _args in izip(*arg_list)]
        return task_list
    # checks existance
    arg_list2 = [append_common(_args) for _args in izip(*arg_list) if not exists(_args[1])]
    task_list = [(func, _args) for _args in iter(arg_list2)]
    nSkip = len(zip(*arg_list)) - len(arg_list2)
    print('[parallel] Already computed %d %s tasks' % (nSkip, func.func_name))
    return task_list


def parallelize_tasks(task_list, num_procs, task_lbl='', verbose=True):
    '''
    Used for embarissingly parallel tasks, which write output to disk
    '''
    nTasks = len(task_list)
    msg = ('Distributing %d %s tasks to %d processes' % (nTasks, task_lbl, num_procs)
           if num_procs > 1 else
           'Executing %d %s tasks in serial' % (nTasks, task_lbl))
    with helpers.Timer(msg=msg):
        if num_procs > 1:
            # Parallelize tasks
            return _compute_in_parallel(task_list, num_procs, task_lbl, verbose)
        else:
            return _compute_in_serial(task_list, task_lbl, verbose)


def _compute_in_serial(task_list, task_lbl='', verbose=True):
    # Serialize Tasks
    result_list = []
    nTasks = len(task_list)
    if verbose:
        mark_progress = helpers.progress_func(nTasks, lbl=task_lbl)
        # Compute each task
        for count, (fn, args) in enumerate(task_list):
            mark_progress(count)
            #sys.stdout.flush()
            result = fn(*args)
            result_list.append(result)
        print('')
    else:
        # Compute each task
        for (fn, args) in iter(task_list):
            result = fn(*args)
            result_list.append(result)
        print('[parallel]  ... done')
    return result_list


def _compute_in_parallel(task_list, num_procs, task_lbl='', verbose=True):
    '''
    Input: task list: [ (fn, args), ... ]
    '''
    task_queue = mp.Queue()
    done_queue = mp.Queue()
    nTasks = len(task_list)
    # queue tasks
    for task in iter(task_list):
        task_queue.put(task)
    # start processes
    for i in xrange(num_procs):
        mp.Process(target=_worker, args=(task_queue, done_queue)).start()
    # wait for results
    if verbose:
        mark_progress = helpers.progress_func(nTasks, lbl=task_lbl, spacing=num_procs)
        for count in xrange(len(task_list)):
            done_queue.get()
            mark_progress(count)
        print('')
    else:
        for i in xrange(nTasks):
            done_queue.get()
        print('[parallel]  ... done')
    # stop children processes
    for i in xrange(num_procs):
        task_queue.put('STOP')
    return done_queue


if __name__ == '__main__':
    print('test parallel')
    import multiprocessing
    import numpy as np

    p = multiprocessing.Pool(processes=8)
    data_list = [np.random.rand(1000, 9) for _ in xrange(1000)]
    data = data_list[0]

    def complex_func(data):
        tmp = 0
        for ix in xrange(0, 100):
            _r = np.random.rand(10, 10)
            u1, s1, v1 = np.linalg.svd(_r)
            tmp += s1[0]
        u, s, v = np.linalg.svd(data)
        return s[0] + tmp

    with helpers.Timer('ser'):
        x2 = map(complex_func, data_list)
    with helpers.Timer('par'):
        x1 = p.map(complex_func, data_list)

    '''
    %timeit p.map(numpy.sqrt, x)
    %timeit map(numpy.sqrt, x)
    '''
