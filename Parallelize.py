# http://docs.python.org/2/library/multiprocessing.html
from __future__ import print_function, division
import multiprocessing as mp
from helpers import Timer
import sys
import os.path
import params

def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])


def _calculate(func, args):
    result = func(*args)
    #arg_names = func.func_code.co_varnames[:func.func_code.co_argcount]
    #arg_list  = [n+'='+str(v) for n,v in iter(zip(arg_names, args))]
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

def cpu_count():
    return mp.cpu_count()

def parallel_compute(func, arg_list, num_procs=None, lazy=True):
    num_procs = params.NUM_PROCS if num_procs is None else num_procs
    if lazy:
        task_list = make_task_list_lazy(func, *arg_list)
    else:
        task_list = make_task_list(func, *arg_list)
    if len(task_list) == 0:
        print('[parallel] ... No '+func.func_name+' tasks left to compute!')
        return None
    if num_procs > 1:
        msg = 'Distributing %d %s tasks to %d parallel processes' % \
                (len(task_list), func.func_name, num_procs)
    else: 
        msg = 'Executing %d %s tasks in serial' % \
                (len(task_list), func.func_name)
    try:
        ret = parallelize_tasks(task_list, num_procs, msg=msg)
    except Exception as ex:
        print('Problem while parallelizing task: ')
        print('task_list: ')
        for task in task_list:
            print('  %r' % (task,))
        print('num_procs = %r ' % (num_procs,))
        print('msg = %r ' % (msg,))
        raise
    return ret

def make_task_list_lazy(func, *args):
    # The input should alawyas be argument 1
    # The output should always be argument 2
    task_list = []
    lazy_skips = 0
    has_output = len(args) >= 2
    for _args in iter(zip(*args)):
        if has_output and os.path.exists(_args[1]):
            lazy_skips += 1
        else:
            task_list.append((func, _args))
    print('[parallel] Already computed '+str(lazy_skips)+' '+func.func_name+' tasks')
    return task_list
def make_task_list(func, *args):
    arg_iterator = iter(zip(*args))
    task_list    = [(func, _args) for _args in arg_iterator]
    return task_list

def parallelize_tasks(task_list, num_procs, msg=''):
    '''
    Used for embarissingly parallel tasks, which write output to disk
    '''
    with Timer(msg=msg) as t:
        if num_procs > 1:
            if False:
                print('[parallel] Computing in parallel process')
            return _parallelize_tasks(task_list, num_procs, False)
        else:
            result_list = []
            if False: 
                print('[parallel] Computing in serial process')
            total = len(task_list)
            sys.stdout.write('    ')
            for count, (fn, args) in enumerate(task_list):
                if False:
                    print('[parallel] computing %d / %d ' % (count, total))
                else: 
                    sys.stdout.write('.')
                    if (count+1) % 80 == 0:
                        sys.stdout.write('\n    ')
                    sys.stdout.flush()
                result = fn(*args)
                result_list.append(result)
            sys.stdout.write('\n')
            return result_list

def _parallelize_tasks(task_list, num_procs, verbose):
    '''
    Input: task list: [ (fn, args), ... ] 
    '''
    task_queue = mp.Queue()
    done_queue = mp.Queue()
    if verbose: 
        print('[parallel] Submiting '+str(len(task_list))+' tasks:')
    # queue tasks
    for task in iter(task_list):
        task_queue.put(task)
    # start processes
    for i in xrange(num_procs):
        mp.Process(target=_worker, args=(task_queue, done_queue)).start()
    # Get and print results
    if verbose:
        print('[parallel] Unordered results:')
        for i in xrange(len(task_list)):
            print(done_queue.get())
    else:
        sys.stdout.write('    ')
        newln_len = num_procs * int(80/num_procs)
        for i in xrange(len(task_list)):
            done_queue.get()
            sys.stdout.write('.')
            if (i+1) % num_procs == 0: sys.stdout.write(' ')
            if (i+1) % newln_len == 0: sys.stdout.write('\n    ')
            sys.stdout.flush()
        print('\n[parallel]  ... done')
    # Tell child processes to stop
    for i in xrange(num_procs):
        task_queue.put('STOP')
    return done_queue
