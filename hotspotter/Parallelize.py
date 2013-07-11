# http://docs.python.org/2/library/multiprocessing.html
from __future__ import print_function, division
import multiprocessing as mp
from hotspotter.helpers import Timer
import sys

def _calculate(func, args):
    result = func(*args)
    arg_names = func.func_code.co_varnames[:func.func_code.co_argcount]
    arg_list  = [n+'='+str(v) for n,v in iter(zip(arg_names, args))]
    arg_str = '\n    *** '+str('\n    *** '.join(arg_list))
    return '  * %s finished:\n    ** %s%s \n    ** %s' % \
            (mp.current_process().name,
             func.__name__,
             arg_str,
             result)

def _worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = _calculate(func, args)
        output.put(result)

def cpu_count():
    return mp.cpu_count()

def parallelize_tasks(task_list, num_procs, verbose=False):
    '''
    Used for embarissingly parallel tasks, which write output to disk
    '''
    timer_msg = 'Distrubiting '+str(len(task_list))+' tasks to ' + str(num_procs) + ' parallel processes'
    
    with Timer(msg=timer_msg) as t:
        if num_procs > 1:
            if verbose:
                print('  * Computing in parallel process')
            _parallelize_tasks(task_list, num_procs, verbose)
        else:
            if verbose: 
                print('Computing in serial process')
            total = len(task_list)
            sys.stdout.write('    ')
            for count, (fn, args) in enumerate(task_list):
                if verbose:
                    print('  * computing %d / %d ' % (count, total))
                else: 
                    sys.stdout.write('.')
                    if (count+1) % 80 == 0:
                        sys.stdout.write('\n    ')
                    sys.stdout.flush()
                fn(*args)
            sys.stdout.write('\n')

def _parallelize_tasks(task_list, num_procs, verbose):
    '''
    Input: task list: [ (fn, args), ... ] 
    '''
    task_queue = mp.Queue()
    done_queue = mp.Queue()
    if verbose: 
        print('  * Submiting '+str(len(task_list))+' tasks:')
    # queue tasks
    for task in iter(task_list):
        task_queue.put(task)
    # start processes
    for i in xrange(num_procs):
        mp.Process(target=_worker, args=(task_queue, done_queue)).start()
    # Get and print results
    if verbose:
        print('  * Unordered results:')
        for i in xrange(len(task_list)):
            print(done_queue.get())
    else:
        sys.stdout.write('    ')
        newln_len = num_procs * int(80/num_procs)
        for i in xrange(len(task_list)):
            done_queue.get()
            sys.stdout.write('.')
            if (i+1) % num_procs == 0:
                sys.stdout.write(' ')
            if (i+1) % newln_len == 0:
                sys.stdout.write('\n    ')
            sys.stdout.flush()
        print('\n  ... Finished')
    # Tell child processes to stop
    for i in xrange(num_procs):
        task_queue.put('STOP')
