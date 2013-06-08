# http://docs.python.org/2/library/multiprocessing.html
import multiprocessing as mp
from hotspotter.helpers import Timer

def _calculate(func, args):
    result = func(*args)
    return '  * %s finished:\n    ** %s%s \n    ** %s' % \
            (mp.current_process().name,
             func.__name__,
             str(args).replace(',', ',\n    *** '),
             result)

def _worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = _calculate(func, args)
        output.put(result)

def cpu_count():
    return mp.cpu_count()

def parallelize_tasks(task_list, num_procs):
    '''
    Used for embarissingly parallel tasks, which write output to disk
    '''
    with Timer() as t:
        if num_procs > 1:
            print('  * Computing in parallel process')
            _parallelize_tasks(task_list, num_procs)
        else:
            print('Computing in serial process')
            total = len(task_list)
            for count, (fn, args) in enumerate(task_list):
                print('  * computing %d / %d ' % (count, total))
                fn(*args)

def _parallelize_tasks(task_list, num_procs):
    '''
    Input: task list: [ (fn, args), ... ] 
    '''
    task_queue = mp.Queue()
    done_queue = mp.Queue()
    print('  * Submiting tasks:')
    # queue tasks
    for task in task_list:
        task_queue.put(task)
    # start processes
    print('  * Starting ' + str(num_procs) + ' processes')
    for i in xrange(num_procs):
        mp.Process(target=_worker, args=(task_queue, done_queue)).start()
    # Get and print results
    print('  * Unordered results:')
    for i in xrange(len(task_list)):
        print(done_queue.get())
    # Tell child processes to stop
    for i in xrange(num_procs):
        task_queue.put('STOP')
