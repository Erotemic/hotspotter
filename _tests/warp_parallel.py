'''
There is an issue with cv2.warpAffine on macs.
This is a test to further investigate the issue.
python -c "import cv2; help(cv2.warpAffine)"
'''
from __future__ import division, print_function
import os
import sys
from os.path import dirname, join, expanduser, exists
from PIL import Image
import numpy as np
import multiprocessing
import cv2
from itertools import izip


def try_get_path(path_list):
    tried_list = []
    for path in path_list:
        if path.find('~') != -1:
            path = expanduser(path)
        tried_list.append(path)
        if exists(path):
            return path
    return (False, tried_list)


def get_lena_fpath():
    possible_lena_locations = [
        'lena.png',
        '~/code/hotspotter/_tpl/extern_feat/lena.png',
        '_tpl/extern_feat/lena.png',
        '../_tpl/extern_feat/lena.png',
        '~/local/lena.png',
        '../lena.png',
        '/lena.png',
        'C:\\lena.png']
    lena_fpath = try_get_path(possible_lena_locations)
    if not isinstance(lena_fpath, str):
        raise Exception('cannot find lena: tried: %r' % (lena_fpath,))
    return lena_fpath

try:
    test_dir = join(dirname(__file__))
except NameError as ex:
    test_dir = os.getcwd()

# Get information
gfpath = get_lena_fpath()
cfpath = join(test_dir, 'tmp_chip.png')
roi = [0, 0, 100, 100]
new_size = (500, 500)
theta = 0
img_path = gfpath

# parallel tasks
nTasks = 8
gfpath_list = [gfpath] * nTasks
cfpath_list = [cfpath] * nTasks
roi_list = [roi] * nTasks
theta_list = [theta] * nTasks
chipsz_list = [new_size] * nTasks

printDBG = print


def imread(img_fpath):
    try:
        img = Image.open(img_fpath)
        img = np.asarray(img)
        #img = skimage.util.img_as_uint(img)
    except Exception as ex:
        print('[io] Caught Exception: %r' % ex)
        print('[io] ERROR reading: %r' % (img_fpath,))
        raise
    return img


def _calculate(func, args):
    printDBG('[parallel] * %s calculating...' % (multiprocessing.current_process().name,))
    result = func(*args)
    #arg_names = func.func_code.co_varnames[:func.func_code.co_argcount]
    #arg_list  = [n+'='+str(v) for n,v in izip(arg_names, args)]
    #arg_str = '\n    *** '+str('\n    *** '.join(arg_list))
    printDBG('[parallel]  * %s finished:\n    ** %s' %
            (multiprocessing.current_process().name,
             func.__name__))
    return result


def _worker(input, output):
    printDBG('[parallel] START WORKER input=%r output=%r' % (input, output))
    for func, args in iter(input.get, 'STOP'):
        printDBG('[parallel] worker will calculate %r' % (func))
        result = _calculate(func, args)
        printDBG('[parallel] worker has calculated %r' % (func))
        output.put(result)
        #printDBG('[parallel] worker put result in queue.')
    #printDBG('[parallel] worker is done input=%r output=%r' % (input, output))


def mark_progress(count):
    sys.stdout.write('.')


def compute_in_parallel(task_list, num_procs, task_lbl='', verbose=True):
    '''
    Input: task list: [ (fn, args), ... ]
    '''
    task_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()
    nTasks = len(task_list)
    # queue tasks
    for task in iter(task_list):
        task_queue.put(task)
    # start processes
    proc_list = []
    for i in xrange(num_procs):
        printDBG('[parallel] creating process %r' % (i,))
        proc = multiprocessing.Process(target=_worker, args=(task_queue, done_queue))
        proc.start()
        proc_list.append(proc_list)
    # wait for results
    printDBG('[parallel] waiting for results')
    sys.stdout.flush()
    result_list = []
    if verbose:
        for count in xrange(len(task_list)):
            printDBG('[parallel] done_queue.get()')
            result_list.append(done_queue.get())
            mark_progress(count)
        print('')
    else:
        for i in xrange(nTasks):
            done_queue.get()
        print('[parallel]  ... done')
    printDBG('[parallel] stopping children')
    # stop children processes
    for i in xrange(num_procs):
        task_queue.put('STOP')
    return result_list


def extract_chip(img_path, roi, theta, new_size):
    'Crops chip from image ; Rotates and scales; Converts to grayscale'
    # Read parent image
    np_img = imread(img_path)
    # Build transformation
    (rx, ry, rw, rh) = roi
    (rw_, rh_) = new_size
    #Aff = cc2.build_transform(rx, ry, rw, rh, rw_, rh_, theta, affine=True)
    Aff = np.array([[ 2., 0.,  0.],
                    [ 0., 1.,  0.]])
    print('built transform Aff=\n%r' % Aff)
    # Rotate and scale
    #chip = cv2.warpAffine(np_img, Aff, (rw_, rh_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    chip = cv2.warpAffine(np_img, Aff, (rw_, rh_))
    print('warped')
    return chip


func = extract_chip
# Try one task
#func(gfpath, cfpath, roi, theta, new_size)

task_list = [(func, _args) for _args in izip(gfpath_list, roi_list, theta_list, chipsz_list)]
for task in task_list:
    print(task)

result_list = compute_in_parallel(task_list, 2)
for result in result_list:
    print(result)


#from hotspotter import draw_func2 as df2
#df2.imshow(result_list[0])
#exec(df2.present())
