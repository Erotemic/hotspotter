'''
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    print('[helpers] You ran helpers as main!')
    module = sys.modules[__name__]
    seen = set(['numpy', 'matplotlib', 'scipy', 'pyflann', 'skimage', 'cv2'])

    hs2_basic = set(['draw_func2', 'params', 'mc2'])
    python_basic = set(['os', 'sys', 'warnings', 'inspect', 'copy', 'imp', 'types'])
    tpl_basic = set(['pyflann', 'cv2'])
    science_basic = set(['numpy',
                         'matplotlib',
                         'matplotlib.pyplot',
                         'scipy',
                         'scipy.sparse'])
    seen = set(list(python_basic) + list(science_basic) + list(tpl_basic))
    seen = set([])
    print('[helpers] seen=%r' % seen)
    explore_module(module, maxdepth=0, seen=seen, nonmodules=False)
'''
