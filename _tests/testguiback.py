'''
# Main Test Script
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('__main__ = gui.py')
    app, is_root = guitools.init_qtapp()
    back = guitools.make_main_window()
    front = back.front
    ui = front.ui
    guitools.run_main_loop(app, is_root, back)
'''
