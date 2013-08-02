import atexit

from IPython.zmq.ipkernel import IPKernelApp
from IPython.lib.kernel import find_connection_file
from IPython.frontend.qt.kernelmanager import QtKernelManager
from IPython.frontend.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.utils.traitlets import TraitError
from PyQt4 import QtGui, QtCore

def event_loop(kernel):
    kernel.timer = QtCore.QTimer()
    kernel.timer.timeout.connect(kernel.do_one_iteration)
    kernel.timer.start(1000*kernel._poll_interval)

def default_kernel_app():
    app = IPKernelApp.instance()
    app.initialize(['python'])
    app.kernel.eventloop = event_loop
    return app

def default_manager(kernel):
    connection_file = find_connection_file(kernel.connection_file)
    manager = QtKernelManager(connection_file=connection_file)
    manager.load_connection_file()
    manager.start_channels()
    atexit.register(manager.cleanup_connection_file)
    return manager

def console_widget(manager):
    try: # Ipython v0.13
        widget = RichIPythonWidget(gui_completion='droplist')
    except TraitError:  # IPython v0.12
        widget = RichIPythonWidget(gui_completion=True)
    widget.kernel_manager = manager
    return widget

def terminal_widget(**kwargs):
    kernel_app = default_kernel_app()
    manager = default_manager(kernel_app)
    widget = console_widget(manager)
    widget.set_default_style(colors='linux')

    #update namespace                                                           
    kernel_app.shell.user_ns.update(kwargs)

    kernel_app.start()
    return widget

def run_console2(app=None):
    create_app_bit = app==None
    if create_app_bit:
        app = QtGui.QApplication([])
    widget = terminal_widget(testing=123)
    widget.show()
    if create_app_bit:
        app.exec_()

if __name__ == '__main__':
    ret = run_console2()
