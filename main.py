from PyQt4.QtCore import QCoreApplication
from PyQt4.Qt import QApplication, QEventLoop
from Facade import Facade
import sys
import inspect
from other.helpers import *
from other.logger import *
try:
    import tpl
    sys.path.append(os.path.join(os.path.dirname(tpl.__file__),'tpl',sys.platform,'lib'))
except Exception: 
    print '''You must download hotspotter\'s 3rd party libraries before you can run it. 
    git clone https://github.com/Erotemic:tpl-hotspotter.git tpl'''


use_gui         = True
autoload        = True
experiment_bit  = False
init_prefs = {}
logmsg('Starting the program')
for (argc, argv) in enumerate(sys.argv):
    if argv == '--no-plotwidget':
        init_prefs['plotwidget_bit'] = True
    if argv == '--plotwidget':
        init_prefs['plotwidget_bit'] = False
    if argv == '--no-gui':
        use_gui = False
    if argv == '--global-logs':
        hsl.enable_global_logs()
    if argv == '--no-autoload':
        autoload = False
    if argv == '--run-experiments':
        experiment_bit = True

# Attach to QtConsole's QApplication if able
app = QCoreApplication.instance() 
in_qtconsole_bit = (not app is None)
if not in_qtconsole_bit:
    app = QApplication(sys.argv)

# Start HotSpotter via the Facade
fac = Facade(use_gui=use_gui, autoload=autoload, init_prefs=init_prefs)
for (name, value) in inspect.getmembers(Facade, predicate=inspect.ismethod):
    if name.find('_') != 0:
        exec('def '+name+'(*args, **kdgs): fac.'+name+'(*args, **kdgs)')
# Defined Aliases
stat, status   = [lambda          : fac.print_status()]*2
removec,       = [lambda          : fac.remove_cid()]
rename,        = [lambda new_name : fac.rename_cid(new_name)]
gview,         = [lambda          : fac.change_view('image_view')]
rview,         = [lambda          : fac.change_view('result_view')]
cview,         = [lambda          : fac.change_view('chip_view')]

# Add developer namespace
import dev 
_ = dev.get_namespace(fac)
print _
exec(_)

# TODO Move to dev
if experiment_bit:
    em.run_quick_experiment()
    em.show_problems()

# Execute the threads 
if not in_qtconsole_bit:
    print 'Running the app until Exit'
    sys.stdout.flush()
    sys.exit(app.exec_())
else: 
    print 'Starting Command Line Interaction'
sys.stdout.flush()
