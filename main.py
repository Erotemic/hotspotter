#!/bin/python
#TODO Find what else is using matplotlib other than drawmanager and 
# get rid of it
import matplotlib
matplotlib.use('Qt4Agg')

from Facade import Facade
from PyQt4.Qt import QApplication, QEventLoop
from PyQt4.QtCore import QCoreApplication
from other.helpers import *
from other.logger import *
import argparse
import inspect
import sys

try:
    import tpl
    sys.path.append(os.path.join(os.path.dirname(tpl.__file__),'tpl',sys.platform,'lib'))
except Exception: 
    print '''You must download hotspotter\'s 3rd party libraries before you can run it. 
    git clone https://github.com/Erotemic:tpl-hotspotter.git tpl'''

init_prefs = {}
logmsg('Starting the program')

parser = argparse.ArgumentParser(description='HotSpotter - Instance Recognition', prefix_chars='+-')

def_on  = {'action':'store_false', 'default':True}
def_off = {'action':'store_true', 'default':False}

parser.add_argument('-l', '--log-all',         dest='logall_bit',   help='Writes all logs', **def_off)
parser.add_argument('--cmd',                   dest='cmd_bit',   help='Forces command line mode', **def_off)
parser.add_argument('-r', '--run-experiments', dest='runexpt_bit',  help='Runs the experiments', **def_off)
parser.add_argument('-g', '--gui-off',         dest='gui_bit',      help='Runs HotSpotter in command line mode', **def_on)
parser.add_argument('-a', '--autoload-off',    dest='autoload_bit', help='Starts HotSpotter without loading a database', **def_on)

args = parser.parse_args()

if args.logall_bit:
    hsl.enable_global_logs()

# Attach to QtConsole's QApplication if able
app = QCoreApplication.instance() 
in_qtc_bit = (not app is None)
if not in_qtc_bit:
    app = QApplication(sys.argv)

# Start HotSpotter via the Facade
fac = Facade(use_gui=args.gui_bit, autoload=args.autoload_bit, init_prefs=init_prefs)
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
dnspc = dev.get_namespace(fac)
exec(dnspc)

# TODO Move to dev
if args.runexpt_bit:
    em.run_quick_experiment()
    em.show_problems()


# be careful to not block the command line interface thread. 
run_new_exec_loop_bit = False
if args.cmd_bit or not args.gui_bit:
    try:
        __IPYTHON__
    except NameError:
        try:
            print "Starting IPython Command Line Interaction"
            import IPython
            IPython.embed()
        except Exception as e:
            run_new_exec_loop_bit = True
            print e
elif in_qtc_bit:    
    print 'Starting QtConsole Command Line Interaction'
else:
    run_new_exec_loop_bit = True
if run_new_exec_loop_bit:
    print 'Running the application event loop'
    sys.stdout.flush()
    sys.exit(app.exec_())

sys.stdout.flush()
