#!/usr/bin/env python
#TODO: Find a way to make this ugly code nice
from hotspotter.HotSpotterAPI import HotSpotterAPI
#-------------------------------------------
# Figure out which environment we are in
# and set assocated preferences

# Attach to QtConsole's QApplication if able
from PyQt4.Qt import QCoreApplication
app = QCoreApplication.instance() 
in_qtc_bit = (not app is None)

if not in_qtc_bit:
    'if not in qtconsole, then configure matplotlib'
    import matplotlib
    matplotlib.use('Qt4Agg')


from PyQt4.Qt import QApplication, QEventLoop
from hotspotter.Facade import Facade
from hotspotter.other.logger import logmsg, hsl
import argparse
import inspect
import os.path
import sys

try:
    # Append the tpl lib to your path
    import hotspotter
    TPL_LIB_DIR = os.path.join(os.path.dirname(hotspotter.__file__), 'tpl', sys.platform,'lib')
    BOOST_LIB_DIR = r'C:\boost_1_53_0\stage\lib'
    sys.path.append(TPL_LIB_DIR)
    sys.path.append(BOOST_LIB_DIR)
except Exception: 
    print '''You must download hotspotter\'s 3rd party libraries before you can run it. 
    git clone https://github.com/Erotemic:tpl-hotspotter.git tpl'''

logmsg('Starting the program')

parser = argparse.ArgumentParser(description='HotSpotter - Instance Recognition', prefix_chars='+-')

def_on  = {'action':'store_false', 'default':True}
def_off = {'action':'store_true', 'default':False}

parser.add_argument('-l', '--log-all',         dest='logall_bit',   help='Writes all logs', **def_off)
parser.add_argument('--cmd',                   dest='cmd_bit',   help='Forces command line mode', **def_off)
parser.add_argument('-r', '--run-experiments', dest='runexpt_bit',  help='Runs the experiments', **def_off)
parser.add_argument('-g', '--gui-off',         dest='gui_bit',      help='Runs HotSpotter in command line mode', **def_on)
parser.add_argument('-a', '--autoload-off',    dest='autoload_bit', help='Starts HotSpotter without loading a database', **def_on)
parser.add_argument('-dp', '--delete-preferences', dest='delpref_bit', help='Deletes the HotSpotter preferences in ~/.hotspotter', **def_off)


args = parser.parse_args()

if args.logall_bit:
    hsl.enable_global_logs()

if not in_qtc_bit:
    app = QApplication(sys.argv)

# TODO: Remove the Facade, Have only the HotSpotterAPI
# Start HotSpotter via the Facade
fac = Facade(use_gui=args.gui_bit, autoload=args.autoload_bit)

if args.delpref_bit:
    fac.hs.delete_preferences()

for (name, value) in inspect.getmembers(Facade, predicate=inspect.ismethod):
    if name.find('_') != 0:
        exec('def '+name+'(*args, **kwargs): fac.'+name+'(*args, **kwargs)')
# Defined Aliases
stat, status   = [lambda          : fac.print_status()]*2
removec,       = [lambda          : fac.remove_cid()]
rename,        = [lambda new_name : fac.rename_cid(new_name)]
gview,         = [lambda          : fac.change_view('image_view')]
rview,         = [lambda          : fac.change_view('result_view')]
cview,         = [lambda          : fac.change_view('chip_view')]

# Add developer namespace
from PyQt4.Qt   import \
        QApplication, QMainWindow, QMessageBox, QAbstractItemView, QObject
from hotspotter.QueryManager import RawResults, QueryResult
#from PyQt4.QtCore import SIGNAL, Qt, pyqtSlot, pyqtSignal
from PIL import Image
import types
from hotspotter.other.ConcretePrintable import *
# Get commonly used variables for command line usage
hs = fac.hs
uim = hs.uim
hsgui = uim.hsgui
if hsgui != None:
    epw = hsgui.epw
cm,  nm,  gm,  am,  dm,  vm,  qm,  iom,  em = hs.get_managers(
'cm','nm','gm','am','dm','vm','qm','iom','em')

qcx = 1
#import dev 
#dnspc = dev.get_namespace(fac)
#exec(dnspc)

# TODO Move to dev
if args.runexpt_bit:
    em.run_quick_experiment()
    em.show_problems()


# be careful to not block the command line interface thread. 
run_new_exec_loop_bit = False
if args.cmd_bit or not args.gui_bit:
    try:
        print "Checking __IPYTHON__"
        __IPYTHON__
    except NameError as nex:
        try:
            print "Starting IPython Command Line Interaction"
            import IPython
            IPython.embed()
        except Exception as ex:
            print "IPython is not installed"
            run_new_exec_loop_bit = True
            print ex
elif in_qtc_bit:    
    print 'Starting QtConsole Command Line Interaction'
else:
    run_new_exec_loop_bit = True

if run_new_exec_loop_bit:
    print 'Running the application event loop'
    sys.stdout.flush()
    sys.exit(app.exec_())

sys.stdout.flush()
