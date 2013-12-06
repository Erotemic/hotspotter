import time
import collections
import traceback
import os
import sys
import types
from hotspotter.other.AbstractPrintable import AbstractPrintable
from PyQt4.Qt import QObject
from PyQt4.QtGui import QMessageBox
from PyQt4.QtCore import Qt


#import os
#os.spawnl(os.P_DETACH, 'some_log_running_command')
# Have this happen.^^ 
# http://stackoverflow.com/questions/1196074/starting-a-background-process-in-python

## Embaressingly parallel stuff can use this
#import subprocess
#subprocess.Popen(["rm","-r","some.file"])
#
#
class CallInfoObject(AbstractPrintable):
    def __init__(self, module, line, func, indent, calltype, prefix):
        super(CallInfoObject,self).__init__()
        self.module = module
        self.line = line
        self.func = func
        self.indent = indent
        self.calltype = calltype
        self.prefix = prefix
#---------------

debug_blacklist = '''
helpers.dircheck
_hsPrefs
ChipManager.add_chip
ImageManager.add_img
NameManager.add_name
alloc
IOManager.__
_hsGlobals.x2_info
'''

#---------------

def callinfo(num_up=2):
    'finds how much to indent'
    stack_list = traceback.extract_stack()
    indent = 0
    calltype = '   '
    for stack_tup in reversed(stack_list): 
        if stack_tup[2] == 'start_event_loop_qt4':
            calltype = 'gui'
            break
        elif stack_tup[2] == 'execfile':
            calltype = 'ini'
            break
        elif stack_tup[2] == 'run_code':
            calltype = 'ipy'
            break
        elif stack_tup[0] == 'main.py':
            calltype = 'cmd'
            break
        indent += 1
    #print stack_list
    caller_tup = stack_list[ -num_up ]
    modul_name = os.path.split(caller_tup[0])[-1].replace('.py','')
    line_num   = caller_tup[1]
    func_name  = caller_tup[2]
    #prefix = modul_name+'.'+func_name+'('+str(line_num)+')' # WITH LINE 
    prefix = modul_name+'.'+func_name # WITHOUT LINE
    return CallInfoObject(modul_name, line_num, func_name, indent-num_up, calltype, prefix)

#---------------
class HotSpotterLogger(object):
    def enable_global_logs(hsl):
        logfile = os.path.realpath('hotspotter_global_logs.log')
        print('Enableing active logging to file: ' +  logfile)
        hsl.global_logs_on = True
        if hsl.global_logs_on:
            hsl.global_log_file = open(logfile,'a')
    def __init__(hsl):
        hsl.error_num = 0
        hsl.global_logs_on = False
        hsl.logged_lines = collections.deque(maxlen=2000)
        hsl.debug_blacklist = debug_blacklist            
        hsl.cursor_x = 0
        hsl.cursor = '/-\|'
        hsl.kirby = \
        ['(>\'-\')>  %4d',
         '<(\'-\'<)  %4d',
         '^(\'- \')^ %4d',
         '<(\'-\'<)  %4d']
        hsl.delete_kirby = '\b'*13
        hsl.prev_time = time.time()
        hsl.non_modal_qt_handles = []
    
    def non_modal_critical_dialog(hsl, title, msg):
        try:
            # Make a non modal critical QMessageBox
            msgBox = QMessageBox( None );
            msgBox.setAttribute( Qt.WA_DeleteOnClose )
            msgBox.setStandardButtons( QMessageBox.Ok )
            msgBox.setWindowTitle( title )
            msgBox.setText( msg )
            msgBox.setModal( False )
            msgBox.open( msgBox.close )
            msgBox.show()
            hsl.non_modal_qt_handles.append(msgBox)
            # Old Modal Version: QMessageBox.critical(None, 'ERROR', msg)
        except Exception as ex:
            print('non_modal_critical_dialog: '+str(ex))

    def __str__(hsl):
        return hsl.hidden_logs()

    def hidden_logs(hsl, use_blacklist_bit=True):
        logged_hidden = ''
        for line in hsl.logged_lines:
            if (not use_blacklist_bit) or all( [len(bl_) == 0 or line.find(bl_) == -1 for bl_ in debug_blacklist.splitlines()] ):
                logged_hidden += line + '\n'
        return logged_hidden

    def log(hsl, msg, noprint=False, noformat=False):
        if noformat:
            fmted_msg = msg
        else: 
            info = callinfo(4) # Get Prefix Info
            indent_str = info.calltype+'. '*info.indent
            # Format Info
            indent_pfx = indent_str+info.prefix
            indent_msg = str(msg).replace('\n', '\n'+(' '*len(indent_str)))
            fmted_msg  = indent_pfx+': '+indent_msg
        hsl.logged_lines.append(fmted_msg)
        if hsl.global_logs_on: # Log to global logs
            hsl.global_log_file.write(fmted_msg+'\n')
        if noprint: # If not print show there is progress
            since_last  = time.time() - hsl.prev_time
            time_thresh = 1
            if since_last > time_thresh:
                if hsl.cursor_x == -1:
                    pass
                elif hsl.cursor_x ==  0:
                    # Write Kirby
                    #sys.stdout.write('Working: '+(hsl.kirby[hsl.cursor_x % 4] % hsl.cursor_x))
                    #sys.stdout.flush()
                    pass
                else:
                    # Write Kirby
                    #sys.stdout.write(hsl.delete_kirby+(hsl.kirby[hsl.cursor_x % 4] % hsl.cursor_x))
                    pass
                    #sys.stdout.flush()
                hsl.cursor_x += 1
                hsl.prev_time = time.time()
        else:
            if hsl.cursor_x > 0:
                fmted_msg = '\n'+fmted_msg 
            hsl.cursor_x = -1
            #sys.stdout.write(fmted_msg+'\n')
            print(msg)
            hsl.prev_time = time.time()

hsl = HotSpotterLogger()

class FuncLogException(Exception):
    'FuncLog Exceptsions have already been handled by the wrapper and are benign'
    def __init__(self, value):
        sys.stdout.flush(); sys.stderr.flush()
        self.value = value
    def __str__(self):
        return str(self.value)

class LogErrorException(Exception):
    def __init__(self, error_num=-1):
        self.error_num = error_num
    def __str__(self):
        return str('<LogError Num '+str(self.error_num)+'>')


def logwarn(msg):
    hsl.log('<WARN START **************************')
    hsl.log('<WARNING-TRACEBACK> '+traceback.format_exc())
    hsl.log('<WARNING> '+msg)
    hsl.log('WARN END **************************>')
    sys.stdout.flush(); sys.stderr.flush()

def logerr(msg=None):
    error_num = hsl.error_num
    hsl.error_num += 1
    hsl.log('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    hsl.log('<ERROR Number %d>' % error_num)
    hsl.log('\n\n *!* HotSpotter Raised Exception: %s \n' % str(msg))
    hsl.log('<ERROR Number %d>' % error_num)
    hsl.log('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #hsl.log('\n\n *!* HotSpotter Exception Traceback: \n'+traceback.format_exc())
    sys.stdout.flush(); sys.stderr.flush()
    hsl.non_modal_critical_dialog('ERROR #%d' % error_num, msg)
    raise LogErrorException(error_num)

def logmsg(msg):
    hsl.log(msg)

def logdbg(msg):
    hsl.log('> '+msg, noprint=True)

def logio(msg):    
    hsl.log('IO> '+msg, noprint=True)

def func_debug(fn):
    import traceback
    def func_debug_wraper(*args, **kwargs):
        print('\n\n *!!* Func Debug Traceback: \n\n\n'+str(traceback.format_stack()))
        logdbg('\n\n *!!* Func Debug Traceback: \n\n'+str(traceback.format_stack()))
        return fn(*args, **kwargs)
    return func_debug_wraper

    

def func_log(fn):
    def func_log_wraper(*args, **kwargs):
        # -- Format Logs
        # -- Arg Info
        argnames = fn.func_code.co_varnames[:fn.func_code.co_argcount]
        argprint = ''
        arg_length_cutoff = 100
        if len(args) > 0 and\
           isinstance(args[0], AbstractPrintable) or\
           isinstance(args[0], QObject):
            #argprint = ', '.join( '%s=%r' % entry for entry in zip(argnames[1:],args[1:]) + kwargs.items())
            arg_rep  = lambda argstr: argstr if len(argstr) < arg_length_cutoff else '...'
            arg_iter = iter(zip(argnames[1:],args[1:]) + kwargs.items())
            argprint = ', '.join( var+'='+arg_rep(repr(val)) for (var,val) in arg_iter)
        else:
            arg_rep  = lambda argstr: argstr if len(argstr) < arg_length_cutoff else '...'
            arg_iter = iter(zip(argnames,args) + kwargs.items())
            argprint = ', '.join( var+'='+arg_rep(repr(val)) for (var,val) in arg_iter)
        # -- Module / Line Info
        info = callinfo(3) 
        module = str(fn.func_code)
        module = module[max(module.find('\\'), module.rfind('/'))+1:module.rfind('.py')]
        align_space = 80
        function_name = fn.func_name
        #if info.indent < 1: # Hack to make kirby not appear every time you do anything
        #    logmsg(module+'.'+function_name+'('+argprint+')')
        into_str = 'In  '+module+'.'+function_name+'('+argprint+')'
        outo_str = 'Out '+module+'.'+function_name+'('+argprint+')'#+'\n'
        indent1 = '> '*info.indent
        prefix_sep = '--'
        fill_length = max(0,align_space-len(indent1)-len(prefix_sep))
        indent2 = ' '*fill_length
        prefixIN  = info.calltype+indent1+prefix_sep+indent2
        indent1OUT = ' '*len(indent1)
        indent2OUT = indent2.replace('  ',' <')
        prefixOUT = info.calltype+indent1OUT+prefix_sep+indent2OUT
        # -- Log Enter Function 
        hsl.log(prefixIN+into_str, noprint=True, noformat=True)
        # -- Run Function
        ret = None
        try:
            ret = fn(*args, **kwargs)
        except FuncLogException as ex:
            logdbg('Caught FuncLog-Exception: '+str(ex))
        except LogErrorException as ex: 
            logdbg('Caught LogError-Exception: '+str(ex))
            et, ei, tb = sys.exc_info()
            #et, ei, tb = sys.exc_info()
            #raise FuncLogException, FuncLogException(e), tb
        except Exception as ex: 
            logmsg('\n\n *!!* HotSpotter Logger Raised Exception: '+str(ex))
            logmsg('\n\n *!!* HotSpotter Logger Exception Traceback: \n\n'+traceback.format_exc())
            sys.stdout.flush()
            et, ei, tb = sys.exc_info()
            #raise FuncLogException, FuncLogException(e), tb
        # --- Log Exit Function
        ret_str = ' returned '+str(ret)
        hsl.log(prefixOUT+outo_str+ret_str, noprint=True, noformat=True)
        if info.indent < 1:
            hsl.log('\n\n', noprint=True, noformat=True)
        sys.stdout.flush(); sys.stderr.flush()
        return ret
    func_log_wraper.__name__ = fn.__name__
    #func_log_wraper.__doc__ = fn.__doc__
    #func_log_wraper.__dict__.update(fn.__dict__)
    return func_log_wraper

