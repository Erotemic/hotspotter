#'''The hotspotter python module'''
##------------------------------------------------------------------------------
## Future Imports
#from __future__ import division
#DEBUGGING = False
#if DEBUGGING:
    ## Multiprocessing checks
    #import multiprocessing as mp
    #if mp.current_process().name != 'MainProcess':
        #raise Exception('__init__> Cannot import from parallel process')
    ## Matplotlib initialization
    #import matplotlib
    #if matplotlib.get_backend() != 'Qt4Agg':
        #print('__init__>matplotlib.use(Qt4Agg)')
        #matplotlib.use('Qt4Agg', warn=True, force=True)
        #matplotlib.rcParams['toolbar'] = 'None'
##------------------------------------------------------------------------------
## Bread and butter imports
#import os
#import sys
#from os.path import join, relpath, realpath, normpath, dirname, exists
## Ensure that hotspotter is in the PYTHONPATH
#try:
    #source_dir = os.path.dirname(__file__)
#except NameError as ex:
    #source_dir = os.getcwd()
    #pass
#print(' * import hotspotter from: %r' % source_dir)
#if os.environ['PYTHONPATH'].find(source_dir) == -1:
    #toappend = source_dir + os.pathsep
    #os.environ['PYTHONPATH'] = toappend + os.environ['PYTHONPATH']

##------------------------------------------------------------------------------
## Standard Library
#from itertools import izip
#import cPickle
#import cStringIO
#import code
#import datetime
#import fnmatch
#import functools
#import imp
#import inspect
#import itertools
#import multiprocessing
#import re
#import shutil
#import signal
#import subprocess
#import textwrap 
#import time
#import traceback
#import types
#import warnings
#from collections import defaultdict

##------------------------------------------------------------------------------
## Scientific Imports
#import matplotlib.gridspec as gridspec 
#import matplotlib.pyplot as plt
#import pylab
#import cv2
#import pyflann
#import PIL.Image
#import PIL.ImageOps
#import numpy as np 
#import numpy.linalg
#import scipy
#import scipy.signal
#import scipy.ndimage.filters
#import scipy.sparse
#import scipy.sparse.linalg
#import skimage
#import skimage.exposure
#import skimage.filter.rank
#import skimage.morphology
#import skimage.util
#import sklearn.preprocessing
#from PIL import Image

##------------------------------------------------------------------------------
#from hotspotter.Parallelize import parallel_compute
#from hotspotter.Printable import DynStruct
## Hotspotter Imports
#import hotspotter.draw_func2 as df2
#import hotspotter.Pref as Pref
#import hotspotter.algos as algos
#import hotspotter.fileio as io
#import hotspotter.gui as gui
#import hotspotter.tpl.extern_feat as extern_feat
#import hotspotter.params as params
#import hotspotter.helpers as helpers
## Main Pipeline
#import hotspotter.load_data2 as ld2
#import hotspotter.chip_compute2 as cc2
#import hotspotter.feature_compute2 as fc2
#import hotspotter.match_chips2 as mc2
#import hotspotter.spatial_verification2 as sv2
#import hotspotter.report_results2 as rr2

##------------------------------------------------------------------------------
## Helper functions
#__version__ = '1.9.9+'+repr(np.complex(0,.001))
#__author__  = 'Jon Crall'
#__email__   = 'hotspotter.ir@gmail.com'

#if not '__file__' in vars():
    #__file__ = realpath('../hotspotter/__init__.py')
#HSDIR = dirname(__file__)

#def reload_module():
    #import imp
    #import sys
    #imp.reload(sys.modules[__name__])

#def hotspotter_modulenames():
    #modpath_list = helpers.glob(HSDIR, '*.py')
    #def just_name(path):
        #return os.path.splitext(os.path.split(path)[1])[0]
    #modname_list = [just_name(path) for path in modpath_list]
    #return modname_list

#def all_hots_imports():
    #explore_list = []
    #hots_modlist, mymods_dict = get_loaded_hotspotter_modules()
    ##for module in non_hots_modlist:
        ##print('not reloading %r ' % (module,))
    #for name, module in mymods_dict['hotspotter']:
        #helpers.reload_module()
        #explore_str = helpers.explore_module(module, maxdepth=1)
        #explore_list.append(explore_str)
    #print '\n'.join(explore_list)

#def get_loaded_modules():
    #hotspotter_modnames = set(hotspotter_modulenames())
    #scientific_modnames = ['numpy', 'scipy', 'matplotlib', 'sklearn', 'PIL']
    #standard_modnames = ['os', 'sys']
    #mymods_dict = {'standard':[], 'hotspotter':[], 'scientific':[], 'other':[]}
    #def modname_isin(names, modname_list):
        #for modname in modname_list:
            #at_front = name.find(modname) == 0
            #at_hotspotter = name.find('hotspotter.'+modname) == 0
            #at_all = name.find(modname) > -1
            #if at_front or at_hotspotter:
                #return True
        #return False
    #for name, module in sys.modules.iteritems():
        #if modname_isin(name, hotspotter_modnames):
            #mymods_dict['hotspotter'].append((name, module))
        #elif modname_isin(name, scientific_modnames): 
            #mymods_dict['scientific'].append((name, module))
        #elif modname_isin(name, standard_modnames): 
            #mymods_dict['standard'].append((name, module))
        #else:
            #mymods_dict['other'].append((name, module))
    #return mymods_dict

#def print_loaded_modules():
    #mymods_dict = get_loaded_modules()
    #for key, modlist in mymods_dict.iteritems():
        #print ('Module Type: '+str(key))
        #mod_names = [name for name, _ in modlist]
        #mod_list  = [modules for _, modules in modlist]
        ## do dot hack
        #mod_names = list(set([name if name.find('.') == -1 else name[0:name.find('.')] for name in mod_names]))
        ##print(helpers.indent('\n'.join(sorted(mod_names))))
        #print(helpers.indent('\n'.join(sorted(mod_names))))
    #for key, modlist in mymods_dict.iteritems():
        #print ('Module  Type: '+str(key))
        #print (' #loaded='+str(len(modlist)))

#def get_loaded_hotspotter_modules():
    #mymods_dict = get_loaded_modules()
    #return mymods_dict['hotspotter'], mymods_dict

#def reload_all_hotspotter_modules():
    #hots_modlist, mymods_dict = get_loaded_hotspotter_modules()
    ##for module in non_hots_modlist:
        ##print('not reloading %r ' % (module,))
    #for name, module in hots_modlist:
        #print('reloading %r ' % (module,))
        #imp.reload(module)

#def rrr():
    #'alias for reload'
    #reload()
            
#def reload2():
    #'Reloads all modules'
    #reload_all_hotspotter_modules()

#def reload():
    #'Reloads all modules'
    #cc2.reload_module()
    #df2.reload_module()
    #ld2.reload_module()
    #mc2.reload_module()
    #rr2.reload_module()
    #sv2.reload_module()
    #sv1.reload_module()
    #Pref.reload_module()
    #algos.reload_module()
    #helpers.reload_module()
    #params.reload_module()
    #report_results2.reload_module()
    #extern_feat.reload_module()

#if __name__ == '__main__':
    #mp.freeze_support()
    #print('Main: __init__.py')
    #print('This script does nothing')
