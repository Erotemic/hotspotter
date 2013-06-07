from hotspotter.HotSpotterAPI import HotSpotterAPI 
from hotspotter.algo.spatial_functions import ransac
from hotspotter.other.AbstractPrintable import AbstractManager
from hotspotter.other.ConcretePrintable import DynStruct
from hotspotter.other.helpers import alloc_lists
from hotspotter.other.logger import logdbg, logerr, hsl
from numpy import spacing as eps
from os.path import join
import os, cPickle
import numpy as np
import pylab

print "RUNNING EXPERIMENTS"

hsl.enable_global_logs()

workdir = '/media/SSD_Extra/'
bajo_bonito = workdir+'LF_Bajo_bonito'
optimizas   = workdir+'LF_OPTIMIZADAS_NI_V_E'
westpoint   = workdir+'LF_WEST_POINT_OPTIMIZADAS'

hsdb1 = HotSpotterAPI(bajo_bonito)
hsdb2 = HotSpotterAPI(optimizas)
hsdb3 = HotSpotterAPI(westpoint)

hsdb1.ensure_model()
hsdb2.ensure_model()
hsdb3.ensure_model()

#print hsdb1.query(1)
#res = hsdb2.query(1, hsdb1)
#res.visualize()

#pylab.show() # keep things on screen
