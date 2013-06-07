from hotspotter.HotSpotterAPI import HotSpotterAPI 
from hotspotter.algo.spatial_functions import ransac
from hotspotter.other.AbstractPrintable import AbstractManager
from hotspotter.other.ConcretePrintable import DynStruct
from hotspotter.other.helpers import alloc_lists, Timer
from hotspotter.other.logger import logdbg, logerr, hsl
from numpy import spacing as eps
from os.path import join
import os, cPickle
import numpy as np
import pylab
import sys

print "RUNNING EXPERIMENTS"

hsl.enable_global_logs()

workdir = '/media/SSD_Extra/'
if sys.platform == 'win32':
    workdir = 'D:/data/work/LF_all/'
bajo_bonito = workdir+'LF_Bajo_bonito'
optimizas   = workdir+'LF_OPTIMIZADAS_NI_V_E'
westpoint   = workdir+'LF_WEST_POINT_OPTIMIZADAS'

hsdb1 = HotSpotterAPI(bajo_bonito)
hsdb2 = HotSpotterAPI(optimizas)
hsdb3 = HotSpotterAPI(westpoint)

with Timer() as t:
    hsdb1.ensure_model()
with Timer() as t:
    hsdb2.ensure_model()
with Timer() as t:
    hsdb3.ensure_model()

def query_db_vs_db(hsA, hsB):
    print 'Running '+hsA.get_dbid()+' vs '+hsB.get_dbid()
    query_cxs = hsA.cm.get_valid_cxs()
    total = len(query_cxs)
    for count, qcx in enumerate(query_cxs):
        with Timer() as t:
            print 'Query %d / %d ' % (count, total)
            rr = hsB.qm.cx2_rr(qcx, hsA)


dbvslist =  [(hsdb1, hsdb2),
           (hsdb2, hsdb1),
           (hsdb1, hsdb3),
           (hsdb3, hsdb1),
           (hsdb2, hsdb3),
           (hsdb2, hsdb3)]
for hsA, hsB in dbvslist:
    query_db_vs_db(hsA, hsB)

#print hsdb1.query(1)
#res = hsdb2.query(1, hsdb1)
#res.visualize()

#pylab.show() # keep things on screen
