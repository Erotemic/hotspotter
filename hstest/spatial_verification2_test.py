#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
from hsdev import test_api
from hsviz import viz
import multiprocessing

if __name__ == '__main__':
    # INITIALIZATION CODE
    # For windows
    multiprocessing.freeze_support()
    # Create a HotSpotter API (hs) and GUI backend (back)
    hs = test_api.main(defaultdb='NAUTS', preload=True, app=None)
    # The test api returns a list of interesting chip indexes
    qcx = test_api.get_test_cxs(hs, 1)[0]
    # Convert chip-index in to chip-id

    viz.viz_spatial_verification(hs, qcx)

    exec(viz.df2.present(num_rc=(1, 1), wh=2500))


'''
Some test variables for SV things
fx1_m  = np.array( (1, 2, 3, 4, 5))
x1_m   = np.array( (1, 2, 1, 4, 5))
y1_m   = np.array( (1, 2, 1, 4, 5))
acd1_m = np.array(((1, 1, 1, 1, 1),
                   (0, 0, 0, 0, 0),
                   (1, 1, 1, 1, 1)))

fx2_m  = np.array( (1, 2, 3, 2, 5))
x2_m   = np.array( (1, 2, 1, 4, 5))
y2_m   = np.array( (1, 2, 1, 4, 5))
acd2_m = np.array(((1, 1, 1, 1, 1),
                   (0, 0, 0, 0, 0),
                   (1, 1, 1, 1, 1)))

acd1_m = array([[ 105.65855929,   69.88258445,   50.26711542,   47.0972872, 37.77338979,
                  80.37862456,   65.7670833 ,   52.42491175, 47.73791486,  47.73791486],
                  [  40.25470409,   33.37290799,  -14.38396778,    5.09841855, 8.36304015,
                  9.40799471,   -0.22772558,   21.09104681, 33.6183116 ,   33.6183116 ],
                  [  85.21461723,   38.1541563 ,   49.27567372,   19.63477339, 24.12673413,
                  34.08558994,   35.23499677,   19.37915367, 29.8612585 ,   29.8612585 ]])

acd2_m = array([[ 27.18315876,  40.44774347,  18.83472822,  46.03951988, 25.48597903,
                42.33150267,  34.53070584,  45.37374314, 42.9485725 ,  53.62149774],
                [ 11.08605802,  -7.47303884,  -9.39089399,  -6.87968738, 0.61334048,
                15.89417442, -38.28506581,   5.9434218 , 25.10330357,  28.30194991],
                [ 14.73551714,  16.44658993,  33.51034403,  19.36112975, 39.17426044,
                31.73842067,  27.55071888,  21.49176377, 21.40969283,  23.89992898]])

ai = acd1_m[0][0]
ci = acd1_m[1][0]
di = acd1_m[2][0]

aj = acd2_m[0][0]
cj = acd2_m[1][0]
dj = acd2_m[2][0]

Ai = np.array([[ai,0],[ci,di]])
Aj = np.array([[aj,0],[cj,dj]])

Ah = np.array([(ai, 0, 0),(ci, di, 0), (0,0,1)])
'''
