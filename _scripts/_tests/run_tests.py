#!/usr/bin/python2.7
import os
import sys
import helpers

dataset_list = []
#dataset_list += [' GZ']
dataset_list += [' PZ']
expt_cmd = ['python', 'experiments.py']

QUIET = False
if '--quiet' in sys.argv:
    QUIET = True

vizcmd = ['--viz']
if '--noviz' in sys.argv:
    vizcmd = ['--noviz']

expt_cmd += vizcmd

def execute(args):
    print('\n==== Execute Command ====')
    cmd = ' '.join(args)
    print(cmd)
    if QUIET:
        rss = helpers.RedirectStdout(); rss.start()
    os.system(cmd)
    if QUIET: 
        rss.stop()
    print('===========================\n')

# run bag of words tests
for dataset in dataset_list:
    execute(expt_cmd+[dataset, 'leave-out', '--bagofwords'])
# run vsmany tests
for dataset in dataset_list:
    execute(expt_cmd+[dataset, '--vsmany'])
# run vsmany tests
for dataset in dataset_list:
    execute(expt_cmd+[dataset, '--vsmany', '--lnratio'])
# run vsmany tests
for dataset in dataset_list:
    execute(expt_cmd+[dataset, '--vsmany', '--ratio'])
# dump results
for dataset in dataset_list:
    execute(expt_cmd+[dataset])
    execute(expt_cmd+[dataset, 'leave-out', '--bagofwords'])
# try this too
#for dataset in dataset_list:
    #execute(expt_cmd+[dataset, 'leave-out', '--vsmany'])

# dare I? 
execute(expt_cmd+['philbin'])
execute(expt_cmd+['--bagofwords', 'OXFORD'])
execute(expt_cmd+['--vsmany', 'OXFORD'])
execute(expt_cmd+['--bagofwords', 'leave-out', 'OXFORD'])
execute(expt_cmd+['--vsmany', 'leave-out', 'OXFORD'])
    
'''
python experiments.py GZ leave-out --bagofwords --noviz
python experiments.py PZ leave-out --bagofwords --noviz
python experiments.py GZ --bagofwords --noviz
python experiments.py PZ --bagofwords --noviz

python experiments.py GZ leave-out --noviz
python experiments.py PZ leave-out --noviz 

python experiments.py GZ --noviz
python experiments.py PZ --noviz
'''
