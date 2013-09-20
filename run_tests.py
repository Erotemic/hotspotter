#!/usr/bin/python2.7
import os

dataset_list = []
dataset_list += [' GZ']
dataset_list += [' PZ']
expt_cmd = ['python', 'experiments.py']


def execute(args):
    print('\n==== Execute Command ====')
    cmd = ' '.join(args)
    print(cmd)
    os.system(cmd)
    print('===========================\n')

# run bag of words tests
for dataset in dataset_list:
    execute(expt_cmd+[dataset, 'leave-out', '--bagofwords'])
# run vsmany tests
for dataset in dataset_list:
    execute(expt_cmd+[dataset, '--noviz', '--vsmany'])
# dump results
for dataset in dataset_list:
    execute(expt_cmd+[dataset])
    execute(expt_cmd+[dataset, 'leave-out', '--bagofwords'])
# try this too
for dataset in dataset_list:
    execute(expt_cmd+[dataset, 'leave-out', '--vsmany'])

# dare I? 
#execute(expt_cmd+['philbin'])
#execute(expt_cmd+['--bagofwords', 'OXFORD'])
#execute(expt_cmd+['--vsmany', 'OXFORD'])
#execute(expt_cmd+['--bagofwords', 'leave-out', 'OXFORD'])
#execute(expt_cmd+['--vsmany', 'leave-out', 'OXFORD'])
    
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
