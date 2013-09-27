import os
dataset_list = '''
LF_ALL
WD_SIVA
GZ
PZ_DanExt_Test
PZ_DanExt_All
Wildebeast_ONLY_MATCHES
PZ_Marianne
Wildebeast
JAG_Kieryn
'''.strip().split('\n')

for dataset in dataset_list:
    os.system('python autodetect.py '+dataset)

