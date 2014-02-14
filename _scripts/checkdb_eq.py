from __future__ import print_function, division
from os.path import join, normpath, exists, dirname

# on baker street
local_work = "D:/data/work"
remote_work = "Z:/data/work"


def checktext(rpath):
    fpath1 = normpath(join(local_work, rpath))
    fpath2 = normpath(join(remote_work, rpath))

    text1 = open(fpath1, 'r').read()
    text2 = open(fpath2, 'r').read()

    if text1 != text2:
        print('difference in: %r' % rpath)
        from hscom import helpers
        helpers.vd(dirname(fpath1))
        helpers.vd(dirname(fpath2))
        raise Exception('difference')


standard_dbs = ['GZ_ALL', 'HSDB_zebra_with_mothers']
for dbname in standard_dbs:
    chip_rpath = join(dbname, '_hsdb', 'chip_table.csv')
    name_rpath = join(dbname, '_hsdb', 'name_table.csv')
    image_rpath = join(dbname, '_hsdb', 'image_table.csv')

    checktext(chip_rpath)
    checktext(name_rpath)
    checktext(image_rpath)


print('all good')
