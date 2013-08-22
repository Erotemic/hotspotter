from __future__ import division
from __init__ import *

def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])


#df2.reset()
print(textwrap.dedent('''

__+----------------------+__
  +~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ +
  !--- * DEV SCRIPT * ---!
  + ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~+
__|______________________|__

                      '''))

def test_data():
    print('\n\n * ================== * ')
    print('Grabbing test data')
    db_dir = ld2.DEFAULT
    hs = ld2.HotSpotter(db_dir)
    qcx = 0
    cx = hs.get_other_cxs(qcx)[0]
    fm, fs, score = hs.get_assigned_matches_to(qcx, cx)
    rchip1 = hs.get_chip(qcx)
    rchip2 = hs.get_chip(cx)
    # Get keypoints
    kpts1 = hs.get_kpts(qcx)
    kpts2 = hs.get_kpts(cx)
    # local noramlize debugging
    chip = hs.get_chip_pil(qcx)
    return locals()

test_locals = test_data()
test_locals_keys = test_locals.keys()
test_locals_types = map(str, map(type, [test_locals[key] for key in test_locals]))
test_locals_info = ''.join(
    ['\n    '+helpers.list_replace(typestr, ["<type '", "'>", "<class '"])+' '+valstr
    for typestr, valstr in zip(test_locals_types, test_locals_keys)])
print('Adding test locals to namespace: \n    '+test_locals_info)
exec(helpers.dict_execstr(test_locals, 'test_locals'))

experiments.param_config1()
print params.param_string()

run_expt_str = helpers.get_exec_src(experiments.run_experiment)

